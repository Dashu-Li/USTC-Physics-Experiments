from pathlib import Path
import csv

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
IMG_DIR = BASE / "5.25"
OUT_DIR = BASE / "processed"
OUT_DIR.mkdir(exist_ok=True)

IMAGES = [
    ("Pic_20260525192628038.png", "图(a)"),
    ("Pic_20260525194646918.png", "图(b)"),
    ("Pic_20260525195435698.png", "图(c)"),
    ("Pic_20260525195448209.png", "图(d)"),
    ("Pic_20260525203147728.png", "图(e)"),
    ("Pic_20260525203201995.png", "图(f)"),
]

ROI_OVERRIDES = {
    # 图(b)条纹倾斜且有效条纹区域较宽，自动 ROI 过窄时会使方向识别偏向局部光斑边缘。
    "Pic_20260525194646918.png": (306, 843, 283, 782),
    # 图(c)为毛玻璃未接电时的散斑图样，干涉条纹不完整，只取局部可见条纹区分析。
    "Pic_20260525195435698.png": (160, 460, 710, 950),
}

ANGLE_OVERRIDES = {
    # 角度为垂直于条纹的投影坐标方向，单位为度。
    "Pic_20260525194646918.png": 75.0,
    "Pic_20260525195435698.png": 105.0,
}

# 取红色通道作为主要强度通道；实验干涉图像以红色条纹为主。
def load_intensity(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    return arr[:, :, 0]


def detect_roi(g: np.ndarray):
    # 去掉少量极亮噪声后，以相对阈值确定有效光斑区域。
    p99 = np.percentile(g, 99.5)
    threshold = max(2.0, 0.18 * p99)
    mask = g > threshold
    ys, xs = np.where(mask)
    if len(xs) < 100:
        h, w = g.shape
        return 0, w, int(0.35 * h), int(0.65 * h)
    x0, x1 = np.percentile(xs, [2, 98]).astype(int)
    y0, y1 = np.percentile(ys, [10, 90]).astype(int)
    # 取中心水平带，避免上下边缘暗场影响。
    cy = (y0 + y1) // 2
    half_h = max(70, int(0.18 * (y1 - y0)))
    y0 = max(0, cy - half_h)
    y1 = min(g.shape[0], cy + half_h)
    return int(x0), int(x1), int(y0), int(y1)


def get_roi(fname: str, g: np.ndarray):
    return ROI_OVERRIDES.get(fname, detect_roi(g))


def get_profile_angle(fname: str, g: np.ndarray, x0: int, x1: int, y0: int, y1: int):
    return ANGLE_OVERRIDES.get(fname, estimate_profile_angle(g, x0, x1, y0, y1))


def smooth(profile: np.ndarray, width: int = 21) -> np.ndarray:
    kernel = np.ones(width) / width
    return np.convolve(profile, kernel, mode="same")


def projected_profile(g: np.ndarray, x0: int, x1: int, y0: int, y1: int, theta_deg: float):
    """沿与条纹近似平行方向平均，得到垂直于条纹方向的投影剖面。"""
    sub = g[y0:y1, x0:x1]
    yy, xx = np.indices(sub.shape)
    xx = xx + x0
    yy = yy + y0
    theta = np.deg2rad(theta_deg)
    coord = xx * np.cos(theta) + yy * np.sin(theta)
    c0 = int(np.floor(np.min(coord)))
    bins = np.rint(coord - c0).astype(int)
    sums = np.bincount(bins.ravel(), weights=sub.ravel())
    counts = np.bincount(bins.ravel())
    valid = counts > 0
    coords = c0 + np.arange(len(sums))[valid]
    profile = sums[valid] / counts[valid]
    return coords.astype(float), profile.astype(float)


def estimate_profile_angle(g: np.ndarray, x0: int, x1: int, y0: int, y1: int):
    """搜索使投影剖面调制度最大的方向，作为垂直于条纹方向的坐标轴。"""
    best_angle, best_score = 0.0, -np.inf
    for angle in np.linspace(-60, 60, 61):
        coords, profile = projected_profile(g, x0, x1, y0, y1, angle)
        if len(profile) < 160:
            continue
        prof = smooth(profile, width=25)
        mean = float(np.mean(prof))
        if mean <= 0:
            continue
        score = float((np.percentile(prof, 90) - np.percentile(prof, 10)) / mean)
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def local_visibility(g: np.ndarray, x0: int, x1: int, y0: int, y1: int, theta_deg: float, win: int = 120, step: int = 20):
    """由垂直于条纹方向的一维平均强度剖面计算局部可见度。

    先在 ROI 内沿条纹近似平行方向平均，再平滑；在滑动窗口内用 90% 与 10%
    分位数近似明暗纹强度，以降低孤立热像素和暗背景对 Michelson 公式的影响。
    """
    coords, raw_profile = projected_profile(g, x0, x1, y0, y1, theta_deg)
    profile = smooth(raw_profile, width=25)
    bg = float(np.percentile(profile, 20))
    ss, vis, mean_int = [], [], []
    for start in range(0, len(profile) - win + 1, step):
        seg = profile[start:start + win]
        mean = float(np.mean(seg))
        if mean <= bg + 0.01:
            continue
        imax = float(np.percentile(seg, 90))
        imin = float(np.percentile(seg, 10))
        v = (imax - imin) / (imax + imin) if (imax + imin) > 0 else np.nan
        ss.append(float(coords[start + win // 2]))
        vis.append(v)
        mean_int.append(mean)
    return np.array(ss), np.array(vis), np.array(mean_int), coords, profile

def main():
    summary_rows = []
    profile_rows = []

    fig, axes = plt.subplots(3, 2, figsize=(10, 10), dpi=160, constrained_layout=True)
    axes = axes.ravel()

    for ax, (fname, label) in zip(axes, IMAGES):
        g = load_intensity(IMG_DIR / fname)
        x0, x1, y0, y1 = get_roi(fname, g)
        theta = get_profile_angle(fname, g, x0, x1, y0, y1)
        ss, vis, mean_int, coords, profile = local_visibility(g, x0, x1, y0, y1, theta)
        if len(ss) == 0:
            continue
        s0, s1 = float(np.min(coords)), float(np.max(coords))
        width = s1 - s0
        left_mask = ss < s0 + width / 3
        mid_mask = (ss >= s0 + width / 3) & (ss < s0 + 2 * width / 3)
        right_mask = ss >= s0 + 2 * width / 3

        def avg(mask):
            return float(np.nanmean(vis[mask])) if np.any(mask) else float("nan")

        v_left, v_mid, v_right = avg(left_mask), avg(mid_mask), avg(right_mask)
        valid = np.isfinite(vis)
        v_max = float(np.nanmax(vis[valid]))
        s_at_max = float(ss[np.nanargmax(vis)])

        summary_rows.append({
            "image": fname,
            "label": label,
            "profile_angle_deg": theta,
            "roi_x0": x0,
            "roi_x1": x1,
            "roi_y0": y0,
            "roi_y1": y1,
            "V_left": v_left,
            "V_middle": v_mid,
            "V_right": v_right,
            "V_max": v_max,
            "s_at_V_max": s_at_max,
            "mean_intensity": float(np.mean(g[y0:y1, x0:x1])),
            "valid_width_pixel": int(ss[-1] - ss[0]) if len(ss) > 1 else 0,
        })
        for s, v, m in zip(ss, vis, mean_int):
            profile_rows.append({"image": fname, "label": label, "profile_angle_deg": theta, "s_pixel": float(s), "visibility": float(v), "mean_intensity": float(m)})

        ax.plot(ss, vis, marker="o", ms=2, lw=1)
        ax.axvline(s0 + width / 3, color="gray", lw=0.8, ls="--")
        ax.axvline(s0 + 2 * width / 3, color="gray", lw=0.8, ls="--")
        ax.set_title(f"{label[-2]} {fname[-18:-4]}")
        ax.set_xlabel("s / pixel")
        ax.set_ylabel("V")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    fig.suptitle("Local fringe visibility versus profile coordinate", fontsize=14)
    fig.savefig(OUT_DIR / "visibility_profiles.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(10, 10), dpi=160, constrained_layout=True)
    axes = axes.ravel()
    for ax, (fname, label) in zip(axes, IMAGES):
        g = load_intensity(IMG_DIR / fname)
        x0, x1, y0, y1 = get_roi(fname, g)
        theta = get_profile_angle(fname, g, x0, x1, y0, y1)
        _, _, _, coords, profile = local_visibility(g, x0, x1, y0, y1, theta)
        ax.plot(coords, profile, lw=1)
        ax.set_title(f"{label[-2]} projected intensity profile")
        ax.set_xlabel("s / pixel")
        ax.set_ylabel("mean red value")
        ax.grid(alpha=0.3)
    fig.savefig(OUT_DIR / "intensity_profiles.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(10, 12), dpi=160, constrained_layout=True)
    axes = axes.ravel()
    for ax, (fname, label) in zip(axes, IMAGES):
        g = load_intensity(IMG_DIR / fname)
        x0, x1, y0, y1 = get_roi(fname, g)
        theta = get_profile_angle(fname, g, x0, x1, y0, y1)
        ax.imshow(g, cmap="gray")
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=1.5))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        length = min(x1 - x0, y1 - y0) * 0.35
        ax.arrow(cx, cy, length * np.cos(np.deg2rad(theta)), length * np.sin(np.deg2rad(theta)),
                 color="cyan", width=2, head_width=18, length_includes_head=True)
        ax.set_title(f"{label[-2]}: ROI")
        ax.axis("off")
    fig.savefig(OUT_DIR / "roi_selection.png")
    plt.close(fig)

    with open(OUT_DIR / "visibility_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(OUT_DIR / "visibility_profiles.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(profile_rows[0].keys()))
        writer.writeheader()
        writer.writerows(profile_rows)

    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
