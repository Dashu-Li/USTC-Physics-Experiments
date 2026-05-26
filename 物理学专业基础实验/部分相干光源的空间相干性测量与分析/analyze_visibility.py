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


def smooth(profile: np.ndarray, width: int = 21) -> np.ndarray:
    kernel = np.ones(width) / width
    return np.convolve(profile, kernel, mode="same")


def local_visibility(g: np.ndarray, x0: int, x1: int, y0: int, y1: int, win: int = 120, step: int = 20):
    """由水平一维平均强度剖面计算局部可见度。

    先对 ROI 内每一列沿 y 方向平均，再平滑；在滑动窗口内用 90% 与 10%
    分位数近似明暗纹强度，以降低孤立热像素和纯黑背景对 Michelson 公式的影响。
    """
    profile = smooth(np.mean(g[y0:y1, :], axis=0), width=25)
    bg = float(np.percentile(profile, 20))
    xs, vis, mean_int = [], [], []
    for xc in range(x0 + win // 2, x1 - win // 2 + 1, step):
        seg = profile[xc - win // 2:xc + win // 2]
        mean = float(np.mean(seg))
        if mean <= bg + 0.01:
            continue
        imax = float(np.percentile(seg, 90))
        imin = float(np.percentile(seg, 10))
        v = (imax - imin) / (imax + imin) if (imax + imin) > 0 else np.nan
        xs.append(xc)
        vis.append(v)
        mean_int.append(mean)
    return np.array(xs), np.array(vis), np.array(mean_int), profile

summary_rows = []
profile_rows = []

fig, axes = plt.subplots(3, 2, figsize=(10, 10), dpi=160, constrained_layout=True)
axes = axes.ravel()

for ax, (fname, label) in zip(axes, IMAGES):
    g = load_intensity(IMG_DIR / fname)
    x0, x1, y0, y1 = detect_roi(g)
    xs, vis, mean_int, profile = local_visibility(g, x0, x1, y0, y1)
    if len(xs) == 0:
        continue
    width = x1 - x0
    left_mask = xs < x0 + width / 3
    mid_mask = (xs >= x0 + width / 3) & (xs < x0 + 2 * width / 3)
    right_mask = xs >= x0 + 2 * width / 3

    def avg(mask):
        return float(np.nanmean(vis[mask])) if np.any(mask) else float("nan")

    v_left, v_mid, v_right = avg(left_mask), avg(mid_mask), avg(right_mask)
    valid = np.isfinite(vis)
    v_max = float(np.nanmax(vis[valid]))
    x_at_max = int(xs[np.nanargmax(vis)])

    summary_rows.append({
        "image": fname,
        "label": label,
        "roi_x0": x0,
        "roi_x1": x1,
        "roi_y0": y0,
        "roi_y1": y1,
        "V_left": v_left,
        "V_middle": v_mid,
        "V_right": v_right,
        "V_max": v_max,
        "x_at_V_max": x_at_max,
        "mean_intensity": float(np.mean(g[y0:y1, x0:x1])),
        "valid_width_pixel": int(xs[-1] - xs[0]) if len(xs) > 1 else 0,
    })
    for x, v, m in zip(xs, vis, mean_int):
        profile_rows.append({"image": fname, "label": label, "x_pixel": int(x), "visibility": float(v), "mean_intensity": float(m)})

    ax.plot(xs, vis, marker="o", ms=2, lw=1)
    ax.axvline(x0 + width / 3, color="gray", lw=0.8, ls="--")
    ax.axvline(x0 + 2 * width / 3, color="gray", lw=0.8, ls="--")
    ax.set_title(f"{label[-2]} {fname[-18:-4]}")
    ax.set_xlabel("x / pixel")
    ax.set_ylabel("V")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

fig.suptitle("Local fringe visibility versus horizontal pixel position", fontsize=14)
fig.savefig(OUT_DIR / "visibility_profiles.png")
plt.close(fig)

fig, axes = plt.subplots(3, 2, figsize=(10, 10), dpi=160, constrained_layout=True)
axes = axes.ravel()
for ax, (fname, label) in zip(axes, IMAGES):
    g = load_intensity(IMG_DIR / fname)
    x0, x1, y0, y1 = detect_roi(g)
    _, _, _, profile = local_visibility(g, x0, x1, y0, y1)
    ax.plot(profile, lw=1)
    ax.axvspan(x0, x1, color="orange", alpha=0.15)
    ax.set_title(f"{label[-2]} horizontal intensity profile")
    ax.set_xlabel("x / pixel")
    ax.set_ylabel("mean red value")
    ax.grid(alpha=0.3)
fig.savefig(OUT_DIR / "intensity_profiles.png")
plt.close(fig)

# 生成每张图的 ROI 标注图。
fig, axes = plt.subplots(3, 2, figsize=(10, 12), dpi=160, constrained_layout=True)
axes = axes.ravel()
for ax, (fname, label) in zip(axes, IMAGES):
    g = load_intensity(IMG_DIR / fname)
    x0, x1, y0, y1 = detect_roi(g)
    ax.imshow(g, cmap="gray")
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=1.5))
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
