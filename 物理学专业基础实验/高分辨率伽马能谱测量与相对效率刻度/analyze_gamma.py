import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "assets"
OUT.mkdir(exist_ok=True)


def read_spe(path: Path):
    lines = path.read_text(errors="ignore").splitlines()
    live = real = None
    counts = []
    in_data = False
    expected = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s == "$MEAS_TIM:":
            a = lines[i + 1].split()
            live, real = float(a[0]), float(a[1])
        elif s == "$DATA:":
            lo, hi = map(int, lines[i + 1].split())
            expected = hi - lo + 1
            in_data = True
            continue
        elif s.startswith("$"):
            in_data = False
        elif in_data:
            try:
                counts.append(int(s))
            except ValueError:
                pass
            if expected and len(counts) >= expected:
                in_data = False
    return np.arange(len(counts)), np.asarray(counts, dtype=float), real or live


def fit_peak(ch, y, approx, halfwin=18):
    lo = max(0, int(round(approx - halfwin)))
    hi = min(len(y) - 1, int(round(approx + halfwin)))
    x = ch[lo:hi + 1]
    yy = y[lo:hi + 1]
    # local baseline from endpoints
    edge = max(3, min(6, len(yy) // 5))
    bl = np.mean(yy[:edge])
    bh = np.mean(yy[-edge:])
    baseline = bh + (bl - bh) * (np.cumsum(yy) - yy) / max(np.sum(yy), 1)
    net = yy - baseline
    net_pos = np.maximum(net, 0)
    if np.sum(net_pos) > 0:
        mu = float(np.sum(x * net_pos) / np.sum(net_pos))
        sigma = float(np.sqrt(np.sum((x - mu) ** 2 * net_pos) / np.sum(net_pos)))
    else:
        mu = float(x[np.argmax(yy)])
        sigma = float(max(1.0, halfwin / 4))
    imax = int(np.argmax(net_pos))
    halfmax = net_pos[imax] / 2
    left_idx = imax
    while left_idx > 0 and net_pos[left_idx] >= halfmax:
        left_idx -= 1
    right_idx = imax
    while right_idx < len(net_pos) - 1 and net_pos[right_idx] >= halfmax:
        right_idx += 1
    def interp(i1, i2):
        y1, y2 = net_pos[i1], net_pos[i2]
        if y2 == y1:
            return float(x[i1])
        return float(x[i1] + (halfmax - y1) * (x[i2] - x[i1]) / (y2 - y1))
    left = interp(left_idx, min(left_idx + 1, len(x) - 1))
    right = interp(max(right_idx - 1, 0), right_idx)
    fwhm_ch = right - left if right > left else 2 * math.sqrt(2 * math.log(2)) * sigma
    net_area = float(np.sum(np.maximum(net, 0)))
    return {
        "lo": lo, "hi": hi, "mu": float(mu), "sigma": float(abs(sigma)), "fwhm_ch": float(fwhm_ch),
        "area": net_area, "area_gauss": float(net_area), "max": float(np.max(yy)),
        "baseline_mean": float(np.mean(baseline)), "x": x, "y": yy, "fit": baseline + net_pos,
        "baseline": baseline,
    }


def calib_from_co():
    ch, y, t = read_spe(BASE / "Co60.Spe")
    # The low-channel region contains electronics noise and Compton continuum.
    # Local-prominence inspection gives the two Co-60 full-energy peaks at about
    # channels 3023 and 3433.
    chosen = [3023, 3433]
    fits = [fit_peak(ch, y, p, 35) for p in chosen]
    E = np.array([1.17323, 1.33249])
    C = np.array([fits[0]["mu"], fits[1]["mu"]])
    k, b = np.polyfit(C, E, 1)
    return ch, y, t, fits, k, b


def energy_to_channel(E, k, b):
    return (E - b) / k


def main():
    co_ch, co_y, co_t, co_fits, k, b = calib_from_co()
    eu_ch, eu_y, eu_t = read_spe(BASE / "Eu152.Spe")

    co_fwhm_E = co_fits[1]["fwhm_ch"] * k * 1000
    resolution = co_fwhm_E / 1332.49 * 100
    # peak-compton ratio: 1040-1096 keV plateau average, max of 1332 peak
    e_axis_co = k * co_ch + b
    comp_mask = (e_axis_co >= 1.040) & (e_axis_co <= 1.096)
    pcr = co_fits[1]["max"] / np.mean(co_y[comp_mask])

    eu_E = np.array([1.40801, 1.11212, 0.96401, 0.77887, 0.34428, 0.12178])
    br = np.array([20.57, 13.35, 13.20, 12.70, 26.20, 28.00])
    approx = [energy_to_channel(E, k, b) for E in eu_E]
    windows = [38, 30, 28, 25, 18, 14]
    eu_fits = [fit_peak(eu_ch, eu_y, p, w) for p, w in zip(approx, windows)]
    areas = np.array([f["area"] for f in eu_fits])
    norm_area = areas / br
    # relative to 121.78 keV normalized area
    rel_eff = norm_area / norm_area[-1]

    def eff_model(E, c1, c2, c3):
        lnE = np.log(E)
        return np.exp(c1 + c2 * lnE + c3 * lnE ** 2)

    c3, c2, c1 = np.polyfit(np.log(eu_E), np.log(rel_eff), 2)
    popt = np.array([c1, c2, c3])
    pred_log = popt[0] + popt[1] * np.log(eu_E) + popt[2] * np.log(eu_E) ** 2
    ss_res = float(np.sum((np.log(rel_eff) - pred_log) ** 2))
    ss_tot = float(np.sum((np.log(rel_eff) - np.mean(np.log(rel_eff))) ** 2))
    r2 = 1 - ss_res / ss_tot

    # plots
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    ax.semilogy(e_axis_co * 1000, np.maximum(co_y, 1), lw=0.8)
    ax.set_xlim(0, 1500)
    ax.set_ylim(1, max(co_y[100:]) * 1.4)
    ax.set_xlabel("能量 / keV")
    ax.set_ylabel("计数")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(r"$^{60}$Co 高纯锗 $\gamma$ 能谱")
    for E, f in zip([1173.23, 1332.49], co_fits):
        ax.axvline(E, color="r", ls="--", lw=0.8)
        ax.text(E + 15, f["max"] * 1.05, f"{E:.0f} keV", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "co60_spectrum.png")
    plt.close(fig)

    e_axis_eu = k * eu_ch + b
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    ax.semilogy(e_axis_eu * 1000, np.maximum(eu_y, 1), lw=0.8)
    ax.set_xlim(0, 1500)
    ax.set_ylim(1, max(eu_y[100:]) * 1.4)
    ax.set_xlabel("能量 / keV")
    ax.set_ylabel("计数")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(r"$^{152}$Eu 高纯锗 $\gamma$ 能谱")
    for E in eu_E * 1000:
        ax.axvline(E, color="r", ls="--", lw=0.6)
        ax.text(E + 10, 1.7, f"{E:.0f}", rotation=90, fontsize=7, va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "eu152_spectrum.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=180)
    order = np.argsort(eu_E)
    Eline = np.linspace(min(eu_E) * 0.9, max(eu_E) * 1.08, 300)
    ax.scatter(eu_E[order], rel_eff[order], color="tab:blue", label="实验点")
    ax.plot(Eline, eff_model(Eline, *popt), color="tab:red", label="拟合曲线")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("能量 E / MeV")
    ax.set_ylabel("相对效率 p")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "relative_efficiency.png")
    plt.close(fig)

    # summary markdown/tex snippets
    out = []
    out.append(f"calib k MeV/ch={k:.9g}, b MeV={b:.9g}")
    out.append("Co peaks:")
    for E, f in zip([1.17323, 1.33249], co_fits):
        out.append(f"E={E:.5f} MeV channel={f['mu']:.3f} FWHM_ch={f['fwhm_ch']:.3f} area={f['area']:.1f} max={f['max']:.1f}")
    out.append(f"FWHM_1332_keV={co_fwhm_E:.3f} keV resolution={resolution:.4f}% peak_compton={pcr:.2f}")
    out.append("Eu peaks:")
    for E, B, f, a, na, reff in zip(eu_E, br, eu_fits, areas, norm_area, rel_eff):
        out.append(f"E={E:.5f} Br={B:.2f} approx={energy_to_channel(E,k,b):.2f} channel={f['mu']:.3f} range={f['lo']}-{f['hi']} area={a:.1f} norm={na:.2f} rel_eff={reff:.5f}")
    out.append(f"eff ln p = {popt[0]:.6f} + {popt[1]:.6f} lnE + {popt[2]:.6f} (lnE)^2, R2log={r2:.5f}")
    (OUT / "analysis_summary.txt").write_text("\n".join(out), encoding="utf-8")

if __name__ == "__main__":
    main()
