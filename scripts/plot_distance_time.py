"""
Distance vs Time plot for volleyball tracking data.
Cumulative real-world distance (meters) derived from px/m scale estimates
based on court dimensions (9m x 18m) and video frame geometry.
Slope of linear fit = average ball velocity in m/s.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Video frame dimensions (annotated_fusion.mp4)
FRAME_W = 3420
FRAME_H = 2214

# Court crop fractions (from TrackNetTracker — strips browser chrome)
COURT_Y1_FRAC = 0.08
COURT_Y2_FRAC = 0.82

# Derived px/m scales
# X: full frame width = court width (9 m)
# Y: cropped court region height = court length (18 m)
COURT_W_M = 9.0
COURT_L_M = 18.0
PX_PER_M_X = FRAME_W / COURT_W_M                                   # 380 px/m
PX_PER_M_Y = (FRAME_H * (COURT_Y2_FRAC - COURT_Y1_FRAC)) / COURT_L_M  # ~91 px/m

SOURCE_COLORS = {
    "tracknet": "#00FF50",
    "yolo":     "#FFB400",
    "fusion":   "#00DCFF",
    "kalman":   "#50B4FF",
    "interp":   "#A064FF",
    "detected": "#00FF00",
}
DEFAULT_COLOR = "#AAAAAA"


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def compute_velocity_ms(df: pd.DataFrame) -> np.ndarray:
    """Instantaneous speed (m/s) between consecutive detections."""
    dx_m = df["x_pixel"].diff() / PX_PER_M_X
    dy_m = df["y_pixel"].diff() / PX_PER_M_Y
    dt = df["time_s"].diff()
    speed = np.sqrt(dx_m**2 + dy_m**2) / dt
    speed.iloc[0] = np.nan  # no previous point for first frame
    return speed.values


def plot(csv_path: Path, out_path: Path | None = None):
    df = load_csv(csv_path)
    if df.empty:
        print("No data in CSV.")
        return

    df = df.sort_values("frame").reset_index(drop=True)
    t = df["time_s"].values
    speed = compute_velocity_ms(df)
    sources = df["source"].values

    valid = ~np.isnan(speed)
    avg_speed = float(np.nanmean(speed))
    peak_speed = float(np.nanmax(speed))

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#141414")

    # Fill under the curve
    ax.fill_between(t[valid], speed[valid], alpha=0.15, color="#00DCFF", zorder=2)
    ax.plot(t[valid], speed[valid], color="#00DCFF", linewidth=1.5, zorder=3)

    # Scatter colored by source
    for src in pd.unique(sources[valid]):
        mask = valid & (sources == src)
        color = SOURCE_COLORS.get(src, DEFAULT_COLOR)
        ax.scatter(
            t[mask], speed[mask],
            color=color, s=80, zorder=5, label=src,
            edgecolors="white", linewidths=0.4,
        )

    # Average speed line
    ax.axhline(avg_speed, color="#FF4444", linewidth=1.8, linestyle="--", zorder=4,
               label=f"Avg  {avg_speed:.2f} m/s")
    ax.annotate(
        f"  avg {avg_speed:.2f} m/s",
        xy=(t[valid][-1], avg_speed),
        color="#FF4444", fontsize=11, va="bottom", fontweight="bold",
    )

    ax.set_xlabel("Time (s)", color="white", fontsize=13)
    ax.set_ylabel("Speed (m/s)", color="white", fontsize=13)
    ax.set_title(
        f"Ball Velocity vs Time — {csv_path.stem}",
        color="white", fontsize=15, pad=14,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    ax.legend(facecolor="#1E1E1E", edgecolor="#444444", labelcolor="white", fontsize=10, loc="upper left")

    duration = t[-1] - t[0]
    stats = (
        f"Frames:  {len(df)}\n"
        f"Duration:  {duration:.3f} s\n"
        f"Avg speed:  {avg_speed:.2f} m/s\n"
        f"Peak speed:  {peak_speed:.2f} m/s\n"
        f"─────────────────\n"
        f"Scale X:  {PX_PER_M_X:.0f} px/m\n"
        f"Scale Y:  {PX_PER_M_Y:.0f} px/m\n"
        f"(approx — no homography calibration)"
    )
    ax.text(
        0.98, 0.05, stats,
        transform=ax.transAxes, color="white", fontsize=9,
        va="bottom", ha="right",
        bbox=dict(facecolor="#1E1E1E", edgecolor="#444444", boxstyle="round,pad=0.5"),
    )

    plt.tight_layout()

    if out_path is None:
        out_path = csv_path.with_name(csv_path.stem + "_dist_time.png")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "positions_fusion.csv")
    out_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    plot(Path(csv_arg), out_arg)
