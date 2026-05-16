"""
╔══════════════════════════════════════════════════════════════╗
║         EDGE PERCEPTION DASHBOARD — KOVA LABS DEMO          ║
║         Visual Loop Closure + Drift-Corrected Trajectory     ║
╚══════════════════════════════════════════════════════════════╝

Panels:
  [A] Live IR feed — ORB keypoints coloured by response intensity
       (small dots, plasma colormap, colorbar shown)
  [B] ORB match count graph
  [C] 3D trajectory — auto-spins, pauses when you grab it, resumes after 3s
  [D] Top-down (X-Y) view — raw vs drift-corrected path, loop closure visible
  [E] Status panel

Usage:
    python dashboard_demo.py

Dependencies:
    pip install mcap mcap-ros2-support opencv-python numpy matplotlib
"""

import time
import threading
from collections import deque

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MP4_PATH   = "/home/rishika/defence/claude/infra1.mp4"
MCAP_PATH  = "/home/rishika/defence/claude/loopclosure.mcap"
ODOM_TOPIC = "/fmu/out/vehicle_odometry"

IMG_SAMPLE_EVERY  = 5
MIN_GOOD_MATCHES  = 60
WEAK_STD          = 2.0
STRONG_STD        = 3.0
ODOM_SAMPLE_EVERY = 5
PLAYBACK_FPS      = 20
FRAME_DELAY_S     = 0.05

# Keypoint display
KP_RADIUS    = 3      # pixels — small so video is readable underneath
KP_THICKNESS = 1      # 1 = thin outline circle; -1 = filled
KP_CMAP      = cm.plasma   # plasma: dark-purple (low) → yellow-white (high)

# ─────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────
BG       = "#0a0d0f"
PANEL_BG = "#0f1418"
BORDER   = "#1e2a35"
ACCENT   = "#00e5ff"
ACCENT2  = "#ff6b35"
GREEN    = "#39ff14"
RED      = "#ff3b3b"
YELLOW   = "#ffd700"
TEXT     = "#c8d8e8"
DIM_TEXT = "#4a6070"

matplotlib.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL_BG,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT,
    "xtick.color":      DIM_TEXT,
    "ytick.color":      DIM_TEXT,
    "text.color":       TEXT,
    "grid.color":       BORDER,
    "grid.linewidth":   0.6,
    "font.family":      "monospace",
})

# ─────────────────────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────────────────────
class State:
    lock = threading.Lock()

    current_frame        = None
    current_frame_time_s = 0.0

    time_series   = deque()
    match_series  = deque()
    weak_thresh   = None
    strong_thresh = None

    loop_detected    = False
    loop_result      = "SCANNING..."
    loop_time_s      = None
    loop_match_count = None
    no_loop_final    = False

    positions_raw            = []
    positions_corrected_live = []

    data_loaded = False
    done        = False

S = State()

# ─────────────────────────────────────────────────────────────
# ORB
# ─────────────────────────────────────────────────────────────
orb = cv2.ORB_create(nfeatures=1000)
bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def draw_kp_intensity(frame_gray, keypoints):
    """
    Overlay ORB keypoints on the grayscale frame as small fixed-radius
    circles coloured by response intensity (plasma colormap).

      Low  response  →  dark purple
      High response  →  bright yellow / white

    Radius is KP_RADIUS pixels (small) so the underlying image stays
    visible. Returns an RGB numpy array.
    """
    out = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    if not keypoints:
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    responses = np.array([kp.response for kp in keypoints], dtype=np.float32)
    r_min, r_max = responses.min(), responses.max()
    span   = r_max - r_min if r_max > r_min else 1.0
    r_norm = (responses - r_min) / span          # [0, 1]

    for kp, rn in zip(keypoints, r_norm):
        rgba = KP_CMAP(float(rn))                # (R, G, B, A) in [0, 1]
        bgr  = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(out, (x, y), KP_RADIUS, bgr, KP_THICKNESS, cv2.LINE_AA)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────
# BACKGROUND READER THREAD
# ─────────────────────────────────────────────────────────────
def reader_thread():
    # ── Pass 1: video frames ─────────────────────────────────
    print(f"Opening video: {MP4_PATH}")
    cap = cv2.VideoCapture(MP4_PATH)
    if not cap.isOpened():
        with S.lock:
            S.loop_result = "ERROR: Cannot open MP4"
            S.done = True
        return

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames @ {video_fps:.1f} fps")

    frames_raw  = []
    frame_times = []
    i = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if i % IMG_SAMPLE_EVERY == 0:
            frames_raw.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
            frame_times.append(i / video_fps)
        i += 1
    cap.release()
    print(f"Extracted {len(frames_raw)} sampled frames")

    if len(frames_raw) < 5:
        with S.lock:
            S.loop_result = "ERROR: Not enough video frames"
            S.done = True
        return

    time_s_arr = np.array(frame_times, dtype=float)

    # ── Pass 2: odometry ─────────────────────────────────────
    print(f"Reading odometry: {MCAP_PATH}")
    positions_all = []
    with open(MCAP_PATH, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for j, (schema, channel, message, decoded_msg) in enumerate(
            reader.iter_decoded_messages(topics=[ODOM_TOPIC])
        ):
            if j % ODOM_SAMPLE_EVERY != 0:
                continue
            positions_all.append(list(decoded_msg.position))

    if len(positions_all) < 2:
        with S.lock:
            S.loop_result = "ERROR: No odometry"
            S.done = True
        return

    positions_np = np.array(positions_all)
    print(f"Loaded {len(positions_np)} odometry points")

    # ── Pass 3: ORB match counts ──────────────────────────────
    kp_start, des_start = orb.detectAndCompute(frames_raw[0], None)
    if des_start is None:
        with S.lock:
            S.loop_result = "ERROR: No ORB features in first frame"
            S.done = True
        return

    print(f"Reference frame: {len(kp_start)} keypoints")
    print("Pre-computing ORB match counts...")

    all_match_counts = []
    for frame in frames_raw:
        kp, des = orb.detectAndCompute(frame, None)
        if des is None or len(kp) < 2:
            all_match_counts.append(0)
            continue
        matches = bf.knnMatch(des_start, des, k=2)
        good = [m for pair in matches if len(pair) == 2
                for m, n in [pair] if m.distance < 0.75 * n.distance]
        all_match_counts.append(len(good))
    all_match_counts = np.array(all_match_counts)

    skip          = len(all_match_counts) // 4
    region        = all_match_counts[skip:]
    weak_thresh   = np.mean(region) + WEAK_STD   * np.std(region)
    strong_thresh = np.mean(region) + STRONG_STD * np.std(region)

    with S.lock:
        S.weak_thresh   = weak_thresh
        S.strong_thresh = strong_thresh

    print(f"Thresholds — weak: {weak_thresh:.0f}  strong: {strong_thresh:.0f}")

    # ── Pass 4: replay ───────────────────────────────────────
    loop_fired = False
    corrected  = None

    for idx, (frame, t_s, match_count) in enumerate(
        zip(frames_raw, time_s_arr, all_match_counts)
    ):
        kp, _ = orb.detectAndCompute(frame, None)
        frame_rgb = draw_kp_intensity(frame, kp)

        if not loop_fired and idx >= skip:
            pv = all_match_counts[skip:idx + 1].max()
            if pv > strong_thresh and pv > MIN_GOOD_MATCHES:
                result = "⬤  LOOP CLOSURE CONFIRMED"
            elif pv > weak_thresh and pv > MIN_GOOD_MATCHES:
                result = "◎  POSSIBLE LOOP CLOSURE"
            else:
                result = "SCANNING..."

            if result != "SCANNING...":
                loop_fired = True
                drift      = positions_np[-1] - positions_np[0]
                n          = len(positions_np)
                correction = np.outer(np.linspace(0, 1, n), drift)
                corrected  = positions_np - correction
                with S.lock:
                    S.loop_detected       = True
                    S.loop_result         = result
                    S.loop_time_s         = t_s
                    S.loop_match_count    = int(match_count)
                print(f"Loop closure at T+{t_s:.1f}s | matches={match_count}")

        odom_frac  = (idx + 1) / len(frames_raw)
        odom_count = max(2, int(odom_frac * len(positions_np)))

        with S.lock:
            S.current_frame        = frame_rgb
            S.current_frame_time_s = t_s
            S.time_series.append(t_s)
            S.match_series.append(int(match_count))
            S.positions_raw = positions_np[:odom_count].tolist()
            S.positions_corrected_live = (
                corrected[:odom_count].tolist()
                if (loop_fired and corrected is not None) else []
            )
            S.data_loaded = True

        time.sleep(FRAME_DELAY_S)

    with S.lock:
        if not S.loop_detected:
            S.no_loop_final = True
            S.loop_result   = "✕  NO LOOP DETECTED"
        S.done = True

    print("Replay complete.")


# ─────────────────────────────────────────────────────────────
# FIGURE LAYOUT
#
#  Left block (cols 0-1):
#    Row 0: [A] camera  [B] graph
#    Row 1: [C] 3D      [D] top-down
#  Col 2 (narrow): colorbar for keypoints (row 0 only)
#  Col 3 (right):  [E] status panel (full height, added as fixed axes)
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 11), facecolor=BG)
fig.canvas.manager.set_window_title("EDGE PERCEPTION SYSTEM — KOVA LABS")

gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    left=0.03, right=0.71,
    top=0.90,  bottom=0.05,
    wspace=0.32, hspace=0.42,
    width_ratios=[1, 1, 0.07],   # narrow col 2 = colorbar
    height_ratios=[1, 1.35],     # bottom row taller for 3D / top-down
)

ax_cam   = fig.add_subplot(gs[0, 0])
ax_cbar  = fig.add_subplot(gs[0, 2])
ax_graph = fig.add_subplot(gs[0, 1])
ax_traj  = fig.add_subplot(gs[1, 0], projection="3d")
ax_top   = fig.add_subplot(gs[1, 1:])        # spans cols 1 + 2

# Status panel — fixed right strip
ax_status = fig.add_axes([0.73, 0.05, 0.25, 0.85])

# ── Title ─────────────────────────────────────────────────────
fig.text(0.5, 0.96,
         "◈  EDGE PERCEPTION SYSTEM  ◈  VISUAL LOOP CLOSURE + DRIFT CORRECTION",
         ha="center", fontsize=13, fontweight="bold", color=ACCENT)
fig.text(0.5, 0.928,
         "NVIDIA Jetson Orin Nano  |  Intel RealSense Infrared  |  ORB Feature Matching",
         ha="center", fontsize=9, color=DIM_TEXT)

# ── Panel labels ──────────────────────────────────────────────
ax_cam.set_title(
    "[ A ]  LIVE IR FEED — ORB KEYPOINTS (plasma = response intensity)",
    color=ACCENT, fontsize=8, pad=5, loc="left"
)
ax_graph.set_title("[ B ]  MATCH SCORE vs. MISSION START",
                   color=ACCENT, fontsize=8, pad=5, loc="left")
ax_traj.set_title(
    "[ C ]  3D TRAJECTORY  (drag=rotate · auto-resumes 3s)",
    color=ACCENT, fontsize=8, pad=5, loc="left"
)
ax_top.set_title("[ D ]  TOP-DOWN VIEW  (X-Y plane · loop closure gap shown)",
                 color=ACCENT, fontsize=8, pad=5, loc="left")
ax_status.set_title("[ E ]  SYSTEM STATUS",
                    color=ACCENT, fontsize=8, pad=5, loc="left")

# ─────────────────────────────────────────────────────────────
# PANEL A — Camera
# ─────────────────────────────────────────────────────────────
ax_cam.set_xticks([]); ax_cam.set_yticks([])
ax_cam.spines[:].set_color(BORDER)

im_obj = ax_cam.imshow(np.zeros((480, 640, 3), dtype=np.uint8), aspect="auto")
cam_time_text = ax_cam.text(
    0.02, 0.97, "T+0.0s", transform=ax_cam.transAxes,
    color=GREEN, fontsize=10, va="top", fontfamily="monospace"
)
cam_kp_text = ax_cam.text(
    0.98, 0.97, "KP: --", transform=ax_cam.transAxes,
    color=ACCENT2, fontsize=10, va="top", ha="right", fontfamily="monospace"
)

# ─────────────────────────────────────────────────────────────
# COLORBAR — ORB response intensity
# ─────────────────────────────────────────────────────────────
ax_cbar.set_xticks([])
ax_cbar.set_yticks([])
ax_cbar.spines[:].set_color(BORDER)

sm   = cm.ScalarMappable(cmap=KP_CMAP, norm=mcolors.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, cax=ax_cbar, orientation="vertical")
cbar.set_ticks([0.0, 0.5, 1.0])
cbar.set_ticklabels(["Low", "Mid", "High"], fontsize=7, color=TEXT)
cbar.outline.set_edgecolor(BORDER)
cbar.ax.tick_params(colors=TEXT, length=3)
cbar.ax.set_title("Resp.", color=DIM_TEXT, fontsize=7, pad=4)

# ─────────────────────────────────────────────────────────────
# PANEL B — Match graph
# ─────────────────────────────────────────────────────────────
ax_graph.set_xlabel("Mission Time (s)", fontsize=8)
ax_graph.set_ylabel("ORB Match Count",  fontsize=8)
ax_graph.tick_params(labelsize=8)
ax_graph.grid(True, alpha=0.3)
ax_graph.spines[:].set_color(BORDER)

line_match, = ax_graph.plot([], [], color=ACCENT, lw=1.5, label="Matches")
line_weak   = ax_graph.axhline(0, color=YELLOW, ls="--", lw=1.0, label="Weak",   alpha=0)
line_strong = ax_graph.axhline(0, color=RED,    ls="--", lw=1.0, label="Strong", alpha=0)
line_min    = ax_graph.axhline(
    MIN_GOOD_MATCHES, color=DIM_TEXT, ls=":", lw=0.8, label=f"Min ({MIN_GOOD_MATCHES})"
)
vline_loop  = ax_graph.axvline(0, color=GREEN, ls="--", lw=1.5, alpha=0, label="Loop")
ax_graph.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=BORDER,
                labelcolor=TEXT, loc="upper left")

# ─────────────────────────────────────────────────────────────
# PANEL C — 3D trajectory (interactive + auto-spin)
# ─────────────────────────────────────────────────────────────
for pane in [ax_traj.xaxis.pane, ax_traj.yaxis.pane, ax_traj.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor(BORDER)
ax_traj.set_xlabel("X (m)", fontsize=8, color=TEXT)
ax_traj.set_ylabel("Y (m)", fontsize=8, color=TEXT)
ax_traj.set_zlabel("Z (m)", fontsize=8, color=TEXT)
ax_traj.tick_params(colors=DIM_TEXT, labelsize=7)

traj_raw_line,  = ax_traj.plot([], [], [], color=DIM_TEXT, lw=0.8, alpha=0.4,
                                label="Raw odom")
traj_corr_line, = ax_traj.plot([], [], [], color=ACCENT,   lw=1.8, alpha=0.9,
                                label="Corrected")
traj_start_3d   = ax_traj.scatter([], [], [], color=GREEN, s=80,  zorder=5)
traj_end_3d     = ax_traj.scatter([], [], [], color=RED,   s=80,  zorder=5)
ax_traj.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=BORDER,
               labelcolor=TEXT, loc="upper left")

# Interaction tracking — stop spinning while user drags
_last_interaction = [0.0]
_RESUME_S         = 3.0

def _on_press(event):
    if event.inaxes == ax_traj:
        _last_interaction[0] = time.time()

def _on_release(event):
    if event.inaxes == ax_traj:
        _last_interaction[0] = time.time()

def _on_motion(event):
    if event.inaxes == ax_traj and event.button is not None:
        _last_interaction[0] = time.time()

fig.canvas.mpl_connect("button_press_event",   _on_press)
fig.canvas.mpl_connect("button_release_event", _on_release)
fig.canvas.mpl_connect("motion_notify_event",  _on_motion)

# ─────────────────────────────────────────────────────────────
# PANEL D — Top-down (X-Y) view
# ─────────────────────────────────────────────────────────────
ax_top.set_facecolor(PANEL_BG)
ax_top.spines[:].set_color(BORDER)
ax_top.set_xlabel("X (m)", fontsize=8)
ax_top.set_ylabel("Y (m)", fontsize=8)
ax_top.tick_params(labelsize=7, colors=DIM_TEXT)
ax_top.grid(True, alpha=0.25)
ax_top.set_aspect("equal", adjustable="datalim")

top_raw_line,  = ax_top.plot([], [], color=DIM_TEXT, lw=0.8, alpha=0.45, label="Raw odom")
top_corr_line, = ax_top.plot([], [], color=ACCENT,   lw=2.0, alpha=0.95, label="Corrected")
top_start_pt,  = ax_top.plot([], [], "o", color=GREEN,  ms=9,  zorder=6, label="Start")
top_end_pt,    = ax_top.plot([], [], "^", color=RED,    ms=9,  zorder=6, label="Current")
top_loop_pt,   = ax_top.plot([], [], "*", color=YELLOW, ms=14, zorder=7, label="Loop pt",
                              alpha=0)
ax_top.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=BORDER,
              labelcolor=TEXT, loc="upper left")
top_status_txt = ax_top.text(
    0.98, 0.97, "SCANNING...", transform=ax_top.transAxes,
    color=YELLOW, fontsize=9, ha="right", va="top", fontfamily="monospace"
)
_loop_chord_added = [False]

# ─────────────────────────────────────────────────────────────
# PANEL E — Status
# ─────────────────────────────────────────────────────────────
ax_status.set_xticks([]); ax_status.set_yticks([])
ax_status.spines[:].set_color(BORDER)

_S = {
    "hdr":       (0.50, 0.96, "SYSTEM ONLINE",    ACCENT,   12, "bold"),
    "d1":        (0.50, 0.93, "─"*30,             BORDER,   8,  "normal"),
    "lbl_mode":  (0.04, 0.89, "MODE",             DIM_TEXT, 9,  "normal"),
    "val_mode":  (0.96, 0.89, "GPS-DENIED NAV",   GREEN,    9,  "normal"),
    "lbl_sns":   (0.04, 0.84, "SENSOR",           DIM_TEXT, 9,  "normal"),
    "val_sns":   (0.96, 0.84, "IR STEREO",        TEXT,     9,  "normal"),
    "lbl_algo":  (0.04, 0.79, "ALGORITHM",        DIM_TEXT, 9,  "normal"),
    "val_algo":  (0.96, 0.79, "ORB + BF-MATCH",   TEXT,     9,  "normal"),
    "lbl_hw":    (0.04, 0.74, "HARDWARE",         DIM_TEXT, 9,  "normal"),
    "val_hw":    (0.96, 0.74, "JETSON ORIN",      TEXT,     9,  "normal"),
    "d2":        (0.50, 0.70, "─"*30,             BORDER,   8,  "normal"),
    "hdr2":      (0.50, 0.67, "LIVE METRICS",     ACCENT2,  10, "bold"),
    "lbl_t":     (0.04, 0.62, "MISSION TIME",     DIM_TEXT, 9,  "normal"),
    "val_t":     (0.96, 0.62, "0.0 s",            TEXT,     9,  "normal"),
    "lbl_fr":    (0.04, 0.57, "FRAMES PROC.",     DIM_TEXT, 9,  "normal"),
    "val_fr":    (0.96, 0.57, "0",                TEXT,     9,  "normal"),
    "lbl_mx":    (0.04, 0.52, "MATCH SCORE",      DIM_TEXT, 9,  "normal"),
    "val_mx":    (0.96, 0.52, "--",               TEXT,     9,  "normal"),
    "lbl_od":    (0.04, 0.47, "ODOM PTS",         DIM_TEXT, 9,  "normal"),
    "val_od":    (0.96, 0.47, "0",                TEXT,     9,  "normal"),
    "d3":        (0.50, 0.42, "─"*30,             BORDER,   8,  "normal"),
    "hdr3":      (0.50, 0.39, "LOOP CLOSURE",     ACCENT2,  10, "bold"),
    "lbl_st":    (0.04, 0.34, "STATUS",           DIM_TEXT, 9,  "normal"),
    "val_st":    (0.50, 0.29, "SCANNING...",      YELLOW,   11, "bold"),
    "lbl_lt":    (0.04, 0.22, "DETECT TIME",      DIM_TEXT, 9,  "normal"),
    "val_lt":    (0.96, 0.22, "--",               TEXT,     9,  "normal"),
    "lbl_lm":    (0.04, 0.17, "PEAK MATCHES",     DIM_TEXT, 9,  "normal"),
    "val_lm":    (0.96, 0.17, "--",               TEXT,     9,  "normal"),
    "lbl_dr":    (0.04, 0.12, "DRIFT CORR.",      DIM_TEXT, 9,  "normal"),
    "val_dr":    (0.96, 0.12, "INACTIVE",         RED,      9,  "normal"),
    "d4":        (0.50, 0.07, "─"*30,             BORDER,   8,  "normal"),
    "foot":      (0.50, 0.03, "◈ KOVA LABS  EDGE PERCEPTION ◈", DIM_TEXT, 8, "normal"),
}

stxt = {}
for k, (x, y, t, c, s, w) in _S.items():
    ha = "center" if x == 0.50 else ("left" if x < 0.5 else "right")
    stxt[k] = ax_status.text(x, y, t, transform=ax_status.transAxes,
                              color=c, fontsize=s, fontweight=w,
                              ha=ha, va="center", fontfamily="monospace")

# ─────────────────────────────────────────────────────────────
# ANIMATION UPDATE
# ─────────────────────────────────────────────────────────────
_flash = [0]

def update(frame_num):
    with S.lock:
        if not S.data_loaded:
            return
        cam_img     = S.current_frame
        t_s         = S.current_frame_time_s
        t_list      = list(S.time_series)
        m_list      = list(S.match_series)
        weak_t      = S.weak_thresh
        strong_t    = S.strong_thresh
        loop_det    = S.loop_detected
        loop_res    = S.loop_result
        loop_t      = S.loop_time_s
        loop_m      = S.loop_match_count
        no_loop_fin = S.no_loop_final
        pos_raw     = list(S.positions_raw)
        pos_corr    = list(S.positions_corrected_live)
        done        = S.done

    n_frames = len(t_list)
    n_odom   = len(pos_raw)
    _flash[0] += 1

    # ── A: camera ────────────────────────────────────────────
    if cam_img is not None:
        im_obj.set_data(cam_img)
        im_obj.set_extent([0, cam_img.shape[1], cam_img.shape[0], 0])
    cam_time_text.set_text(f"T+{t_s:.1f}s")
    if m_list:
        cam_kp_text.set_text(f"KP: {m_list[-1]}")

    # ── B: match graph ────────────────────────────────────────
    if t_list and m_list:
        line_match.set_data(t_list, m_list)
        ax_graph.set_xlim(0, max(t_list) + 1)
        ax_graph.set_ylim(0, max(max(m_list) * 1.2, MIN_GOOD_MATCHES * 1.5, 20))
    if weak_t is not None:
        line_weak.set_ydata([weak_t, weak_t]);       line_weak.set_alpha(0.85)
        line_strong.set_ydata([strong_t, strong_t]); line_strong.set_alpha(0.85)
    if loop_det and loop_t is not None:
        vline_loop.set_xdata([loop_t, loop_t]); vline_loop.set_alpha(0.9)

    # ── C: 3D ─────────────────────────────────────────────────
    if n_odom >= 2:
        raw_np = np.array(pos_raw)
        xr, yr, zr = raw_np[:, 0], raw_np[:, 1], raw_np[:, 2]
        traj_raw_line.set_data_3d(xr, yr, zr)
        traj_start_3d._offsets3d = ([xr[0]], [yr[0]], [zr[0]])

        if pos_corr and len(pos_corr) >= 2:
            cnp = np.array(pos_corr)
            traj_corr_line.set_data_3d(cnp[:, 0], cnp[:, 1], cnp[:, 2])
            traj_end_3d._offsets3d = ([cnp[-1, 0]], [cnp[-1, 1]], [cnp[-1, 2]])
            all_x = np.concatenate([xr, cnp[:, 0]])
            all_y = np.concatenate([yr, cnp[:, 1]])
            all_z = np.concatenate([zr, cnp[:, 2]])
        else:
            traj_end_3d._offsets3d = ([xr[-1]], [yr[-1]], [zr[-1]])
            all_x, all_y, all_z = xr, yr, zr

        pad = 0.5
        ax_traj.set_xlim(all_x.min()-pad, all_x.max()+pad)
        ax_traj.set_ylim(all_y.min()-pad, all_y.max()+pad)
        ax_traj.set_zlim(all_z.min()-pad, all_z.max()+pad)

        # Auto-spin — only when user hasn't touched it recently
        if time.time() - _last_interaction[0] > _RESUME_S:
            ax_traj.view_init(elev=22, azim=(frame_num * 0.4) % 360)

    # ── D: top-down ───────────────────────────────────────────
    if n_odom >= 2:
        raw_np = np.array(pos_raw)
        top_raw_line.set_data(raw_np[:, 0], raw_np[:, 1])
        top_start_pt.set_data([raw_np[0, 0]], [raw_np[0, 1]])

        if pos_corr and len(pos_corr) >= 2:
            cnp = np.array(pos_corr)
            top_corr_line.set_data(cnp[:, 0], cnp[:, 1])
            top_end_pt.set_data([cnp[-1, 0]], [cnp[-1, 1]])

            # Draw the closure chord once (start → current end of corrected path)
            if loop_det and not _loop_chord_added[0]:
                ax_top.plot(
                    [cnp[0, 0], cnp[-1, 0]],
                    [cnp[0, 1], cnp[-1, 1]],
                    "--", color=YELLOW, lw=1.4, alpha=0.75, label="Closure gap"
                )
                top_loop_pt.set_data([cnp[-1, 0]], [cnp[-1, 1]])
                top_loop_pt.set_alpha(1.0)
                _loop_chord_added[0] = True
                ax_top.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=BORDER,
                              labelcolor=TEXT, loc="upper left")
        else:
            top_end_pt.set_data([raw_np[-1, 0]], [raw_np[-1, 1]])

        # Axis limits with padding
        all_xt = list(raw_np[:, 0])
        all_yt = list(raw_np[:, 1])
        if pos_corr:
            cnp = np.array(pos_corr)
            all_xt += list(cnp[:, 0]); all_yt += list(cnp[:, 1])
        pt = max((max(all_xt) - min(all_xt)) * 0.12, 1.0)
        ax_top.set_xlim(min(all_xt)-pt, max(all_xt)+pt)
        ax_top.set_ylim(min(all_yt)-pt, max(all_yt)+pt)

    if loop_det:
        top_status_txt.set_text("LOOP CLOSED ✓"); top_status_txt.set_color(GREEN)
    elif no_loop_fin:
        top_status_txt.set_text("NO LOOP");       top_status_txt.set_color(RED)
    else:
        top_status_txt.set_text("SCANNING" + "." * ((_flash[0] // 8) % 4))

    # ── E: status ─────────────────────────────────────────────
    stxt["val_t"].set_text(f"{t_s:.1f} s")
    stxt["val_fr"].set_text(str(n_frames))
    stxt["val_mx"].set_text(str(m_list[-1]) if m_list else "--")
    stxt["val_od"].set_text(str(n_odom))

    if loop_det:
        fc = GREEN if (_flash[0] // 4) % 2 == 0 else ACCENT
        stxt["val_st"].set_text(loop_res);      stxt["val_st"].set_color(fc)
        stxt["val_lt"].set_text(f"{loop_t:.1f} s" if loop_t else "--")
        stxt["val_lm"].set_text(str(loop_m) if loop_m else "--")
        stxt["val_dr"].set_text("ACTIVE ✓");   stxt["val_dr"].set_color(GREEN)
    elif no_loop_fin:
        stxt["val_st"].set_text("✕  NO LOOP"); stxt["val_st"].set_color(RED)
        stxt["val_lt"].set_text("N/A");         stxt["val_lm"].set_text("N/A")
        stxt["val_dr"].set_text("NOT NEEDED"); stxt["val_dr"].set_color(DIM_TEXT)
    else:
        stxt["val_st"].set_text("SCANNING" + "." * ((_flash[0] // 6) % 4))
        stxt["val_st"].set_color(YELLOW)

    if done:
        stxt["hdr"].set_text("MISSION COMPLETE" + (" ✓" if loop_det else ""))
        stxt["hdr"].set_color(GREEN if loop_det else ACCENT)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║    EDGE PERCEPTION DASHBOARD  —  KOVA LABS      ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"Video : {MP4_PATH}")
    print(f"Odom  : {MCAP_PATH}")
    print()
    print("3D plot controls:")
    print("  Left-click + drag   → rotate  (auto-spin pauses, resumes after 3s)")
    print("  Right-click + drag  → zoom")
    print("  Middle-click + drag → pan")

    thread = threading.Thread(target=reader_thread, daemon=True)
    thread.start()

    anim = FuncAnimation(
        fig, update,
        interval=1000 // PLAYBACK_FPS,
        cache_frame_data=False,
        blit=False,
    )

    plt.show()
    print("Dashboard closed.")