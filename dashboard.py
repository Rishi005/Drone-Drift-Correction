"""
╔══════════════════════════════════════════════════════════════╗
║         EDGE PERCEPTION DASHBOARD — KOVA LABS DEMO          ║
║         Visual Loop Closure + Drift-Corrected Trajectory     ║
╚══════════════════════════════════════════════════════════════╝

Run this script to show a live 4-panel dashboard:
  [A] Live infrared camera feed with ORB keypoints drawn
  [B] ORB match count graph building in real-time
  [C] 3D trajectory updating as odometry is processed
  [D] Status panel — system state, stats, loop closure event

Inputs:
  - infra1.mp4  : infrared camera video (replaces MCAP camera topic)
  - .mcap file  : odometry data only (vehicle_odometry topic)

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
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# ─────────────────────────────────────────────────────────────
# CONFIG — edit these to match your setup
# ─────────────────────────────────────────────────────────────

# Camera: MP4 video file (infrared)
MP4_PATH  = "/home/rishika/defence/claude/infra1.mp4"

# Odometry: still read from MCAP
MCAP_PATH  = "/home/rishika/defence/claude/loopclosure.mcap"
ODOM_TOPIC = "/fmu/out/vehicle_odometry"

# Loop closure
# IMG_SAMPLE_EVERY: process 1 out of every N video frames for ORB matching.
# Higher = faster but coarser. Lower = slower but catches closure sooner.
IMG_SAMPLE_EVERY  = 5      # MP4 often has more frames than MCAP topic; tune as needed
MIN_GOOD_MATCHES  = 60
WEAK_STD          = 2.0
STRONG_STD        = 3.0

# Trajectory
ODOM_SAMPLE_EVERY = 5      # Denser for smoother live update
NUM_MARKERS       = 25

# Dashboard
PLAYBACK_FPS      = 20     # Target animation frames per second
FRAME_DELAY_S     = 0.05   # Delay between frame releases (controls replay speed)
                            # Lower = faster playback. Try 0.02 for fast, 0.1 for slow.

# ─────────────────────────────────────────────────────────────
# COLOUR PALETTE  (military/tactical dark theme)
# ─────────────────────────────────────────────────────────────
BG        = "#0a0d0f"
PANEL_BG  = "#0f1418"
BORDER    = "#1e2a35"
ACCENT    = "#00e5ff"      # cyan
ACCENT2   = "#ff6b35"      # orange
GREEN     = "#39ff14"      # neon green
RED       = "#ff3b3b"
YELLOW    = "#ffd700"
TEXT      = "#c8d8e8"
DIM_TEXT  = "#4a6070"

matplotlib.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "xtick.color":       DIM_TEXT,
    "ytick.color":       DIM_TEXT,
    "text.color":        TEXT,
    "grid.color":        BORDER,
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
})

# ─────────────────────────────────────────────────────────────
# SHARED STATE  (populated by background reader thread)
# ─────────────────────────────────────────────────────────────
class State:
    lock = threading.Lock()

    # Camera
    current_frame        = None   # np.ndarray (H, W) uint8
    current_frame_kp     = None   # list of cv2.KeyPoint
    current_frame_time_s = 0.0

    # Match graph
    time_series   = deque()   # float seconds
    match_series  = deque()   # int match counts
    weak_thresh   = None
    strong_thresh = None

    # Loop closure
    loop_detected    = False
    loop_result      = "SCANNING..."
    loop_time_s      = None
    loop_match_count = None

    # Odometry
    positions_raw = []          # list of [x, y, z]
    positions_corrected = []    # filled once loop detected

    # Pipeline control
    data_loaded  = False
    done         = False

S = State()

# ─────────────────────────────────────────────────────────────
# ORB / BFMatcher  (created once)
# ─────────────────────────────────────────────────────────────
orb = cv2.ORB_create(nfeatures=1000)
bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

kp_start  = None
des_start = None
start_frame_global = None

# ─────────────────────────────────────────────────────────────
# BACKGROUND DATA READER THREAD
# ─────────────────────────────────────────────────────────────
def reader_thread():
    global kp_start, des_start, start_frame_global

    # ── Pass 1: load camera frames from MP4 ──────────────────
    print(f"Opening video: {MP4_PATH}")
    cap = cv2.VideoCapture(MP4_PATH)
    if not cap.isOpened():
        with S.lock:
            S.loop_result = "ERROR: Cannot open MP4"
            S.done = True
        print(f"ERROR: Could not open {MP4_PATH}")
        return

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames @ {video_fps:.1f} fps")

    frames_raw  = []
    frame_times = []   # synthetic timestamps in seconds from video position

    i = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if i % IMG_SAMPLE_EVERY == 0:
            # Convert to grayscale for ORB (same as original infrared topic)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frames_raw.append(gray)
            frame_times.append(i / video_fps)   # seconds into video
        i += 1

    cap.release()
    print(f"Extracted {len(frames_raw)} sampled frames from MP4")

    if len(frames_raw) < 5:
        with S.lock:
            S.loop_result = "ERROR: Not enough frames in MP4"
            S.done = True
        return

    time_s_arr = np.array(frame_times, dtype=float)

    # ── Pass 2: load odometry from MCAP ──────────────────────
    print(f"Reading odometry from MCAP: {MCAP_PATH}")
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
            S.loop_result = "ERROR: No odometry in MCAP"
            S.done = True
        return

    positions_np = np.array(positions_all)
    print(f"Loaded {len(positions_np)} odometry points from MCAP")

    # ── Pass 3: compute ORB features on all sampled frames ───
    # First frame is the reference (mission start)
    start_frame_global = frames_raw[0]
    kp_start, des_start = orb.detectAndCompute(start_frame_global, None)

    if des_start is None:
        with S.lock:
            S.loop_result = "ERROR: No ORB features in first frame"
            S.done = True
        return

    print(f"Reference frame keypoints: {len(kp_start)}")
    print("Computing ORB match counts for all frames (pre-pass)...")

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

    # Adaptive thresholds (computed on second half of sequence to skip trivial start)
    skip          = len(all_match_counts) // 4
    search_region = all_match_counts[skip:]
    weak_thresh   = np.mean(search_region) + WEAK_STD  * np.std(search_region)
    strong_thresh = np.mean(search_region) + STRONG_STD * np.std(search_region)

    with S.lock:
        S.weak_thresh   = weak_thresh
        S.strong_thresh = strong_thresh

    print(f"Thresholds — weak: {weak_thresh:.0f}, strong: {strong_thresh:.0f}")
    print("Starting dashboard replay...")

    # ── Pass 4: replay frame-by-frame to dashboard ───────────
    loop_already_fired = False
    corrected = None   # will be set when loop fires

    for idx, (frame, t_s, match_count) in enumerate(
        zip(frames_raw, time_s_arr, all_match_counts)
    ):
        # Draw ORB keypoints onto a colour version of the grayscale frame
        # (convert gray→BGR so cyan keypoints are visible in colour)
        kp, _ = orb.detectAndCompute(frame, None)
        frame_bgr_disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame_kp = cv2.drawKeypoints(
            frame_bgr_disp, kp, None,
            color=(0, 229, 255),   # cyan in BGR
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        # Convert BGR → RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame_kp, cv2.COLOR_BGR2RGB)

        # ── Loop closure check ────────────────────────────────
        if not loop_already_fired and idx >= skip:
            pv = all_match_counts[skip:idx + 1].max()
            if pv > strong_thresh and pv > MIN_GOOD_MATCHES:
                result = "⬤  LOOP CLOSURE CONFIRMED"
            elif pv > weak_thresh and pv > MIN_GOOD_MATCHES:
                result = "◎  POSSIBLE LOOP CLOSURE"
            else:
                result = "SCANNING..."

            if result != "SCANNING..." and not loop_already_fired:
                loop_already_fired = True
                # Linear drift correction across full odometry
                drift      = positions_np[-1] - positions_np[0]
                n          = len(positions_np)
                correction = np.outer(np.linspace(0, 1, n), drift)
                corrected  = positions_np - correction
                with S.lock:
                    S.loop_detected       = True
                    S.loop_result         = result
                    S.loop_time_s         = t_s
                    S.loop_match_count    = int(match_count)
                    S.positions_corrected = corrected.tolist()
                print(f"Loop closure at T+{t_s:.1f}s | matches={match_count}")

        # Proportionally index odometry to current frame position
        odom_frac  = (idx + 1) / len(frames_raw)
        odom_count = max(2, int(odom_frac * len(positions_np)))

        with S.lock:
            S.current_frame        = frame_rgb
            S.current_frame_time_s = t_s
            S.time_series.append(t_s)
            S.match_series.append(int(match_count))
            S.positions_raw        = positions_np[:odom_count].tolist()
            if loop_already_fired and corrected is not None:
                S.positions_corrected_live = corrected[:odom_count].tolist()
            else:
                S.positions_corrected_live = []
            S.data_loaded = True

        time.sleep(FRAME_DELAY_S)

    with S.lock:
        S.done = True
    print("Replay complete.")

# ─────────────────────────────────────────────────────────────
# DASHBOARD LAYOUT
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10), facecolor=BG)
fig.canvas.manager.set_window_title("EDGE PERCEPTION SYSTEM — KOVA LABS")

gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    left=0.04, right=0.97,
    top=0.90,  bottom=0.07,
    wspace=0.28, hspace=0.38
)

ax_cam    = fig.add_subplot(gs[0, 0])   # A: camera feed
ax_graph  = fig.add_subplot(gs[0, 1])   # B: match graph
ax_traj   = fig.add_subplot(gs[1, :2], projection="3d")  # C: 3D trajectory
ax_status = fig.add_subplot(gs[:, 2])   # D: status panel

# Title bar
fig.text(
    0.5, 0.96,
    "◈  EDGE PERCEPTION SYSTEM  ◈  VISUAL LOOP CLOSURE + DRIFT CORRECTION",
    ha="center", va="center",
    fontsize=13, fontweight="bold",
    color=ACCENT, fontfamily="monospace",
)
fig.text(
    0.5, 0.927,
    "NVIDIA Jetson Orin Nano  |  Intel RealSense Infrared  |  ORB Feature Matching",
    ha="center", va="center",
    fontsize=8, color=DIM_TEXT, fontfamily="monospace",
)

# Panel labels
for ax, label in [
    (ax_cam,   "[ A ]  LIVE INFRARED FEED — ORB KEYPOINTS"),
    (ax_graph, "[ B ]  MATCH SCORE vs. MISSION START FRAME"),
    (ax_status,"[ D ]  SYSTEM STATUS"),
]:
    ax.set_title(label, color=ACCENT, fontsize=8, pad=6, loc="left")

ax_traj.set_title("[ C ]  3D TRAJECTORY — REAL-TIME DRIFT CORRECTION", color=ACCENT, fontsize=8, pad=6, loc="left")

# ── Axis A: camera ────────────────────────────────────────────
ax_cam.set_xticks([]); ax_cam.set_yticks([])
ax_cam.spines[:].set_color(BORDER)
im_obj = ax_cam.imshow(np.zeros((480, 640), dtype=np.uint8), cmap="gray",
                       vmin=0, vmax=255, aspect="auto")
cam_time_text = ax_cam.text(
    0.02, 0.97, "T+0.0s", transform=ax_cam.transAxes,
    color=GREEN, fontsize=9, va="top", fontfamily="monospace"
)
cam_kp_text = ax_cam.text(
    0.98, 0.97, "KP: --", transform=ax_cam.transAxes,
    color=ACCENT2, fontsize=9, va="top", ha="right", fontfamily="monospace"
)

# ── Axis B: match graph ───────────────────────────────────────
ax_graph.set_xlabel("Mission Time (s)", fontsize=8)
ax_graph.set_ylabel("ORB Match Count", fontsize=8)
ax_graph.grid(True, alpha=0.3)
ax_graph.spines[:].set_color(BORDER)

line_match,  = ax_graph.plot([], [], color=ACCENT,  linewidth=1.5, label="Matches")
line_weak    = ax_graph.axhline(y=0, color=YELLOW, linestyle="--", linewidth=1,   label="Weak threshold",   alpha=0)
line_strong  = ax_graph.axhline(y=0, color=RED,    linestyle="--", linewidth=1,   label="Strong threshold", alpha=0)
line_min     = ax_graph.axhline(y=MIN_GOOD_MATCHES, color=DIM_TEXT, linestyle=":", linewidth=0.8, label=f"Min ({MIN_GOOD_MATCHES})")
vline_loop   = ax_graph.axvline(x=0, color=GREEN, linestyle="--", linewidth=1.5, alpha=0, label="Loop closure")
ax_graph.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, loc="upper left")

# ── Axis C: 3D trajectory ─────────────────────────────────────
ax_traj.set_facecolor(PANEL_BG)
ax_traj.xaxis.pane.fill = False
ax_traj.yaxis.pane.fill = False
ax_traj.zaxis.pane.fill = False
ax_traj.xaxis.pane.set_edgecolor(BORDER)
ax_traj.yaxis.pane.set_edgecolor(BORDER)
ax_traj.zaxis.pane.set_edgecolor(BORDER)
ax_traj.set_xlabel("X (m)", fontsize=8, color=TEXT)
ax_traj.set_ylabel("Y (m)", fontsize=8, color=TEXT)
ax_traj.set_zlabel("Z (m)", fontsize=8, color=TEXT)
ax_traj.tick_params(colors=DIM_TEXT, labelsize=7)

traj_raw_line,  = ax_traj.plot([], [], [], color=DIM_TEXT,  linewidth=0.8, alpha=0.4, label="Raw odometry")
traj_corr_line, = ax_traj.plot([], [], [], color=ACCENT,    linewidth=1.5, alpha=0.9, label="Corrected path")
traj_start_pt   = ax_traj.scatter([], [], [], color=GREEN,  s=80,  zorder=5)
traj_end_pt     = ax_traj.scatter([], [], [], color=RED,    s=80,  zorder=5)
ax_traj.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT, loc="upper left")

# ── Axis D: status panel ──────────────────────────────────────
ax_status.set_xticks([]); ax_status.set_yticks([])
ax_status.spines[:].set_color(BORDER)

STATUS_LINES = {
    "sys_header":  (0.50, 0.94, "SYSTEM ONLINE", ACCENT,  11, "bold"),
    "div1":        (0.50, 0.91, "─" * 28,         BORDER,  8,  "normal"),

    "lbl_mode":    (0.05, 0.87, "MODE",           DIM_TEXT, 8, "normal"),
    "val_mode":    (0.95, 0.87, "GPS-DENIED NAV", GREEN,    8, "normal"),

    "lbl_sensor":  (0.05, 0.82, "SENSOR",         DIM_TEXT, 8, "normal"),
    "val_sensor":  (0.95, 0.82, "IR STEREO",      TEXT,     8, "normal"),

    "lbl_algo":    (0.05, 0.77, "ALGORITHM",      DIM_TEXT, 8, "normal"),
    "val_algo":    (0.95, 0.77, "ORB + BF-MATCH", TEXT,     8, "normal"),

    "lbl_hw":      (0.05, 0.72, "HARDWARE",       DIM_TEXT, 8, "normal"),
    "val_hw":      (0.95, 0.72, "JETSON ORIN",    TEXT,     8, "normal"),

    "div2":        (0.50, 0.68, "─" * 28,         BORDER,   8, "normal"),
    "lbl_live":    (0.50, 0.65, "LIVE METRICS",   ACCENT2, 8, "bold"),

    "lbl_time":    (0.05, 0.60, "MISSION TIME",   DIM_TEXT, 8, "normal"),
    "val_time":    (0.95, 0.60, "0.0 s",          TEXT,     9, "normal"),

    "lbl_frames":  (0.05, 0.55, "FRAMES PROC.",   DIM_TEXT, 8, "normal"),
    "val_frames":  (0.95, 0.55, "0",              TEXT,     9, "normal"),

    "lbl_matches": (0.05, 0.50, "MATCH SCORE",    DIM_TEXT, 8, "normal"),
    "val_matches": (0.95, 0.50, "--",             TEXT,     9, "normal"),

    "lbl_odom":    (0.05, 0.45, "ODOM POINTS",    DIM_TEXT, 8, "normal"),
    "val_odom":    (0.95, 0.45, "0",              TEXT,     9, "normal"),

    "div3":        (0.50, 0.40, "─" * 28,         BORDER,   8, "normal"),
    "lbl_lc":      (0.50, 0.37, "LOOP CLOSURE",   ACCENT2, 8, "bold"),

    "lbl_status":  (0.05, 0.32, "STATUS",         DIM_TEXT, 8, "normal"),
    "val_status":  (0.50, 0.28, "SCANNING...",    YELLOW,   10, "bold"),

    "lbl_lc_t":    (0.05, 0.22, "DETECT TIME",    DIM_TEXT, 8, "normal"),
    "val_lc_t":    (0.95, 0.22, "--",             TEXT,     8, "normal"),

    "lbl_lc_m":    (0.05, 0.17, "PEAK MATCHES",   DIM_TEXT, 8, "normal"),
    "val_lc_m":    (0.95, 0.17, "--",             TEXT,     8, "normal"),

    "lbl_drift":   (0.05, 0.12, "DRIFT CORR.",    DIM_TEXT, 8, "normal"),
    "val_drift":   (0.95, 0.12, "INACTIVE",       RED,      8, "normal"),

    "div4":        (0.50, 0.07, "─" * 28,         BORDER,   8, "normal"),
    "footer":      (0.50, 0.03, "◈ KOVA LABS  TACTICAL EDGE PERCEPTION ◈",
                    DIM_TEXT, 7, "normal"),
}

status_text_objs = {}
for key, (x, y, txt, color, size, weight) in STATUS_LINES.items():
    ha = "center" if x == 0.50 else ("left" if x < 0.5 else "right")
    status_text_objs[key] = ax_status.text(
        x, y, txt,
        transform=ax_status.transAxes,
        color=color, fontsize=size, fontweight=weight,
        ha=ha, va="center", fontfamily="monospace"
    )

# ─────────────────────────────────────────────────────────────
# ANIMATION UPDATE FUNCTION
# ─────────────────────────────────────────────────────────────
_loop_flash = [0]

def update(frame_num):
    with S.lock:
        if not S.data_loaded:
            return

        # Snapshot shared state
        cam_img    = S.current_frame
        t_s        = S.current_frame_time_s
        t_list     = list(S.time_series)
        m_list     = list(S.match_series)
        weak_t     = S.weak_thresh
        strong_t   = S.strong_thresh
        loop_det   = S.loop_detected
        loop_res   = S.loop_result
        loop_t     = S.loop_time_s
        loop_m     = S.loop_match_count
        pos_raw    = list(S.positions_raw)
        pos_corr   = list(getattr(S, "positions_corrected_live", []))
        done       = S.done

    n_frames = len(t_list)
    n_odom   = len(pos_raw)

    # ── Panel A: camera ───────────────────────────────────────
    if cam_img is not None:
        if len(cam_img.shape) == 2:
            im_obj.set_data(cam_img)
            im_obj.set_cmap("gray")
        else:
            im_obj.set_data(cam_img)
            im_obj.set_cmap(None)
        im_obj.set_extent([0, cam_img.shape[1], cam_img.shape[0], 0])

    cam_time_text.set_text(f"T+{t_s:.1f}s")

    # Count white-ish pixels as proxy for keypoint count (cheap)
    if cam_img is not None and len(m_list) > 0:
        cam_kp_text.set_text(f"KP: {m_list[-1]}")

    # ── Panel B: match graph ──────────────────────────────────
    if t_list and m_list:
        line_match.set_data(t_list, m_list)
        ax_graph.set_xlim(0, max(t_list) + 1)
        ax_graph.set_ylim(0, max(max(m_list) * 1.2, MIN_GOOD_MATCHES * 1.5, 20))

    if weak_t is not None:
        line_weak.set_ydata([weak_t, weak_t])
        line_weak.set_alpha(0.85)
        line_strong.set_ydata([strong_t, strong_t])
        line_strong.set_alpha(0.85)

    if loop_det and loop_t is not None:
        vline_loop.set_xdata([loop_t, loop_t])
        vline_loop.set_alpha(0.9)

    # ── Panel C: trajectory ───────────────────────────────────
    if n_odom >= 2:
        raw_np = np.array(pos_raw)
        xr, yr, zr = raw_np[:, 0], raw_np[:, 1], raw_np[:, 2]
        traj_raw_line.set_data_3d(xr, yr, zr)
        traj_start_pt._offsets3d = ([xr[0]], [yr[0]], [zr[0]])

        if pos_corr and len(pos_corr) >= 2:
            corr_np = np.array(pos_corr)
            xc, yc, zc = corr_np[:, 0], corr_np[:, 1], corr_np[:, 2]
            traj_corr_line.set_data_3d(xc, yc, zc)
            traj_end_pt._offsets3d = ([xc[-1]], [yc[-1]], [zc[-1]])
            all_x = np.concatenate([xr, xc])
            all_y = np.concatenate([yr, yc])
            all_z = np.concatenate([zr, zc])
        else:
            traj_end_pt._offsets3d = ([xr[-1]], [yr[-1]], [zr[-1]])
            all_x, all_y, all_z = xr, yr, zr

        pad = 0.5
        ax_traj.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax_traj.set_ylim(all_y.min() - pad, all_y.max() + pad)
        ax_traj.set_zlim(all_z.min() - pad, all_z.max() + pad)

        # Slowly rotate the 3D view
        ax_traj.view_init(elev=22, azim=(frame_num * 0.4) % 360)

    # ── Panel D: status ───────────────────────────────────────
    status_text_objs["val_time"].set_text(f"{t_s:.1f} s")
    status_text_objs["val_frames"].set_text(str(n_frames))
    status_text_objs["val_matches"].set_text(str(m_list[-1]) if m_list else "--")
    status_text_objs["val_odom"].set_text(str(n_odom))

    if loop_det:
        _loop_flash[0] += 1
        flash_color = GREEN if (_loop_flash[0] // 4) % 2 == 0 else ACCENT
        status_text_objs["val_status"].set_text(loop_res)
        status_text_objs["val_status"].set_color(flash_color)
        status_text_objs["val_status"].set_fontsize(8)
        status_text_objs["val_lc_t"].set_text(f"{loop_t:.1f} s" if loop_t else "--")
        status_text_objs["val_lc_m"].set_text(str(loop_m) if loop_m else "--")
        status_text_objs["val_drift"].set_text("ACTIVE ✓")
        status_text_objs["val_drift"].set_color(GREEN)
    else:
        # Scanning pulse
        _loop_flash[0] += 1
        dots = "." * ((_loop_flash[0] // 6) % 4)
        status_text_objs["val_status"].set_text(f"SCANNING{dots}")
        status_text_objs["val_status"].set_color(YELLOW)

    if done:
        status_text_objs["sys_header"].set_text("MISSION COMPLETE")
        status_text_objs["sys_header"].set_color(GREEN)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║    EDGE PERCEPTION DASHBOARD  —  KOVA LABS      ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"Video  : {MP4_PATH}")
    print(f"Odom   : {MCAP_PATH}")
    print("Starting background data reader...")

    t = threading.Thread(target=reader_thread, daemon=True)
    t.start()

    anim = FuncAnimation(
        fig,
        update,
        interval=1000 // PLAYBACK_FPS,
        cache_frame_data=False,
        blit=False,
    )

    plt.show()
    print("Dashboard closed.")