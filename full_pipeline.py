from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MCAP_PATH = "loopclosure.mcap"

INFRA1_TOPIC = "/camera/camera/infra1/image_rect_raw"
ODOM_TOPIC   = "/fmu/out/vehicle_odometry"

# Loop closure
IMG_SAMPLE_EVERY  = 30
MIN_GOOD_MATCHES  = 60
WEAK_STD          = 2.0
STRONG_STD        = 3.0

# Trajectory
ODOM_SAMPLE_EVERY = 20
NUM_MARKERS       = 25


# ─────────────────────────────────────────
# STEP 1: LOOP CLOSURE DETECTION
# ─────────────────────────────────────────
print("=" * 50)
print("STEP 1: Loop Closure Detection")
print("=" * 50)

frames = []
frame_timestamps = []

print("Reading camera frames...")
with open(MCAP_PATH, "rb") as f:
    reader = make_reader(f, decoder_factories=[DecoderFactory()])
    for i, (schema, channel, message, decoded_msg) in enumerate(
        reader.iter_decoded_messages(topics=[INFRA1_TOPIC])
    ):
        if i % IMG_SAMPLE_EVERY != 0:
            continue
        img = np.frombuffer(decoded_msg.data, dtype=np.uint8).reshape(
            decoded_msg.height, decoded_msg.width
        )
        frames.append(img)
        frame_timestamps.append(message.log_time)

print(f"Extracted {len(frames)} frames")

if len(frames) < 5:
    raise RuntimeError("Not enough frames. Check MCAP path/topic.")

orb = cv2.ORB_create(nfeatures=1000)
bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

start_frame = frames[0]
kp_start, des_start = orb.detectAndCompute(start_frame, None)

if des_start is None:
    raise RuntimeError("Could not extract ORB features from start frame.")

print(f"Start frame keypoints: {len(kp_start)}")

good_match_counts = []
for idx, frame in enumerate(frames):
    kp, des = orb.detectAndCompute(frame, None)
    if des is None or len(kp) < 2:
        good_match_counts.append(0)
        continue
    matches = bf.knnMatch(des_start, des, k=2)
    good = [m for pair in matches if len(pair) == 2
            for m, n in [pair] if m.distance < 0.75 * n.distance]
    good_match_counts.append(len(good))
    if idx % 20 == 0:
        print(f"  Frame {idx}/{len(frames)}: {len(good)} good matches")

good_match_counts = np.array(good_match_counts)
ts     = np.array(frame_timestamps, dtype=float)
time_s = (ts - ts[0]) / 1e9

skip          = len(good_match_counts) // 4
search_region = good_match_counts[skip:]
weak_threshold   = np.mean(search_region) + WEAK_STD  * np.std(search_region)
strong_threshold = np.mean(search_region) + STRONG_STD * np.std(search_region)

peak_rel_idx = np.argmax(search_region)
peak_idx     = skip + peak_rel_idx
peak_val     = good_match_counts[peak_idx]

if peak_val > strong_threshold and peak_val > MIN_GOOD_MATCHES:
    loop_result = "LOOP CLOSURE DETECTED"
elif peak_val > weak_threshold and peak_val > MIN_GOOD_MATCHES:
    loop_result = "POSSIBLE LOOP CLOSURE, LOW CONFIDENCE"
else:
    loop_result = "NO LOOP CLOSURE DETECTED"

loop_detected = loop_result != "NO LOOP CLOSURE DETECTED"

print("\nRESULT")
print("------")
print(loop_result)
print(f"Peak frame:       {peak_idx}")
print(f"Peak time:        {time_s[peak_idx]:.1f} s")
print(f"Peak matches:     {peak_val}")
print(f"Weak threshold:   {weak_threshold:.0f}")
print(f"Strong threshold: {strong_threshold:.0f}")
print(f"Min matches:      {MIN_GOOD_MATCHES}")

# Plot match counts
plt.figure(figsize=(12, 5))
plt.plot(time_s, good_match_counts, label="Good ORB matches vs start frame")
plt.axhline(y=weak_threshold,   color="orange", linestyle="--", label=f"Weak ({weak_threshold:.0f})")
plt.axhline(y=strong_threshold, color="red",    linestyle="--", label=f"Strong ({strong_threshold:.0f})")
plt.axhline(y=MIN_GOOD_MATCHES, color="gray",   linestyle=":",  label=f"Min ({MIN_GOOD_MATCHES})")
if loop_detected:
    plt.axvline(x=time_s[peak_idx], color="green", linestyle="--",
                label=f"Candidate at {time_s[peak_idx]:.1f}s")
plt.xlabel("Time (s)")
plt.ylabel("Good ORB matches")
plt.title(f"Visual Loop Closure Detection — {loop_result}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Show match image if loop found
if loop_detected:
    kp_peak, des_peak = orb.detectAndCompute(frames[peak_idx], None)
    matches = bf.knnMatch(des_start, des_peak, k=2)
    good = sorted(
        [m for pair in matches if len(pair) == 2
         for m, n in [pair] if m.distance < 0.75 * n.distance],
        key=lambda x: x.distance
    )
    match_img = cv2.drawMatches(
        start_frame, kp_start, frames[peak_idx], kp_peak,
        good[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(14, 5))
    plt.imshow(match_img, cmap="gray")
    plt.title(f"Start vs loop candidate — t={time_s[peak_idx]:.1f}s, matches={len(good)}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────
# STEP 2: TRAJECTORY PLOT
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2: Trajectory Plotting")
print("=" * 50)

positions = []
with open(MCAP_PATH, "rb") as f:
    reader = make_reader(f, decoder_factories=[DecoderFactory()])
    for i, (schema, channel, message, decoded_msg) in enumerate(
        reader.iter_decoded_messages(topics=[ODOM_TOPIC])
    ):
        if i % ODOM_SAMPLE_EVERY != 0:
            continue
        positions.append(list(decoded_msg.position))

positions_np = np.array(positions)
print(f"Loaded {len(positions_np)} odometry points")

if loop_detected:
    print("Loop closure confirmed — applying linear drift correction.")
    drift      = positions_np[-1] - positions_np[0]
    n          = len(positions_np)
    correction = np.outer(np.linspace(0, 1, n), drift)
    plot_positions = positions_np - correction
    title = f"Drone 3D Trajectory — Drift Corrected ({loop_result})"
else:
    print("No loop closure — plotting raw odometry.")
    plot_positions = positions_np
    title = "Drone 3D Trajectory — Raw Odometry (No Loop Detected)"

xs, ys, zs = plot_positions[:, 0], plot_positions[:, 1], plot_positions[:, 2]

marker_indices = np.linspace(0, len(plot_positions) - 1, NUM_MARKERS + 2, dtype=int)
mid_indices    = marker_indices[1:-1]

fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection="3d")

ax.plot(xs, ys, zs, color="steelblue", linewidth=1, alpha=0.7, label="Path")
ax.scatter(xs[mid_indices], ys[mid_indices], zs[mid_indices],
           color="orange", s=60, label="Waypoints")
ax.scatter(xs[0],  ys[0],  zs[0],  color="green", s=120, label="Start")
ax.scatter(xs[-1], ys[-1], zs[-1], color="red",   s=120, label="End")

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title(title)
ax.legend()
plt.tight_layout()
plt.show()