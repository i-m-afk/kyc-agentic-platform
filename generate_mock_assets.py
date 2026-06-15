import cv2
import numpy as np
import shutil
import os

print("Generating mock video file...")
# Create a 3-second valid MP4 video (30 frames at 10 FPS)
width, height = 224, 224
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_filename = "mock_video.mp4"
out = cv2.VideoWriter(video_filename, fourcc, 10.0, (width, height))

for i in range(30):
    # Background
    frame = np.ones((height, width, 3), dtype=np.uint8) * 128
    # Draw a moving shape (simulating movement/liveness)
    cv2.circle(frame, (50 + i * 4, 112), 25, (0, 255, 0), -1)
    # Draw a simulated hand contour
    cv2.rectangle(frame, (100, 100), (140, 140), (0, 0, 255), -1)
    out.write(frame)

out.release()
print("mock_video.mp4 generated successfully.")

# Map of target files to copy from our generated templates
assets = {
    # Images
    "mock_id_card.png": [
        "alice_smith_card.jpg",
        "jane_doe_id.jpg",
        "john_doe_card.png",
        "bob_miller_card.jpg",
        "charlie_davis_card.jpg"
    ],
    # Videos
    "mock_video.mp4": [
        "alice_smith_live.mp4",
        "jane_doe_live.mp4",
        "john_doe_spoof.mp4",
        "bob_miller_wrong_gesture.mp4",
        "charlie_davis_deepfake_spoof.mp4"
    ]
}

# Perform copy operation
for src, dsts in assets.items():
    if os.path.exists(src):
        for dst in dsts:
            shutil.copy(src, dst)
            print(f"Copied {src} -> {dst}")
    else:
        print(f"Source file {src} not found!")

print("All mock asset files generated and placed successfully!")
