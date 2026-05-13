"""
Track a volleyball in a video file frame-by-frame.

Outputs:
  - CSV of every detected position: frame, time_s, x_pixel, y_pixel, confidence
  - (optional) annotated video with ball circled at each frame

Usage:
  python scripts/track_video.py mygame.mp4
  python scripts/track_video.py mygame.mp4 --output-video out.mp4 --frame-skip 2
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.detection.tracker import BallTracker


def parse_args():
    p = argparse.ArgumentParser(description="Track volleyball position in a video.")
    p.add_argument("input_video", help="Path to input .mp4 / .mov / .avi file")
    p.add_argument("--output-csv", default="", help="Where to save positions CSV (default: <input>_positions.csv)")
    p.add_argument("--output-video", default="", help="Where to save annotated video (omit to skip)")
    p.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame (1 = every frame)")
    p.add_argument("--mode", default="hosted", choices=["hosted", "local"], help="Detection mode")
    p.add_argument("--model-path", default="", help="Path to .pt weights (local mode only)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input_video):
        print(f"Error: file not found: {args.input_video}")
        sys.exit(1)

    output_csv = args.output_csv or str(Path(args.input_video).stem) + "_positions.csv"

    model_path = args.model_path or None
    tracker = BallTracker(mode=args.mode, model_path=model_path)

    cap = cv2.VideoCapture(args.input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    detected = 0
    with open(output_csv, "w", newline="") as csvfile:
        out = csv.writer(csvfile)
        out.writerow(["frame", "time_s", "x_pixel", "y_pixel", "confidence"])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % args.frame_skip == 0:
                detections = tracker.detect_frame(frame)

                # Keep the highest-confidence volleyball detection
                best = None
                for d in detections:
                    if best is None or d["confidence"] > best["confidence"]:
                        best = d

                if best:
                    x, y = int(best["x"]), int(best["y"])
                    conf = best["confidence"]
                    time_s = frame_idx / fps
                    out.writerow([frame_idx, f"{time_s:.4f}", x, y, f"{conf:.4f}"])
                    detected += 1

                    if writer:
                        cv2.circle(frame, (x, y), 18, (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{conf:.0%}",
                            (x + 22, y + 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            1,
                        )

            if writer:
                writer.write(frame)

            frame_idx += 1
            print(f"\r  {frame_idx}/{total_frames} frames processed  |  {detected} detections", end="", flush=True)

    cap.release()
    if writer:
        writer.release()

    print(f"\n\nDone.")
    print(f"  Positions CSV : {output_csv}")
    print(f"  Frames with ball detected: {detected} / {frame_idx}")
    if args.output_video:
        print(f"  Annotated video: {args.output_video}")
    if args.mode == "hosted":
        print("\n  Note: hosted mode calls the Roboflow API once per frame — processing is slow for long videos.")
        print("  For faster tracking, download a local .pt model and use --mode local --model-path model.pt")


if __name__ == "__main__":
    main()
