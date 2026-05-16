"""
Track a volleyball in a video file frame-by-frame.

Outputs:
  - CSV of every detected position: frame, time_s, x_pixel, y_pixel, confidence, source
  - (optional) annotated video with ball circled at each frame

Usage:
  python scripts/track_video.py mygame.mp4
  python scripts/track_video.py mygame.mp4 --output-video out.mp4
  python scripts/track_video.py mygame.mp4 --mode tracknet
  python scripts/track_video.py mygame.mp4 --mode cv
"""
import argparse
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(description="Track volleyball position in a video.")
    p.add_argument("input_video", help="Path to input .mp4 / .mov / .avi file")
    p.add_argument("--output-csv",   default="", help="Where to save positions CSV (default: <input>_positions.csv)")
    p.add_argument("--output-video", default="", help="Where to save annotated video (omit to skip)")
    p.add_argument("--mode", default="fusion",
                   choices=["fusion", "tracknet", "cv", "hosted", "local"],
                   help="Detection mode (default: fusion = TrackNetV2 + YOLOv8)")
    p.add_argument("--model-path", default="", help="Path to TrackNetV2 .pth.tar weights (fusion/tracknet mode)")
    p.add_argument("--conf",       type=float, default=0.15, help="Primary confidence threshold (default: 0.15)")
    p.add_argument("--yolo-conf",  type=float, default=0.10, help="YOLO confidence threshold (fusion mode, default: 0.10)")
    p.add_argument("--min-radius", type=int,   default=8,    help="Min ball radius in pixels (cv mode)")
    p.add_argument("--max-radius", type=int,   default=40,   help="Max ball radius in pixels (cv mode)")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input_video):
        print(f"Error: file not found: {args.input_video}")
        sys.exit(1)

    stem         = Path(args.input_video).stem
    output_csv   = args.output_csv   or f"{stem}_positions.csv"
    output_video = args.output_video or ""

    # ── Fusion mode (recommended) ─────────────────────────────────────────────
    if args.mode == "fusion":
        from src.detection.fusion_tracker import process_video
        weights = args.model_path or "models/tracknetv2_volleyball_best.pth.tar"
        if not os.path.isfile(weights):
            print(f"Error: TrackNetV2 weights not found at {weights}")
            print("Download with: python3 scripts/download_tracknet_weights.py")
            sys.exit(1)

        result = process_video(
            video_path=args.input_video,
            tracknet_weights=weights,
            output_video=output_video,
            output_csv=output_csv,
            conf=args.conf,
            yolo_conf=args.yolo_conf,
        )
        print(f"\nDone.")
        print(f"  Coverage : {result['covered']}/{result['total_frames']} "
              f"frames ({result['coverage_pct']:.1f}%)")
        print(f"  Real dets: {result['real_detections']}  "
              f"Interp: {result['interpolated']}")
        if output_csv:
            print(f"  CSV      : {output_csv}")
        if output_video:
            print(f"  Video    : {output_video}")
        return

    # ── TrackNetV2-only mode ──────────────────────────────────────────────────
    if args.mode == "tracknet":
        from src.detection.tracknet_tracker import TrackNetTracker
        weights = args.model_path or "models/tracknetv2_volleyball_best.pth.tar"
        if not os.path.isfile(weights):
            print(f"Error: TrackNetV2 weights not found at {weights}")
            sys.exit(1)
        tracker = TrackNetTracker(weights_path=weights, score_threshold=args.conf)
        _run_realtime(args.input_video, tracker, output_csv, output_video)
        return

    # ── OpenCV mode ───────────────────────────────────────────────────────────
    if args.mode == "cv":
        from src.detection.cv_tracker import CVBallTracker
        tracker = CVBallTracker(
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            conf=args.conf,
        )
        _run_realtime(args.input_video, tracker, output_csv, output_video)
        return

    # ── Roboflow / local YOLO modes ───────────────────────────────────────────
    from src.detection.tracker import BallTracker
    tracker = BallTracker(
        mode=args.mode,
        model_path=args.model_path or None,
        conf=args.conf,
    )
    _run_realtime(args.input_video, tracker, output_csv, output_video)


def _run_realtime(video_path, tracker, output_csv, output_video):
    """Generic real-time loop for single-model trackers."""
    import csv as csv_mod

    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    SOURCE_COLORS = {
        "tracknet": (0, 255, 80),
        "yolo":     (255, 180, 0),
        "fusion":   (0, 220, 255),
        "kalman":   (80, 180, 255),
        "interp":   (160, 100, 255),
        "detected": (0, 255, 0),
    }

    detected = kalman_filled = 0
    with open(output_csv, "w", newline="") as csvfile:
        out = csv_mod.writer(csvfile)
        out.writerow(["frame", "time_s", "x_pixel", "y_pixel", "confidence", "source"])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            dets = tracker.detect_frame(frame)
            best = max(dets, key=lambda d: d["confidence"]) if dets else None

            if best:
                x, y   = int(best["x"]), int(best["y"])
                conf   = best["confidence"]
                source = best.get("source", "detected")
                col    = SOURCE_COLORS.get(source, (0, 255, 0))
                time_s = frame_idx / fps

                out.writerow([frame_idx, f"{time_s:.4f}", x, y, f"{conf:.4f}", source])

                if source == "kalman":
                    kalman_filled += 1
                else:
                    detected += 1

                if writer:
                    cv2.circle(frame, (x, y), 30, col, 3)
                    cv2.circle(frame, (x, y), 6,  col, -1)
                    label = f"{conf:.0%}" if conf > 0 else "~kalman"
                    cv2.putText(frame, label, (x+34, y+10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)

            if writer:
                writer.write(frame)

            frame_idx += 1
            total_hits = detected + kalman_filled
            print(
                f"\r  {frame_idx}/{total} frames  |  "
                f"{detected} detected  {kalman_filled} kalman  "
                f"({total_hits} total)",
                end="", flush=True,
            )

    cap.release()
    if writer:
        writer.release()

    total_hits = detected + kalman_filled
    print(f"\n\nDone.")
    print(f"  CSV            : {output_csv}")
    print(f"  Detected       : {detected} / {frame_idx} frames")
    print(f"  Kalman-filled  : {kalman_filled} frames")
    print(f"  Total coverage : {total_hits} / {frame_idx} ({100*total_hits/max(frame_idx,1):.1f}%)")
    if output_video:
        print(f"  Video          : {output_video}")


if __name__ == "__main__":
    main()
