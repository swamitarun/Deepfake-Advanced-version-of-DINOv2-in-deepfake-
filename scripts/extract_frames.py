"""
extract_frames.py — Extract frames from videos for training.

Usage:
    python scripts/extract_frames.py
    python scripts/extract_frames.py --num-frames 32
    python scripts/extract_frames.py --video-dir data/videos --output-dir data/raw

Expected structure:
    data/videos/
        real/
            video001.mp4
            video002.avi
        fake/
            video001.mp4
            video002.avi

Output structure:
    data/raw/
        real/
            video001_frame_0000.jpg
            video001_frame_0001.jpg
            ...
        fake/
            video001_frame_0000.jpg
            ...

WHY uniform sampling: We extract frames evenly spaced across the video
duration. This ensures we capture content from the beginning, middle, and
end — avoiding bias toward any particular segment. Deepfake artifacts may
vary across the video, so uniform coverage is important.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.video_loader import VideoFrameExtractor
from src.utils.helpers import load_config, setup_logging

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}


def _process_single_video(args):
    """Helper for multiprocessing so it can be pickled."""
    video_path, dst_dir, num_frames = args
    try:
        extractor = VideoFrameExtractor(num_frames=num_frames)
        frames = extractor.extract(video_path=str(video_path), output_dir=str(dst_dir))
        return len(frames), None
    except Exception as e:
        return 0, str(e)


def extract_all_frames(
    video_dir: str,
    output_dir: str,
    num_frames: int = 16,
):
    """
    Extract frames from all videos in real/ and fake/ subdirectories.
    Runs in PARALLEL using multiprocessing.
    """
    total_videos = 0
    total_frames = 0
    max_workers = 16  # Use up to 16 cores for fast extraction

    print("\n" + "=" * 60)
    print("  FAST VIDEO FRAME EXTRACTION (MULTIPROCESSING)")
    print("=" * 60)
    print(f"  Source:     {video_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Frames/vid: {num_frames}")
    print(f"  CPU Cores:  {max_workers}\n")

    for class_name in ['real', 'fake']:
        src_dir = Path(video_dir) / class_name
        dst_dir = Path(output_dir) / class_name

        if not src_dir.exists():
            continue

        os.makedirs(dst_dir, exist_ok=True)
        video_files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS])

        if not video_files:
            continue

        print(f"  Processing {len(video_files)} {class_name} videos...")
        
        tasks = [(v, dst_dir, num_frames) for v in video_files]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # list(tqdm(..., total=length)) wraps the results into a progress bar
            results = list(tqdm(
                executor.map(_process_single_video, tasks), 
                total=len(tasks), 
                desc=f"  [{class_name}]", 
                ncols=100
            ))
            
            for frames_count, err in results:
                if err:
                    pass
                else:
                    total_frames += frames_count
                    total_videos += 1

    print(f"\n  ✅ Extracted {total_frames} frames from {total_videos} videos")
    print(f"     Saved to: {output_dir}/")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--video-dir', type=str, default=None,
                       help='Video source directory (default: data/videos)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Frame output directory (default: data/raw)')
    parser.add_argument('--num-frames', type=int, default=None,
                       help='Frames to extract per video (default: 16)')
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(log_dir=config['paths']['log_dir'])

    video_dir = args.video_dir or 'data/videos'
    output_dir = args.output_dir or config['data']['raw_dir']
    num_frames = args.num_frames or config['video']['num_frames']

    extract_all_frames(
        video_dir=video_dir,
        output_dir=output_dir,
        num_frames=num_frames,
    )


if __name__ == '__main__':
    main()
