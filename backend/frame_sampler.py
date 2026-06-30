import decord
from decord import VideoReader, cpu, gpu
from PIL import Image
import numpy as np
import torch

class FrameSampler:
    def __init__(self, fps_sample=0.5, use_gpu=False):
        """
        fps_sample=0.5 means 1 frame every 2 seconds.
        fps_sample=1.0 means 1 frame per second.

        Why not sample every frame?
        - A 10-min video at 30fps = 18,000 frames
        - At 1fps = 600 frames (still a lot)
        - At 0.5fps = 300 frames (good balance of coverage vs speed)
        - Adjacent frames in video are nearly identical — no info gain
        """
        self.fps_sample = fps_sample
        self.ctx = cpu(0)

    def sample(self, video_path: str) -> list[dict]:
        """
        Extract frames from video at sample rate.

        Returns:
            List of dicts: [{"frame_idx": 0, "timestamp": 0.0, "image": PIL.Image}, ...]
        """
        vr = VideoReader(video_path, ctx=self.ctx)

        total_frames = len(vr)
        video_fps = vr.get_avg_fps()
        duration = total_frames / video_fps

        # Calculate which frame indices to extract
        sample_interval = int(video_fps / self.fps_sample)  # frames between samples
        frame_indices = list(range(0, total_frames, sample_interval))

        print(f"Video: {duration:.1f}s at {video_fps:.1f}fps → {len(frame_indices)} frames to sample")

        # Batch decode (much faster than one-by-one)
        frames_raw = vr.get_batch(frame_indices).asnumpy()  # shape: (N, H, W, 3)

        results = []
        for i, (frame_idx, frame_np) in enumerate(zip(frame_indices, frames_raw)):
            timestamp = frame_idx / video_fps
            image = Image.fromarray(frame_np.astype(np.uint8))
            results.append({
                "frame_idx": frame_idx,
                "timestamp": round(timestamp, 2),
                "image": image
            })

        return results


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "test_video1.mp4"

    sampler = FrameSampler(fps_sample=0.5)
    frames = sampler.sample(video)

    print(f"\nExtracted {len(frames)} frames")
    print(f"First frame: {frames[0]['timestamp']}s")
    print(f"Last frame: {frames[-1]['timestamp']}s")
    print(f"Frame size: {frames[0]['image'].size}")