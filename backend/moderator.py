"""
VideoModerator — uses VS Loss fine-tuned ViT for NSFW detection.
Model: Falconsai ViT backbone fine-tuned with Variance-Stabilized Loss
       (IEEE Signal Processing Letters, 2025 — Rabidas, Malakar et al.)
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification
from decord import VideoReader, cpu
from temporal_filter import TemporalFilter, FrameScore
from report_generator import generate_report


class VideoModerator:
    def __init__(
        self,
        model_path="models/best_model.pt",
        threshold=0.4,
        fps_sample=1.0
    ):
        self.threshold  = threshold
        self.fps_sample = fps_sample
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading VS Loss fine-tuned model from {model_path}...")
        self.model = ViTForImageClassification.from_pretrained(
            "Falconsai/nsfw_image_detection",
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}.")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.temporal_filter = TemporalFilter(
            threshold=threshold,
            min_duration=0.0,
            gap_tolerance=8.0
        )

    def _sample_frames(self, video_path):
        vr       = VideoReader(video_path, ctx=cpu(0))
        total    = len(vr)
        fps      = vr.get_avg_fps()
        duration = total / fps

        interval = max(1, int(fps / self.fps_sample))
        indices  = list(range(0, total, interval))

        print(f"Video: {duration:.1f}s | Sampling {len(indices)} frames...")
        frames_raw = vr.get_batch(indices).asnumpy()

        frames = []
        for idx, frame_np in zip(indices, frames_raw):
            img = Image.fromarray(frame_np.astype(np.uint8)).convert("RGB")
            frames.append({
                "timestamp": round(idx / fps, 2),
                "image": img
            })
        return frames, duration

    def _classify_frames(self, frames, batch_size=32):
        scores = []
        for i in range(0, len(frames), batch_size):
            batch   = frames[i:i + batch_size]
            tensors = torch.stack([
                self.transform(f["image"]) for f in batch
            ]).to(self.device)

            with torch.no_grad():
                logits = self.model(tensors).logits.squeeze(1)
                probs  = torch.sigmoid(logits).cpu().numpy()

            for f, prob in zip(batch, probs):
                label = 1 if prob >= self.threshold else 0
                scores.append(FrameScore(
                    timestamp=f["timestamp"],
                    label=label,
                    confidence=round(float(prob), 4)
                ))

        nsfw_count = sum(1 for s in scores if s.label == 1)
        print(f"Classified {len(scores)} frames | NSFW frames: {nsfw_count}")
        return scores

    def moderate(self, video_path: str) -> dict:
        frames, duration = self._sample_frames(video_path)
        frame_scores     = self._classify_frames(frames)
        violations       = self.temporal_filter.filter(frame_scores)

        print(f"Violations detected: {len(violations)}")

        return generate_report(
            video_path=video_path,
            violations=violations,
            duration=duration,
            model_version="varstab-full-50ep",
            loss_function="VS Loss (IEEE SPL 2025)"
        )


if __name__ == "__main__":
    import sys, json
    video     = sys.argv[1] if len(sys.argv) > 1 else "test_video2.mp4"
    moderator = VideoModerator(threshold=0.4)
    report    = moderator.moderate(video)
    print(json.dumps(report, indent=2))
