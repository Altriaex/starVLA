import argparse
import math
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
from PIL import Image, ImageDraw


class TraceProcessor:
    def __init__(
        self,
        checkpoint_path: str,
        target_size: tuple[int, int],
        device: str = "cuda",
        begin_track_step: int = 10,
        redraw_frequency: int = 25,
        num_points: int = 5,
        buffer_size: int = 10,
        window_size: int = 15,
        grid_size: int = 30,
        min_active_distance: float = 1.0,
        retry_without_trace_after: int = 5,
        linewidth: int = 2,
        arrow_length: int = 10,
        arrow_angle: int = 40,
    ) -> None:
        import torch
        from cotracker.predictor import CoTrackerPredictor

        self.torch = torch
        self.cotracker_model = CoTrackerPredictor(checkpoint=checkpoint_path).to(device)
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.begin_track_step = begin_track_step
        self.redraw_frequency = redraw_frequency
        self.num_points = num_points
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.grid_size = grid_size
        self.min_active_distance = min_active_distance
        self.retry_without_trace_after = retry_without_trace_after
        self.linewidth = linewidth
        self.arrow_length = arrow_length
        self.arrow_angle = arrow_angle
        self.target_size = target_size
        self.reset()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TraceProcessor":
        return cls(**config)

    def reset(self) -> None:
        self.step = 0
        self.traced = False
        self.trace_buffer: np.ndarray | None = None
        self.image_buffer: deque[np.ndarray] = deque(maxlen=self.buffer_size)
        self.mask_buffer: deque[np.ndarray | None] = deque(maxlen=self.buffer_size)
        self.last_trace_step = 0

    def process_image(
        self,
        image: Image.Image | np.ndarray,
        mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        source_image = self._coerce_image(image)
        self.step += 1

        image_array = np.asarray(source_image, dtype=np.uint8)
        # image_array: [H, W, 3] -> [1, 3, H, W]
        image_tensor = np.ascontiguousarray(image_array.transpose(2, 0, 1))[None, :]
        self.image_buffer.append(image_tensor)

        if mask is not None:
            target_height, target_width = self.target_size
            if mask.shape != (target_height, target_width):
                pil_mask = Image.fromarray(mask.astype(np.uint8) * 255)
                pil_mask = pil_mask.resize((target_width, target_height), resample=Image.Resampling.NEAREST)
                mask = np.asarray(pil_mask, dtype=bool)
        self.mask_buffer.append(mask)

        if self.step < self.begin_track_step:
            self.traced = False
            overlay_image = source_image.copy()
            has_trace = False
            trace_points = None
        else:
            self._update_trace()
            if self.traced:
                trace_points = self.trace_buffer[-self.window_size :]
                overlay_image = self._visualize_trace(source_image.copy(), trace_points)
                has_trace = True
            else:
                overlay_image = source_image.copy()
                trace_points = None
                has_trace = False

        trace_meta = {
            "step": self.step,
            "num_active_points": 0 if trace_points is None else int(trace_points.shape[1]),
            "buffer_size": self.buffer_size,
            "window_size": self.window_size,
            "traced": self.traced,
            "last_trace_step": self.last_trace_step,
        }
        return {
            "source_image": source_image,
            "overlay_image": overlay_image,
            "has_trace": has_trace,
            "trace_points": trace_points,
            "trace_meta": trace_meta,
        }

    def process_example(
        self,
        example: dict[str, Any],
        image_key: str = "image",
        trace_view_index: int = 0,
    ) -> dict[str, Any]:
        image_value = example[image_key]
        if isinstance(image_value, (list, tuple)):
            image_value = image_value[trace_view_index]
        trace_result = self.process_image(image_value)
        return {
            **example,
            "visual_trace_valid": trace_result["has_trace"],
            "visual_trace_image": trace_result["overlay_image"],
            "visual_trace_points": trace_result["trace_points"],
            "visual_trace_meta": trace_result["trace_meta"],
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "traced": self.traced,
            "last_trace_step": self.last_trace_step,
            "buffered_images": len(self.image_buffer),
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "begin_track_step": self.begin_track_step,
            "redraw_frequency": self.redraw_frequency,
            "num_points": self.num_points,
            "buffer_size": self.buffer_size,
            "window_size": self.window_size,
            "grid_size": self.grid_size,
            "min_active_distance": self.min_active_distance,
            "target_size": self.target_size,
        }

    def _coerce_image(self, image: Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        target_height, target_width = self.target_size
        return pil_image.resize((target_width, target_height), resample=Image.Resampling.BILINEAR)

    def _update_trace(self) -> None:
        if (
            self.step % self.redraw_frequency == 0
            or self.step == self.begin_track_step
            or (not self.traced and self.step - self.last_trace_step > self.retry_without_trace_after)
        ):
            self._run_cotracker()
        elif self.trace_buffer is not None:
            self._update_trace_buffer()

    def _run_cotracker(self) -> None:
        video = np.concatenate(list(self.image_buffer), axis=0)
        # video: [T, 3, H, W] -> [1, T, 3, H, W]
        video_tensor = self.torch.from_numpy(video).unsqueeze(0).float().to(device=self.device)

        frame_height = video.shape[2]
        frame_width = video.shape[3]

        mask = self.mask_buffer[0] if len(self.mask_buffer) > 0 else None
        use_mask = mask is not None and np.count_nonzero(mask) >= self.num_points * 2

        if use_mask:
            # Sample queries from within the mask instead of a dense grid.
            ys, xs = np.where(mask)
            num_queries = self.grid_size * self.grid_size
            if len(xs) < num_queries:
                idx = np.random.choice(len(xs), size=num_queries, replace=True)
            else:
                idx = np.random.choice(len(xs), size=num_queries, replace=False)
            queries = np.stack([
                np.zeros(num_queries, dtype=np.float32),
                xs[idx].astype(np.float32),
                ys[idx].astype(np.float32),
            ], axis=1)[None, :]  # [1, N, 3]
            queries_tensor = self.torch.from_numpy(queries).float().to(device=self.device)
            with self.torch.no_grad():
                pred_tracks, _ = self.cotracker_model(video_tensor, queries=queries_tensor)
        else:
            with self.torch.no_grad():
                pred_tracks, _ = self.cotracker_model(video_tensor, grid_size=self.grid_size)

        trace = self._filter_points(pred_tracks[0].cpu().numpy(), (frame_height, frame_width))
        distance = np.mean(np.sum(np.abs(trace[1:] - trace[:-1]), axis=-1), axis=0)
        active_ids = np.where(distance > self.min_active_distance)[0]
        if active_ids.shape[0] >= self.num_points:
            sampled_ids = np.random.choice(active_ids, size=self.num_points, replace=False)
            self.traced = True
            self.trace_buffer = trace[:, sampled_ids]
            return
        self.traced = False
        self.last_trace_step = self.step

    def _update_trace_buffer(self) -> None:
        assert self.trace_buffer is not None
        points_coord = self.trace_buffer[-1]
        time_index = np.ones((self.num_points, 1), dtype=np.float32) * (self.buffer_size - 2)
        # queries: [num_points, 3] -> (t, x, y)
        queries = np.concatenate([time_index, points_coord], axis=1)[None, :]
        video = np.concatenate(list(self.image_buffer), axis=0)
        # video: [T, 3, H, W] -> [1, T, 3, H, W]
        video_tensor = self.torch.from_numpy(video).unsqueeze(0).float().to(device=self.device)
        queries_tensor = self.torch.from_numpy(queries).float().to(device=self.device)
        with self.torch.no_grad():
            pred_tracks, _ = self.cotracker_model(video_tensor, queries=queries_tensor)
        self.trace_buffer = np.concatenate([self.trace_buffer, pred_tracks[0, -1:].cpu().numpy()], axis=0)

    @staticmethod
    def _filter_points(trace: np.ndarray, img_shape: tuple[int, int]) -> np.ndarray:
        height, width = img_shape
        mask = (
            (trace[..., 0] >= 0)
            & (trace[..., 0] < width)
            & (trace[..., 1] >= 0)
            & (trace[..., 1] < height)
        )
        valid_points_mask = np.all(mask, axis=0)
        return trace[:, valid_points_mask, :]

    def _visualize_trace(self, image: Image.Image, trace: np.ndarray) -> Image.Image:
        if trace.shape[0] < 2 or trace.shape[1] == 0:
            return image

        draw = ImageDraw.Draw(image)
        num_steps_to_trace, num_points, _ = trace.shape
        colors = plt.cm.get_cmap("hsv", num_points + 1)

        if num_points > 1:
            for point_index in range(num_points):
                color = tuple((np.array(colors(point_index)[:3]) * 255).astype(int))
                for step_index in range(num_steps_to_trace - 1):
                    start_point = tuple(trace[step_index, point_index])
                    end_point = tuple(trace[step_index + 1, point_index])
                    draw.line([start_point, end_point], fill=color, width=self.linewidth)

        for point_index in range(num_points):
            final_point = np.array(trace[-1, point_index])
            prev_point = np.array(trace[-2, point_index])
            color = tuple((np.array(colors(point_index)[:3]) * 255).astype(int))
            direction = final_point - prev_point
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm != 0 else direction
            cos_angle = math.cos(math.radians(self.arrow_angle))
            sin_angle = math.sin(math.radians(self.arrow_angle))
            left_wing = np.array(
                [
                    final_point[0] - self.arrow_length * cos_angle * direction[0] + self.arrow_length * sin_angle * direction[1],
                    final_point[1] - self.arrow_length * cos_angle * direction[1] - self.arrow_length * sin_angle * direction[0],
                ]
            )
            right_wing = np.array(
                [
                    final_point[0] - self.arrow_length * cos_angle * direction[0] - self.arrow_length * sin_angle * direction[1],
                    final_point[1] - self.arrow_length * cos_angle * direction[1] + self.arrow_length * sin_angle * direction[0],
                ]
            )
            draw.polygon([tuple(final_point), tuple(left_wing), tuple(right_wing)], fill=color, outline=color)
        return image

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--begin_track_step", type=int, default=10)
    parser.add_argument("--redraw_frequency", type=int, default=25)
    parser.add_argument("--num_points", type=int, default=5)
    parser.add_argument("--buffer_size", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=15)
    parser.add_argument("--grid_size", type=int, default=30)
    parser.add_argument("--min_active_distance", type=float, default=1.0)
    parser.add_argument("--retry_without_trace_after", type=int, default=5)
    parser.add_argument("--linewidth", type=int, default=2)
    parser.add_argument("--arrow_length", type=int, default=10)
    parser.add_argument("--arrow_angle", type=int, default=40)
    parser.add_argument("--target_height", type=int, required=True)
    parser.add_argument("--target_width", type=int, required=True)
    args = parser.parse_args()

    video_path = Path(args.video_path)
    output_path = Path(args.output_path)
    checkpoint_path = Path(args.checkpoint)
    assert video_path.is_absolute(), "--video_path must be an absolute path"
    assert output_path.is_absolute(), "--output_path must be an absolute path"
    assert checkpoint_path.is_absolute(), "--checkpoint must be an absolute path"
    video_path = video_path.resolve()
    output_path = output_path.resolve()
    checkpoint_path = checkpoint_path.resolve()

    processor = TraceProcessor(
        checkpoint_path=str(checkpoint_path),
        device=args.device,
        begin_track_step=args.begin_track_step,
        redraw_frequency=args.redraw_frequency,
        num_points=args.num_points,
        buffer_size=args.buffer_size,
        window_size=args.window_size,
        grid_size=args.grid_size,
        min_active_distance=args.min_active_distance,
        retry_without_trace_after=args.retry_without_trace_after,
        linewidth=args.linewidth,
        arrow_length=args.arrow_length,
        arrow_angle=args.arrow_angle,
        target_size=(args.target_height, args.target_width),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    decoded_input_frames_count = 0
    processed_frame_count = 0
    overlay_frames: list[np.ndarray] = []

    with media.VideoReader(str(video_path)) as reader:
        frame_height, frame_width = reader.shape
        fps = reader.fps
        for frame in reader:
            decoded_input_frames_count += 1
            result = processor.process_image(Image.fromarray(frame, mode="RGB"))
            processed_frame_count += 1
            overlay_frames.append(np.asarray(result["overlay_image"].convert("RGB"), dtype=np.uint8))

    assert decoded_input_frames_count > 0, f"No frames were decoded from {video_path}"
    assert fps is not None, f"Video fps is unavailable: {video_path}"
    media.write_video(str(output_path), overlay_frames, fps=fps)
    output_frame_count = len(overlay_frames)
    assert decoded_input_frames_count == processed_frame_count == output_frame_count

    print(
        "[done] "
        f"video_path={video_path} "
        f"output_path={output_path} "
        f"decoded_input_frames_count={decoded_input_frames_count} "
        f"processed_frame_count={processed_frame_count} "
        f"output_frame_count={output_frame_count} "
        f"fps={fps} "
        f"frame_width={frame_width} "
        f"frame_height={frame_height}"
    )


if __name__ == "__main__":
    main()
