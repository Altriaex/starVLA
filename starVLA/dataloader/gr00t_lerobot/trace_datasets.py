from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
    ModalityConfig,
    safe_hash,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG
from starVLA.model.modules.trace.trace_processor import build_trace_aware_prompt


class LeRobotSingleTraceDataset(LeRobotSingleDataset):
    """Single-dataset traced reader that folds trace semantics back into standard fields."""

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms=None,
        delete_pause_frame: bool = False,
        data_cfg: dict[str, Any] | None = None,
        mode: str = "train",
        seed: int = 42,
        trace_dropout_rate: float = 0.0,
        **kwargs,
    ):
        if not 0.0 <= trace_dropout_rate <= 1.0:
            raise ValueError(f"trace_dropout_rate must be within [0, 1], got {trace_dropout_rate}")

        super().__init__(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            transforms=transforms,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            **kwargs,
        )

        video_keys = self.modality_keys["video"]
        video_key_set = set(video_keys)
        dataset_features = self.lerobot_info_meta["features"]
        self.trace_routes: list[tuple[str, str, str]] = []
        # Discover each trace route from the configured video keys instead of hard-coding *_trace_valid inputs.
        for trace_video_key in video_keys:
            if not trace_video_key.endswith("_trace"):
                continue

            base_video_key = trace_video_key.removesuffix("_trace")
            if base_video_key not in video_key_set:
                raise ValueError(
                    f"Trace video key {trace_video_key} requires paired base video key {base_video_key}"
                )

            trace_original_key = self.lerobot_modality_meta.get_key_meta(trace_video_key).original_key
            if trace_original_key is None:
                raise ValueError(f"Trace video key {trace_video_key} must have an original_key")

            trace_field_name = trace_original_key.split(".")[-1]
            if not trace_field_name.endswith("_trace"):
                raise ValueError(
                    f"{trace_video_key} must map to a trace video field ending with '_trace', got {trace_original_key}"
                )

            trace_valid_column = f"{trace_field_name}_valid"
            trace_valid_meta = dataset_features.get(trace_valid_column)
            if trace_valid_meta is None:
                raise ValueError(
                    f"Trace video key {trace_video_key} requires paired validity column {trace_valid_column}"
                )
            if trace_valid_meta.get("dtype") != "bool":
                raise ValueError(
                    f"Trace validity column {trace_valid_column} must have bool dtype, got {trace_valid_meta.get('dtype')}"
                )

            self.trace_routes.append((base_video_key, trace_video_key, trace_valid_column))

        seen_base_video_keys: set[str] = set()
        for base_video_key, _, _ in self.trace_routes:
            if base_video_key in seen_base_video_keys:
                raise ValueError(f"Duplicate trace route detected for base video key {base_video_key}")
            seen_base_video_keys.add(base_video_key)

        self.mode = mode
        self.seed = seed
        self.trace_dropout_rate = trace_dropout_rate

    def _resolve_final_trace_validity(
        self,
        trajectory_id: int,
        step_index: int,
        initial_trace_validity: list[bool],
    ) -> list[bool]:
        if self.trace_dropout_rate == 0.0:
            return initial_trace_validity

        droppable_route_indices = [index for index, is_valid in enumerate(initial_trace_validity) if is_valid]
        if not droppable_route_indices:
            return initial_trace_validity

        seed_items = (self.dataset_name, int(trajectory_id), int(step_index), self.seed)
        if self.mode == "train":
            seed_items = (self.epoch,) + seed_items
        rng = np.random.default_rng(safe_hash(seed_items))

        if rng.random() >= self.trace_dropout_rate:
            return initial_trace_validity

        final_trace_validity = list(initial_trace_validity)
        dropped_route_index = droppable_route_indices[int(rng.integers(0, len(droppable_route_indices)))]
        final_trace_validity[dropped_route_index] = False
        return final_trace_validity

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict[str, Any]:
        """Read one traced step and map its final trace semantics back to standard fields."""
        raw_data = super().get_step_data(trajectory_id, base_index)

        # Read the traced dataset's initial validity flags and then apply runtime dropout on top of them.
        initial_trace_validity = [
            bool(self.curr_traj_data[trace_valid_column].iloc[base_index])
            for _, _, trace_valid_column in self.trace_routes
        ]
        final_trace_validity = self._resolve_final_trace_validity(
            trajectory_id=trajectory_id,
            step_index=base_index,
            initial_trace_validity=initial_trace_validity,
        )

        # Keep four-view image content and language prompt synchronized to the final trace semantics.
        for route_index, (base_video_key, trace_video_key, _) in enumerate(self.trace_routes):
            if not final_trace_validity[route_index]:
                raw_data[trace_video_key] = np.copy(raw_data[base_video_key])

        # Rewrite the language field so the prompt matches the final four-view trace semantics.
        raw_task_description = raw_data[self.modality_keys["language"][0]][0]
        if not isinstance(raw_task_description, str):
            raw_task_description = raw_task_description.item()
        raw_data[self.modality_keys["language"][0]] = [
            build_trace_aware_prompt(
                task_description=raw_task_description,
                video_keys=self.modality_keys["video"],
                trace_video_pairs=[(base_video_key, trace_video_key) for base_video_key, trace_video_key, _ in self.trace_routes],
                trace_validity=final_trace_validity,
            )
        ]
        return raw_data


if __name__ == "__main__":
    # Parse the fixed step-2 acceptance arguments.
    parser = argparse.ArgumentParser(
        description="4.5 step-2 acceptance entry for LeRobotSingleTraceDataset."
    )
    parser.add_argument("--data-root-dir", required=True)
    parser.add_argument("--data-mix", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--include-state", default="false")
    parser.add_argument("--trace-dropout-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Validate the fixed step-2 acceptance boundary.
    data_root_dir = Path(args.data_root_dir)
    if not data_root_dir.is_absolute():
        raise ValueError(f"--data-root-dir must be an absolute path: {data_root_dir}")

    data_root_dir = data_root_dir.resolve()
    if not data_root_dir.is_dir():
        raise ValueError(f"--data-root-dir must be an existing directory: {data_root_dir}")

    include_state_value = args.include_state.strip().lower()
    if include_state_value not in {"true", "false"}:
        raise ValueError(f"--include-state must be 'true' or 'false', got {args.include_state!r}")
    include_state = include_state_value == "true"

    if args.data_mix != "libero_trace_all":
        raise ValueError(
            "trace_datasets.py __main__ is only the 4.5 step-2 acceptance entry and therefore "
            f"only supports 'libero_trace_all', got {args.data_mix!r}"
        )

    dataset_path = data_root_dir / "libero_goal_no_noops_1.0.0_lerobot_trace"
    if not dataset_path.is_dir():
        raise ValueError(f"Expected traced dataset directory: {dataset_path}")
    if not (dataset_path / "meta/info.json").exists():
        raise FileNotFoundError(f"Traced dataset metadata is missing under {dataset_path}")

    # Build the fixed four-view single dataset for step-2 acceptance only.
    expected_four_view_image_keys = [
        "video.primary_image",
        "video.primary_image_trace",
        "video.wrist_image",
        "video.wrist_image_trace",
    ]
    base_data_config = ROBOT_TYPE_CONFIG_MAP["libero_franka"]
    base_modality_configs = base_data_config.modality_config()
    trace_modality_configs = dict(base_modality_configs)
    trace_modality_configs["video"] = ModalityConfig(
        delta_indices=base_modality_configs["video"].delta_indices,
        modality_keys=expected_four_view_image_keys,
    )
    transforms = base_data_config.transform()
    embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG["libero_franka"]

    traced_single_dataset = LeRobotSingleTraceDataset(
        dataset_path=dataset_path,
        modality_configs=trace_modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend="torchvision_av",
        data_cfg={"include_state": include_state},
        mode="val",
        seed=args.seed,
        trace_dropout_rate=args.trace_dropout_rate,
    )

    trace_dataset = LeRobotMixtureDataset(
        data_mixture=[(traced_single_dataset, 1.0)],
        mode="val",
        seed=args.seed,
        data_cfg={"include_state": include_state},
    )
    sample = trace_dataset[args.sample_index]

    # Validate the traced single-dataset contract and the existing mixture contract.
    source_dataset, trajectory_id, step_index = trace_dataset.sample_step(args.sample_index)
    source_dataset_index = source_dataset.all_steps.index((trajectory_id, step_index))
    direct_sample = source_dataset[source_dataset_index]

    assert isinstance(sample, dict)
    expected_mixture_keys = {"action", "image", "lang", "state"} if include_state else {"action", "image", "lang"}
    assert set(sample) == expected_mixture_keys
    assert source_dataset.modality_keys["video"] == expected_four_view_image_keys
    assert source_dataset.trace_routes == [
        ("video.primary_image", "video.primary_image_trace", "image_trace_valid"),
        ("video.wrist_image", "video.wrist_image_trace", "wrist_image_trace_valid"),
    ]
    assert len(sample["image"]) == len(expected_four_view_image_keys)
    assert set(direct_sample) == {"action", "image", "language"}
    assert len(direct_sample["image"]) == len(expected_four_view_image_keys)

    raw_task_description = source_dataset.get_language(
        trajectory_id,
        source_dataset.modality_keys["language"][0],
        step_index,
    )[0]
    if not isinstance(raw_task_description, str):
        raw_task_description = raw_task_description.item()
    trajectory_data = source_dataset.get_trajectory_data(trajectory_id)
    initial_trace_validity = [
        bool(trajectory_data[trace_valid_column].iloc[step_index])
        for _, _, trace_valid_column in source_dataset.trace_routes
    ]
    final_trace_validity = source_dataset._resolve_final_trace_validity(
        trajectory_id=trajectory_id,
        step_index=step_index,
        initial_trace_validity=initial_trace_validity,
    )
    expected_prompt = build_trace_aware_prompt(
        task_description=raw_task_description,
        video_keys=source_dataset.modality_keys["video"],
        trace_video_pairs=[
            (base_video_key, trace_video_key) for base_video_key, trace_video_key, _ in source_dataset.trace_routes
        ],
        trace_validity=final_trace_validity,
    )

    assert direct_sample["language"] != raw_task_description
    assert direct_sample["language"] == expected_prompt
    assert sample["lang"] == expected_prompt
    video_index_by_key = {
        video_key: video_index for video_index, video_key in enumerate(source_dataset.modality_keys["video"])
    }
    for route_index, (base_video_key, trace_video_key, _) in enumerate(source_dataset.trace_routes):
        if not final_trace_validity[route_index]:
            assert np.array_equal(
                np.asarray(direct_sample["image"][video_index_by_key[base_video_key]]),
                np.asarray(direct_sample["image"][video_index_by_key[trace_video_key]]),
            )

    if args.trace_dropout_rate == 1.0:
        assert sum(int(is_valid) for is_valid in final_trace_validity) == max(
            sum(int(is_valid) for is_valid in initial_trace_validity) - 1,
            0,
        )

    print(f"dataset_length={len(trace_dataset)}")
    print(f"sample_index={args.sample_index}")
    print(f"single_image_count={len(direct_sample['image'])}")
    print(f"single_language={direct_sample['language']}")
