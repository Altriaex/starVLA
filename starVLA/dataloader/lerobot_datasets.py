# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025]. 
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025]. 
# Modification: [suport topdowm processing, suport param from config].

from pathlib import Path
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.trace_datasets import LeRobotSingleTraceDataset

def collate_fn(batch):
    return batch


def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
    """
    
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]

    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "decord"

    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend, # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )

def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:  
        dataset_key = (d_name, robot_type)  
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        if data_mix == "libero_trace_all" and d_name.endswith("_trace"):
            data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
            modality_config = data_config.modality_config()
            transforms = data_config.transform()
            dataset_path = Path(data_root_dir) / d_name
            if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
                print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
                embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
            else:
                embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
            
            video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "decord"
            dataset = LeRobotSingleTraceDataset(
                dataset_path=dataset_path,
                modality_configs=modality_config,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=video_backend,
                delete_pause_frame=delete_pause_frame,
                data_cfg=data_cfg,
                mode=mode,
                seed=seed,
                trace_dropout_rate=data_cfg.get("trace_dropout_rate", 0.0) if data_cfg else 0.0,
            )
        else:
            dataset = make_LeRobotSingleDataset(
                Path(data_root_dir),
                d_name,
                robot_type,
                delete_pause_frame=delete_pause_frame,
                data_cfg=data_cfg,
            )

        dataset_mixture.append((dataset, d_weight))

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step-4 traced dataloader routing acceptance entry.")
    parser.add_argument("--data-root-dir", required=True, help="Dataset root directory")
    parser.add_argument("--data-mix", required=True, help="Named mixture registered in mixtures.py")
    parser.add_argument("--include-state", choices=("true", "false"), default="false", help="Whether to return state")
    parser.add_argument("--video-backend", default="decord", help="Video backend passed to single dataset")
    args = parser.parse_args()
    args.include_state = args.include_state == "true"

    data_cfg = OmegaConf.create(vars(args))
    dataset = get_vla_dataset(data_cfg=data_cfg)
    if not isinstance(dataset, LeRobotMixtureDataset):
        raise TypeError(f"Expected LeRobotMixtureDataset, got {type(dataset).__name__}")
    if len(dataset.datasets) == 0:
        raise ValueError("Constructed mixture dataset must contain at least one single dataset")

    single_dataset = dataset.datasets[0]
    for current_dataset in dataset.datasets:
        is_trace_dataset = current_dataset.dataset_name.endswith("_trace")
        if is_trace_dataset != isinstance(current_dataset, LeRobotSingleTraceDataset):
            raise TypeError(
                "Dataset type routing must match whether dataset_name ends with '_trace'"
            )

    sample = dataset[0]
    expected_keys = {"action", "image", "lang"}
    if args.include_state:
        expected_keys.add("state")
    missing_keys = expected_keys - set(sample)
    if missing_keys:
        raise KeyError(f"Sample missing expected keys: {sorted(missing_keys)}")

    print(f"mixture_dataset_type={type(dataset).__name__}")
    print(f"single_dataset_type={type(single_dataset).__name__}")
    print(f"sample_keys={sorted(sample.keys())}")
    print(f"image_count={len(sample['image'])}")
