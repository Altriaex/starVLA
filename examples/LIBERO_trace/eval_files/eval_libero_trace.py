import dataclasses
import datetime as dt
import json
import logging
import math
import os
import re
from pathlib import Path
import time
import pandas as pd

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from model_interface import ModelClient


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _scan_existing_results(video_dir: Path) -> dict[str, dict[int, str]]:
    """Scan video directory and return completed episodes.

    Parses filenames like rollout_{task_segment}_episode{idx}_{suffix}.mp4
    and returns {task_segment: {episode_idx: suffix}}. When multiple suffixes
    exist for the same episode, the one with the latest mtime wins.
    """
    pattern = re.compile(r"^rollout_(.+)_episode(\d+)_(success|failure)\.mp4$")
    results: dict[str, dict[int, tuple[str, float]]] = {}
    if not video_dir.exists():
        return {}
    for fpath in video_dir.iterdir():
        if not fpath.is_file() or fpath.suffix != ".mp4":
            continue
        m = pattern.match(fpath.name)
        if not m:
            continue
        task_segment = m.group(1)
        episode_idx = int(m.group(2))
        suffix = m.group(3)
        mtime = fpath.stat().st_mtime
        task_dict = results.setdefault(task_segment, {})
        if episode_idx not in task_dict or mtime > task_dict[episode_idx][1]:
            task_dict[episode_idx] = (suffix, mtime)
    return {task: {idx: info[0] for idx, info in episodes.items()} for task, episodes in results.items()}


def _build_eval_record_from_videos(video_dir: Path, task_order: list[str]) -> list[dict]:
    """Reconstruct eval_record from existing video files.

    task_order is a list of task_descriptions; their order determines the CSV row order.
    """
    existing = _scan_existing_results(video_dir)
    eval_record = []
    total_successes = 0
    total_episodes = 0
    for task_desc in task_order:
        task_segment = task_desc.replace(" ", "_")
        episodes = existing.get(task_segment, {})
        n_episodes = len(episodes)
        n_success = sum(1 for s in episodes.values() if s == "success")
        total_successes += n_success
        total_episodes += n_episodes
        if n_episodes > 0:
            eval_record.append(
                {
                    "task_description": task_desc,
                    "n_success": n_success,
                    "n_episodes": n_episodes,
                    "success rate": float(n_success) / float(n_episodes),
                }
            )
    if total_episodes > 0:
        eval_record.append(
            {
                "task_description": "Total",
                "n_success": total_successes,
                "n_episodes": total_episodes,
                "success rate": float(total_successes) / float(total_episodes),
            }
        )
    return eval_record


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    trace_checkpoint_path: str
    host: str = "127.0.0.1"
    port: int = 10093
    trace_device: str = "cuda"
    resize_size = [224,224]

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    output_dir: str = "experiments/libero/logs"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    post_process_action: bool = True

    job_name: str = "test"


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"
    video_dir = Path(args.output_dir) / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        trace_checkpoint_path=args.trace_checkpoint_path,
        trace_device=args.trace_device,
    )

    # Scan existing results for resume
    existing_results = _scan_existing_results(video_dir)
    if existing_results:
        total_existing = sum(len(v) for v in existing_results.values())
        logging.info(f"[resume] Found {total_existing} existing episode results in {video_dir}")

    eval_record = []
    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_order = []
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        task_order.append(task_description)
        task_segment = task_description.replace(" ", "_")

        # Start episodes
        task_episodes, task_successes = 0, 0
        # Pre-load statistics from existing results for this task
        pre_existing = existing_results.get(task_segment, {})
        for suffix in pre_existing.values():
            task_episodes += 1
            total_episodes += 1
            if suffix == "success":
                task_successes += 1
                total_successes += 1

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            # Skip if this episode was already evaluated
            if episode_idx in pre_existing:
                logging.info(f"[skip] {task_segment} episode {episode_idx} already done ({pre_existing[episode_idx]})")
                continue

            logging.info(f"\nTask: {task_description}")

            # Reset environment
            client_model.reset(task_description=task_description)  # Reset the client connection
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            primary_trace_replay_images = []
            wrist_trace_replay_images = []
            full_actions = []

            logging.info(f"Starting episode {task_episodes + 1}...")
            step = 0

            # full_actions = np.load("./debug/action.npy")

            while t < max_steps + args.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )

                # Save preprocessed image for replay video
                replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                observation = { # 
                    "observation.primary": np.expand_dims(
                        img, axis=0
                    ),  # (H, W, C), dtype=unit8, range(0-255)
                    "observation.wrist_image": np.expand_dims(
                        wrist_img, axis=0
                    ),  # (H, W, C)
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                # align key with model API --> 这里给了两个图像 --> check training
                example_dict = {
                    "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "lang": observation["instruction"][0],
                }

                
                start_time = time.time()
                
                response = client_model.step(example=example_dict, step=step)
                primary_trace_replay_images.append(response["primary_trace_image"])
                wrist_trace_replay_images.append(response["wrist_trace_image"])
                
                end_time = time.time()
                # print(f"time: {end_time - start_time}")

                # # 
                raw_action = response["raw_action"]

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(f"Unexpected action sizes: "
                                    f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                                    f"Falling back to LIBERO_DUMMY_ACTION.")
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)

                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            video_stem = f"rollout_{task_segment}_episode{episode_idx}_{suffix}"
            fname = f"{video_stem}.mp4"
            primary_trace_fname = f"{video_stem}_primary_trace.mp4"
            wrist_trace_fname = f"{video_stem}_wrist_trace.mp4"
            imageio.mimwrite(
                video_dir / fname,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            imageio.mimwrite(
                video_dir / primary_trace_fname,
                [np.asarray(x) for x in primary_trace_replay_images],
                fps=10,
            )
            imageio.mimwrite(
                video_dir / wrist_trace_fname,
                [np.asarray(x) for x in wrist_trace_replay_images],
                fps=10,
            )
            
            full_actions = np.stack(full_actions)
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )

        # Log final results
        if task_episodes > 0:
            logging.info(
                f"Current task success rate: {float(task_successes) / float(task_episodes)}"
            )
            logging.info(
                f"Current total success rate: {float(total_successes) / float(total_episodes)}"
            )
            eval_record.append(
                {
                    "task_description": task_description,
                    "n_success": task_successes,
                    "n_episodes": task_episodes,
                    "success rate": float(task_successes) / float(task_episodes)
                }
            )
        env.close()
    if total_episodes > 0:
        logging.info(
            f"Total success rate: {float(total_successes) / float(total_episodes)}"
        )
    logging.info(f"Total episodes: {total_episodes}")
    # Rebuild eval_record from videos to guarantee consistency after resume
    eval_record = _build_eval_record_from_videos(video_dir, task_order)
    csv_path = Path(args.output_dir) / "eval_record.csv"
    pd.DataFrame(eval_record).to_csv(csv_path, index=False)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True

if __name__ == "__main__":
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
