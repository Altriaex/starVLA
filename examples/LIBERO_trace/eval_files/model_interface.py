from __future__ import annotations

from collections import deque
import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
from starVLA.model.modules.trace.trace_processor import TraceProcessor, build_trace_aware_prompt


FOUR_VIEW_VIDEO_KEYS = [
    "video.primary_image",
    "video.primary_image_trace",
    "video.wrist_image",
    "video.wrist_image_trace",
]
TRACE_VIDEO_PAIRS = [
    ("video.primary_image", "video.primary_image_trace"),
    ("video.wrist_image", "video.wrist_image_trace"),
]


class ModelClient:
    def __init__(
        self,
        trace_checkpoint_path: str | Path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble: bool = True,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "0.0.0.0",
        port: int = 10095,
        trace_device: str = "cuda",
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        
        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(
                self.action_ensemble_horizon,
                self.adaptive_ensemble_alpha,
            )
        else:
            self.action_ensembler = None
        self.num_image_history = 0
        
        self.action_norm_stats = self.get_action_stats(self.unnorm_key)
        self.action_chunk_size = self.get_action_chunk_size()

        self.primary_trace_runtime = TraceProcessor(
            checkpoint_path=trace_checkpoint_path,
            target_size=(image_size[1], image_size[0]),
            device=trace_device,
        )
        self.wrist_trace_runtime = TraceProcessor(
            checkpoint_path=trace_checkpoint_path,
            target_size=(image_size[1], image_size[0]),
            device=trace_device,
        )

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        
        self.primary_trace_runtime.reset()
        self.wrist_trace_runtime.reset()

    def step(
        self,
        example: dict[str, Any],
        step: int = 0,
        **kwargs,
    ) -> dict[str, Any]:
        task_description = example.get("lang", None)
        if task_description != self.task_description:
            self.reset(task_description)

        primary_image, wrist_image = example["image"]
        primary_result = self.primary_trace_runtime.process_image(primary_image)
        wrist_result = self.wrist_trace_runtime.process_image(wrist_image)

        primary_base_image = np.asarray(primary_result["source_image"], dtype=np.uint8)
        wrist_base_image = np.asarray(wrist_result["source_image"], dtype=np.uint8)

        image_trace_valid = bool(primary_result["has_trace"])
        wrist_image_trace_valid = bool(wrist_result["has_trace"])
        if image_trace_valid:
            primary_trace_image = np.asarray(primary_result["overlay_image"], dtype=np.uint8)
            image_trace_points = np.asarray(primary_result["trace_points"], dtype=np.float32)
        else:
            primary_trace_image = np.copy(primary_base_image)
            image_trace_points = None
        if wrist_image_trace_valid:
            wrist_trace_image = np.asarray(wrist_result["overlay_image"], dtype=np.uint8)
            wrist_image_trace_points = np.asarray(wrist_result["trace_points"], dtype=np.float32)
        else:
            wrist_trace_image = np.copy(wrist_base_image)
            wrist_image_trace_points = None

        request_example = {
            "image": [
                primary_base_image,
                primary_trace_image,
                wrist_base_image,
                wrist_trace_image,
            ],
            "lang": build_trace_aware_prompt(
                task_description=task_description,
                video_keys=FOUR_VIEW_VIDEO_KEYS,
                trace_video_pairs=TRACE_VIDEO_PAIRS,
                trace_validity=[image_trace_valid, wrist_image_trace_valid],
            ),
            "image_trace_valid": image_trace_valid,
            "wrist_image_trace_valid": wrist_image_trace_valid,
            "image_trace_points": image_trace_points,
            "wrist_image_trace_points": wrist_image_trace_points,
        }
        vla_input = {
            "examples": [request_example],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
            "type": "infer",
        }

        action_chunk_size = self.action_chunk_size
        if step % action_chunk_size == 0:
            response = self.client.predict_action(vla_input)
            try:
                normalized_actions = response["data"]["normalized_actions"]
            except KeyError:
                print(f"Response data: {response}")
                raise KeyError(
                    f"Key 'normalized_actions' not found in response data: {response['data'].keys()}"
                ) from None
            normalized_actions = normalized_actions[0]
            self.raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions,
                action_norm_stats=self.action_norm_stats,
            )

        raw_actions = self.raw_actions[step % action_chunk_size][None]
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),
        }
        return {
            "raw_action": raw_action,
            "primary_trace_image": np.array(primary_trace_image, copy=True),
            "wrist_trace_image": np.array(wrist_trace_image, copy=True),
        }


    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    def get_action_stats(self, unnorm_key: str) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        input = {"type": "get_ckpt_config"}
        _, norm_stats = self.client.get_ckpt_config(input)["data"]
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    def get_action_chunk_size(self):
        input = {"type": "get_ckpt_config"}
        model_config, _ = self.client.get_ckpt_config(input)["data"]
        return model_config['framework']['action_model']['future_action_window_size'] + 1

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4.5 step-7 acceptance entry for ModelClient.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5694)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trace-checkpoint-path", required=True)
    parser.add_argument("--trace-device", default="cuda")
    args = parser.parse_args()

    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    resolution = 256
    trace_priming_actions = (
        np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
        np.array([0.0, 0.02, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
        np.array([-0.02, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
        np.array([0.0, -0.02, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_goal"
    task_suite = benchmark_dict[task_suite_name]()
    if not 0 <= args.task_id < task_suite.n_tasks:
        raise ValueError(f"--task-id must be within [0, {task_suite.n_tasks}), got {args.task_id}")

    task = task_suite.get_task(args.task_id)
    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(args.seed)
    initial_states = task_suite.get_task_init_states(args.task_id)
    client_model = ModelClient(
        host=args.host,
        port=args.port,
        image_size=[224, 224],
        trace_checkpoint_path=args.trace_checkpoint_path,
        trace_device=args.trace_device,
    )
    captured_request = {"example": None}
    original_predict_action = client_model.client.predict_action

    def capture_predict_action(vla_input: dict[str, Any]):
        captured_request["example"] = vla_input["examples"][0]
        return original_predict_action(vla_input)

    client_model.client.predict_action = capture_predict_action

    try:
        client_model.reset(task_description=task_description)
        env.reset()
        obs = env.set_init_state(initial_states[0])
        saw_trace_positive = False
        expected_prompt = None
        for step in range(args.num_steps):
            response = client_model.step(
                example={
                    "image": [
                        np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
                        np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
                    ],
                    "lang": task_description,
                },
                step=step * client_model.action_chunk_size,
            )
            request_example = captured_request["example"]
            if request_example is None:
                raise RuntimeError("ModelClient.step() did not send a request example to the policy server")
            expected_prompt = build_trace_aware_prompt(
                task_description=task_description,
                video_keys=FOUR_VIEW_VIDEO_KEYS,
                trace_video_pairs=TRACE_VIDEO_PAIRS,
                trace_validity=[
                    request_example["image_trace_valid"],
                    request_example["wrist_image_trace_valid"],
                ],
            )
            assert list(request_example) == [
                "image",
                "lang",
                "image_trace_valid",
                "wrist_image_trace_valid",
                "image_trace_points",
                "wrist_image_trace_points",
            ]
            assert request_example["lang"] == expected_prompt
            assert request_example["image_trace_valid"] is (request_example["image_trace_points"] is not None)
            assert request_example["wrist_image_trace_valid"] is (
                request_example["wrist_image_trace_points"] is not None
            )
            assert np.array_equal(response["primary_trace_image"], request_example["image"][1])
            assert np.array_equal(response["wrist_trace_image"], request_example["image"][3])
            if request_example["image_trace_valid"]:
                assert not np.array_equal(request_example["image"][0], request_example["image"][1])
                saw_trace_positive = True
            else:
                assert np.array_equal(request_example["image"][0], request_example["image"][1])
            if request_example["wrist_image_trace_valid"]:
                assert not np.array_equal(request_example["image"][2], request_example["image"][3])
                saw_trace_positive = True
            else:
                assert np.array_equal(request_example["image"][2], request_example["image"][3])

            raw_action = response["raw_action"]
            assert list(raw_action) == ["world_vector", "rotation_delta", "open_gripper"]
            delta_action = trace_priming_actions[step % len(trace_priming_actions)]
            obs, _, done, _ = env.step(delta_action.tolist())
            if done and step + 1 < args.num_steps:
                raise RuntimeError("Episode terminated before finishing the fixed reset + 20 step acceptance sample")

        if not saw_trace_positive:
            raise RuntimeError("The fixed reset + 20 step acceptance sample never exercised a trace-positive request")

        final_trace_validity = [
            request_example["image_trace_valid"],
            request_example["wrist_image_trace_valid"],
        ]

        client_model.reset(task_description=task_description)
        assert client_model.primary_trace_runtime.get_state()["step"] == 0
        assert client_model.wrist_trace_runtime.get_state()["step"] == 0
        assert client_model.sticky_action_is_on is False
        assert client_model.previous_gripper_action is None

        print("step7_smoke_ok=1")
        print(f"task_suite_name={task_suite_name}")
        print(f"task_id={args.task_id}")
        print(f"num_steps={args.num_steps}")
        print(f"prompt={expected_prompt}")
        print(f"trace_validity={final_trace_validity}")
    finally:
        client_model.client.predict_action = original_predict_action
        env.close()
        client_model.client.close()
