from collections import deque
from typing import Optional, Sequence
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from deployment.model_server.tools import image_tools
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

class CalvinPolicyClient:
    """Wrapper around websocket client with Calvin-specific preprocessing."""
    
    def __init__(
        self,
        host: str,
        port: int,
        resize_size: int = 224,
        replan_steps: int = 5,
        unnorm_key: str = "",
    ):
        self.client = ModelClient(
            host=host,
            port=port,
            image_size=[resize_size, resize_size],
            unnorm_key=(unnorm_key or None),
        )
        self.resize_size = resize_size
        self.replan_steps = replan_steps
        self.step_count = 0
        
    def reset(self):
        """Reset action plan buffer."""
        self.step_count = 0
        
    def step(self, obs: dict, lang_annotation: str) -> np.ndarray:
        """
        Query policy for action given observation and language instruction.
        
        Args:
            obs: Calvin observation dict with keys:
                - rgb_obs: dict with 'rgb_static' (200x200x3) and 'rgb_gripper' (84x84x3)
                - robot_obs: (15,) proprioceptive state [ee_pos(3), ee_ori(3), gripper(2), joint_pos(7)]
            lang_annotation: Natural language task description
            get_action: If True, query model for new action chunk
            
        Returns:
            action: (7,) array [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        # Preprocess images
        rgb_static = obs['rgb_obs']['rgb_static']  # (200, 200, 3) uint8
        rgb_gripper = obs['rgb_obs']['rgb_gripper']  # (84, 84, 3) uint8
        
        # Resize and pad images
        image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(rgb_static, self.resize_size, self.resize_size)
        )
        wrist_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(rgb_gripper, self.resize_size, self.resize_size)
        )
        
        # Prepare input for policy server (aligned with eval_libero)
        example = {
            "image": [image, wrist_image],
            "lang": lang_annotation,
        }
        
        # Query model
        model_output = self.client.step(example=example, step=self.step_count)
        raw_action = model_output["raw_action"]
        world_vector = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
        rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
        open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
        
        action = np.concatenate([world_vector, rotation_delta, open_gripper], axis=0).astype(np.float32)
        self.step_count += 1
        return action