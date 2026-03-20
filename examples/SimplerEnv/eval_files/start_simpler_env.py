import os
from datetime import datetime
from pathlib import Path

# from IPython import embed; embed()
from examples.SimplerEnv.eval_files.custom_argparse import get_args
from examples.SimplerEnv.eval_files.model2simpler_interface import ModelClient
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator

import numpy as np


def start_debugpy_once():
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    model = ModelClient(
        policy_setup=args.policy_setup,
        host=args.host,
        port=args.port,
        action_scale=args.action_scale,
    )

    # policy model creation; update this if you are using a new policy model
    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_log_path = output_dir / "success_summary.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    repeat_idx = os.getenv("EVAL_REPEAT_IDX", "1")
    repeat_total = os.getenv("EVAL_REPEAT_TOTAL", "1")
    task_name = os.getenv("EVAL_TASK_NAME", "")
    current_run = os.getenv("EVAL_CURRENT_RUN", "")
    success_rate = float(np.mean(success_arr))
    success_count = int(np.sum(success_arr))
    success_items = ",".join(["1" if bool(item) else "0" for item in success_arr])
    summary_line = (
        f"[{timestamp}] "
        f"run={current_run} "
        f"repeat={repeat_idx}/{repeat_total} "
        f"env={args.env_name} "
        f"scene={args.scene_name} "
        f"task={task_name} "
        f"host={args.host} "
        f"port={args.port} "
        f"success={success_count}/{len(success_arr)} "
        f"rate={success_rate:.6f} "
        f"arr=[{success_items}]"
    )
    with summary_log_path.open("a", encoding="utf-8") as summary_log_file:
        summary_log_file.write(summary_line + "\n")
    print(f"Saved success summary to: {summary_log_path}")
    print(args)
    print(" " * 10, "Average success", success_rate)
