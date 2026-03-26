"""
Retrieve the maximum episode length from a LIBERO-Mem dataset in LeRobot format.
lerobot version: v2.1

This utility reads the dataset metadata file located at `{datasets_root_dir}/meta/episodes.json` 
and calculates the maximum value of the `length` field across all recorded episodes. 

Expected JSON structure:
    {"episode_index": [{"length": 150, ...}, {"length": 210, ...}, ...]}

Usage:
    python get_max_length.py --dataset-dir /path/to/your/dataset
"""

import json
import argparse
import sys
from pathlib import Path

def get_max_episode_length(datasets_root_dir: str | Path) -> int:
    """
    Reads and parses episodes.jsonl line-by-line, returning the maximum length value.
    """
    episodes_file = Path(datasets_root_dir) / "meta" / "episodes.jsonl"
    
    if not episodes_file.exists():
        print(f"❌ Error: File not found at {episodes_file}")
        sys.exit(1)
        
    max_len = 0
    
    with open(episodes_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
                
            try:
                episode_data = json.loads(line)
                # Safely get 'length', defaulting to 0 if missing
                current_len = int(episode_data.get("length", 0))
                
                if current_len > max_len:
                    max_len = current_len
                    
            except json.JSONDecodeError:
                print(f"❌ Error: Invalid JSON on line {line_num} in {episodes_file}")
                sys.exit(1)
            except Exception as e:
                print(f"❌ Error processing line {line_num}: {e}")
                sys.exit(1)
                
    if max_len == 0:
        print("⚠️ Warning: Maximum length is 0. Check if the 'length' field exists in your JSONL file.")
        
    return max_len

def main():
    parser = argparse.ArgumentParser(description="Get the maximum episode length from a LeRobot dataset (JSONL format).")
    parser.add_argument(
        "--dataset-dir", 
        type=str, 
        required=True, 
        help="Path to the root directory of the LeRobot dataset (the parent directory of the 'meta' folder)."
    )
    
    args = parser.parse_args()
    
    max_length = get_max_episode_length(args.dataset_dir)
    
    print("-" * 40)
    print(f"📁 Dataset Directory: {args.dataset_dir}")
    print(f"📏 Maximum Episode Length: {max_length}")
    print("-" * 40)

if __name__ == "__main__":
    main()