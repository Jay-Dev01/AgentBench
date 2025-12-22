#!/usr/bin/env python3
"""
Download and prepare SWE-rebench dataset.

Usage:
    python scripts/download_swebench_rebench.py
"""

import json
import random
from pathlib import Path
from datasets import load_dataset

def main():
    print("=" * 60)
    print("SWE-rebench Dataset Downloader")
    print("=" * 60)

    # Create data directory
    data_dir = Path("data/swebench_rebench")
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nData directory: {data_dir}")
    print("\nDownloading dataset from HuggingFace...")
    print("This may take a few minutes...\n")

    # Load dataset
    dataset = load_dataset('nebius/SWE-rebench', split='test')
    total = len(dataset)

    print(f"Total instances: {total:,}")

    # Create dev set (100 instances)
    print("\nCreating dev set (100 instances)...")
    dev_indices = random.sample(range(total), min(100, total))
    dev_file = data_dir / "dev.jsonl"

    with open(dev_file, 'w') as f:
        for idx in dev_indices:
            f.write(json.dumps(dataset[idx]) + '\n')

    print(f"✓ Created: {dev_file}")

    # Create standard set (1000 instances)
    print("\nCreating standard set (1000 instances)...")
    std_indices = random.sample(range(total), min(1000, total))
    std_file = data_dir / "standard.jsonl"

    with open(std_file, 'w') as f:
        for idx in std_indices:
            f.write(json.dumps(dataset[idx]) + '\n')

    print(f"✓ Created: {std_file}")

    # Create full set
    print(f"\nCreating full set ({total:,} instances)...")
    full_file = data_dir / "full.jsonl"

    with open(full_file, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')

    print(f"✓ Created: {full_file}")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {dev_file} (100 instances)")
    print(f"  - {std_file} (1,000 instances)")
    print(f"  - {full_file} ({total:,} instances)")
    print("\nYou can now run the task with:")
    print("  python -m src.assigner --config configs/assignments/swebench_rebench_test.yaml")

if __name__ == '__main__':
    main()
