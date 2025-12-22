#!/usr/bin/env python3
"""
Script to download and prepare SWE-rebench dataset for AgentBench.
"""
import json
import os
from pathlib import Path
from datasets import load_dataset

def download_swebench():
    """Download SWE-rebench dataset and convert to AgentBench format."""
    print("Downloading SWE-rebench dataset from HuggingFace...")

    # Load dataset
    dataset = load_dataset('nebius/SWE-rebench', split='test')

    print(f"Downloaded {len(dataset)} samples")

    # Create output directory
    output_dir = Path(__file__).parent.parent / "data" / "swebench"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to JSONL format
    samples = []
    for idx, item in enumerate(dataset):
        # Extract relevant fields
        sample = {
            "instance_id": item["instance_id"],
            "repo": item["repo"],
            "base_commit": item["base_commit"],
            "problem_statement": item["problem_statement"],
            "hints_text": item.get("hints_text", ""),
            "patch": item["patch"],
            "test_patch": item["test_patch"],
            "version": item.get("version", ""),
            "FAIL_TO_PASS": item.get("FAIL_TO_PASS", []),
            "PASS_TO_PASS": item.get("PASS_TO_PASS", []),
            "environment_setup_commit": item.get("environment_setup_commit", ""),
            "created_at": item.get("created_at", ""),
            "license_name": item.get("license_name", ""),
            "install_config": item.get("install_config", {}),
            "meta": item.get("meta", {})
        }
        samples.append(sample)

        # Progress update
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(dataset)} samples")

    # Save full dataset
    full_output = output_dir / "full.jsonl"
    with open(full_output, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    print(f"Saved full dataset to {full_output}")

    # Create a dev subset (first 100 samples for testing)
    dev_output = output_dir / "dev.jsonl"
    with open(dev_output, 'w') as f:
        for sample in samples[:100]:
            f.write(json.dumps(sample) + '\n')
    print(f"Saved dev subset (100 samples) to {dev_output}")

    # Create a standard subset (first 1000 samples)
    std_output = output_dir / "standard.jsonl"
    with open(std_output, 'w') as f:
        for sample in samples[:1000]:
            f.write(json.dumps(sample) + '\n')
    print(f"Saved standard subset (1000 samples) to {std_output}")

    print("\nDataset preparation complete!")
    print(f"Total samples: {len(samples)}")
    print(f"  - Full dataset: {len(samples)} samples in full.jsonl")
    print(f"  - Dev set: 100 samples in dev.jsonl")
    print(f"  - Standard set: 1000 samples in standard.jsonl")

if __name__ == "__main__":
    download_swebench()
