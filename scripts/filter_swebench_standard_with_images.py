#!/usr/bin/env python3
"""
Filter SWE-rebench standard dataset to only include samples with Docker images.

Usage:
    python scripts/filter_swebench_standard_with_images.py
"""

import json
import os
from pathlib import Path

def main():
    print("=" * 70)
    print("SWE-rebench Standard Dataset Docker Image Filter")
    print("=" * 70)

    # Define paths
    data_dir = Path("data/swebench_rebench")
    standard_file = data_dir / "standard.jsonl"
    output_file = data_dir / "standard_with_images.jsonl"

    # Check if input file exists
    if not standard_file.exists():
        print(f"❌ Error: {standard_file} not found!")
        print("Please run: python scripts/download_swebench_rebench.py")
        return

    print(f"📁 Input file: {standard_file}")
    print(f"📁 Output file: {output_file}")
    print()

    # Process the dataset
    print("🔍 Processing dataset...")
    total_count = 0
    with_images_count = 0
    skipped_examples = []

    with open(standard_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            total_count += 1

            try:
                data = json.loads(line.strip())
                instance_id = data.get('instance_id', f'unknown_{line_num}')
                docker_image = data.get('docker_image')

                # Check if Docker image is available and valid
                if docker_image and docker_image != "None":
                    with_images_count += 1
                    outfile.write(line)
                else:
                    # Collect some examples of skipped instances for logging
                    if len(skipped_examples) < 10:
                        skipped_examples.append(instance_id)

                # Progress indicator
                if line_num % 100 == 0:
                    print(f"  Processed {line_num:,} samples... ({with_images_count:,} with images)")

            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipped malformed JSON at line {line_num}: {e}")
                continue

    print()
    print("=" * 70)
    print("✅ Filtering Complete!")
    print("=" * 70)

    print(f"📊 Statistics:")
    print(f"   Total samples processed: {total_count:,}")
    print(f"   Samples with Docker images: {with_images_count:,}")
    print(f"   Samples without images (skipped): {total_count - with_images_count:,}")
    print(f"   Percentage with images: {with_images_count/total_count*100:.1f}%")
    print()

    print(f"📄 Output file created: {output_file}")
    print(f"   File size: {output_file.stat().st_size / (1024*1024):.1f} MB")
    print()

    if skipped_examples:
        print(f"💡 Examples of skipped instances (no Docker image):")
        for example in skipped_examples:
            print(f"   - {example}")
        if total_count - with_images_count > len(skipped_examples):
            print(f"   ... and {total_count - with_images_count - len(skipped_examples):,} more")
        print()

    print("🎯 Usage:")
    print("   To use this filtered dataset, update your task configuration:")
    print("   configs/tasks/swebench_rebench.yaml")
    print()
    print("   Example:")
    print("   swebench-rebench-std:")
    print("     parameters:")
    print("       name: \"swebench-rebench-std\"")
    print(f"       data_file: \"data/swebench_rebench/standard_with_images.jsonl\"")
    print("       output_file: \"outputs/swebench_rebench_std_predictions.jsonl\"")
    print()


if __name__ == '__main__':
    main()
