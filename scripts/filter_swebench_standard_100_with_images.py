#!/usr/bin/env python3
"""
Filter SWE-rebench standard dataset to get the first 100 samples with Docker images.
This creates a smaller test set that's guaranteed to have Docker images available.

Usage:
    python scripts/filter_swebench_standard_100_with_images.py
"""

import json
import os
from pathlib import Path

def main():
    print("=" * 70)
    print("SWE-rebench Standard Dataset - Top 100 with Docker Images")
    print("=" * 70)

    # Define paths
    data_dir = Path("data/swebench_rebench")
    standard_file = data_dir / "standard.jsonl"
    output_file = data_dir / "standard_100_with_images.jsonl"

    # Check if input file exists
    if not standard_file.exists():
        print(f"❌ Error: {standard_file} not found!")
        print("Please run: python scripts/download_swebench_rebench.py")
        return

    print(f"📁 Input file: {standard_file}")
    print(f"📁 Output file: {output_file}")
    print(f"🎯 Target: First 100 samples with Docker images")
    print()

    # Process the dataset
    print("🔍 Processing dataset...")
    total_count = 0
    with_images_count = 0
    target_count = 100
    skipped_count = 0

    with open(standard_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            # Stop once we have 100 samples with images
            if with_images_count >= target_count:
                # Count remaining lines to show total processed
                for _ in infile:
                    total_count += 1
                break

            total_count += 1

            try:
                data = json.loads(line.strip())
                instance_id = data.get('instance_id', f'unknown_{line_num}')
                docker_image = data.get('docker_image')

                # Check if Docker image is available and valid
                if docker_image and docker_image != "None":
                    with_images_count += 1
                    outfile.write(line)

                    # Progress indicator
                    if with_images_count % 10 == 0:
                        print(f"  Found {with_images_count}/{target_count} samples with images (processed {line_num} lines)")
                else:
                    skipped_count += 1

            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipped malformed JSON at line {line_num}: {e}")
                continue

    print()
    print("=" * 70)
    print("✅ Filtering Complete!")
    print("=" * 70)

    print(f"📊 Statistics:")
    print(f"   Total lines processed: {total_count:,}")
    print(f"   Samples with Docker images (saved): {with_images_count:,}")
    print(f"   Samples without images (skipped): {skipped_count:,}")
    print()

    print(f"📄 Output file created: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")
    print()

    print("🎯 Usage:")
    print("   This is a small, curated test set ideal for:")
    print("   - Quick validation of your agent setup")
    print("   - Testing new prompts or strategies")
    print("   - Debugging without waiting for large datasets")
    print()
    print("   To use this dataset, create an assignment configuration:")
    print("   configs/assignments/swebench_rebench_100.yaml")
    print()
    print("   All 100 samples are guaranteed to have Docker images available,")
    print("   ensuring reliable execution without image pull failures.")
    print()


if __name__ == '__main__':
    main()
