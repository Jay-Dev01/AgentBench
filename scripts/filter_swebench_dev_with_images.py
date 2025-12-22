#!/usr/bin/env python3
"""
Filter SWE-rebench dev dataset to only include samples with Docker images.

Usage:
    python scripts/filter_swebench_dev_with_images.py
"""

import json
import os
from pathlib import Path

def main():
    print("=" * 70)
    print("SWE-rebench Dev Dataset Docker Image Filter")
    print("=" * 70)

    # Define paths
    data_dir = Path("data/swebench_rebench")
    dev_file = data_dir / "dev.jsonl"
    output_file = data_dir / "dev_with_images.jsonl"

    # Check if input file exists
    if not dev_file.exists():
        print(f"❌ Error: {dev_file} not found!")
        print("Please run: python scripts/download_swebench_rebench.py")
        return

    print(f"📁 Input file: {dev_file}")
    print(f"📁 Output file: {output_file}")
    print()

    # Process the dataset
    print("🔍 Processing dataset...")
    total_count = 0
    with_images_count = 0
    skipped_examples = []

    with open(dev_file, 'r') as infile, open(output_file, 'w') as outfile:
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
                    # Collect examples of skipped instances for logging
                    skipped_examples.append(instance_id)

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
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")
    print()

    if skipped_examples:
        print(f"💡 Skipped instances (no Docker image):")
        for example in skipped_examples:
            print(f"   - {example}")
        print()

    print("🎯 Usage:")
    print("   To use this filtered dataset, update your assignment configuration:")
    print("   configs/assignments/swebench_rebench_test.yaml")
    print()
    print("   The dev-with-images dataset is ideal for quick testing with")
    print("   guaranteed Docker image availability.")
    print()


if __name__ == '__main__':
    main()
