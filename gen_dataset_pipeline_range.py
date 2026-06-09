import json
import requests
import os
import argparse
import numpy as np


def download_file(url, dest_folder):
    """Downloads a single file from the given URL to the destination folder."""
    filename = url.split("/")[-1]
    filepath = os.path.join(dest_folder, filename)

    try:
        # stream=True is crucial here because the files are ~3GB+ each.
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print(f"✅ Successfully downloaded: {filename}")
        return filepath
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return None


def process_and_append(filepath, seq_len, num_samples, output_path):
    """Loads a shard, extracts random sequences, appends to JSONL, and returns count."""
    try:
        data = np.load(filepath)
    except Exception as e:
        print(f"❌ Failed to load {filepath}: {e}")
        return 0

    n_tokens = data.shape[0]
    n_sequences = n_tokens // seq_len
    actual_samples = min(num_samples, n_sequences)

    if actual_samples <= 0:
        print(
            f"⚠️ Shard has {n_tokens} tokens, which is less than seq_len ({seq_len}). Skipping."
        )
        return 0

    # Randomly sample indices without replacement
    indices = np.random.choice(n_sequences, size=actual_samples, replace=False)

    # Open in append mode ("a"), write sequences, and close (saves to disk)
    with open(output_path, "a") as f:
        for idx in indices:
            seq = data[idx * seq_len : (idx + 1) * seq_len].tolist()
            f.write(json.dumps({"input_ids": seq}) + "\n")

    return actual_samples


def main():
    parser = argparse.ArgumentParser(
        description="Download shards in a specific range, extract sequences, and save to JSONL."
    )

    # Range-based selection instead of random num_shards
    parser.add_argument(
        "--start_index",
        type=int,
        required=True,
        help="Starting index (inclusive) in the shards list",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        required=True,
        help="Ending index (inclusive) in the shards list",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=3000,
        help="Number of samples to extract per shard (default: 3000)",
    )

    # Additional useful parameters
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Sequence length in tokens (default: 2048)",
    )
    parser.add_argument(
        "--output", type=str, default="dataset.jsonl", help="Output JSONL file path"
    )
    parser.add_argument(
        "--shards_json",
        type=str,
        default="shards.json",
        help="Path to the shards JSON file",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default="dataset/temp",
        help="Directory to temporarily store shards before deletion",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://eu-central-1.hippius.com/teutonic-sn3/",
        help="Base URL for the shards",
    )

    args = parser.parse_args()

    # 1. Load the JSON file
    try:
        with open(args.shards_json, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Could not find the file '{args.shards_json}'.")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: '{args.shards_json}' is not a valid JSON file.")
        return

    total_objects = len(data)
    if total_objects == 0:
        print("❌ The JSON list is empty.")
        return

    # Validate range
    start = max(0, args.start_index)
    end = min(total_objects - 1, args.end_index)

    if start > end:
        print(f"❌ Invalid range: start ({start}) > end ({end})")
        return

    chosen_objects = data[start : end + 1]
    num_to_process = len(chosen_objects)

    print(f"Loaded {total_objects} objects from '{args.shards_json}'.")
    print(
        f"Processing shards from index {start} to {end} (inclusive) → {num_to_process} shards."
    )

    # 2. Prepare directories
    os.makedirs(args.download_dir, exist_ok=True)

    # Clear the output file before starting the run
    with open(args.output, "w") as f:
        pass

    total_extracted = 0

    # 3. Sequentially process each shard (Download -> Extract -> Save -> Delete)
    for i, obj in enumerate(chosen_objects):
        print(
            f"\n--- Processing Shard [{i+1}/{num_to_process}] (index {start + i}) ---"
        )
        url = args.base_url + obj["key"]

        # Download the shard
        filepath = download_file(url, args.download_dir)

        if filepath and os.path.exists(filepath):
            # Extract sequences and append to JSONL
            extracted_count = process_and_append(
                filepath, args.seq_len, args.num_samples, args.output
            )
            total_extracted += extracted_count
            print(
                f"📝 Extracted {extracted_count} sequences. (Total so far: {total_extracted})"
            )

            # Delete the shard to free up disk space immediately
            try:
                os.remove(filepath)
                print(f"🗑️ Deleted temporary shard: {os.path.basename(filepath)}")
            except Exception as e:
                print(f"⚠️ Warning: Could not delete {filepath}: {e}")
        else:
            print("⏭️ Skipping extraction due to download failure.")

    print(f"\n🎉 All downloads and extractions completed!")
    print(f"✅ Total sequences saved to '{args.output}': {total_extracted}")


if __name__ == "__main__":
    main()
