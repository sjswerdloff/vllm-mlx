"""Inspect a steering vector .npz file - shapes, norms, layer coverage."""

import argparse
import mlx.core as mx


def main():
    parser = argparse.ArgumentParser(description="Inspect steering vector file")
    parser.add_argument("file", help="Path to steering_vector.npz")
    args = parser.parse_args()

    data = dict(mx.load(args.file))

    print(f"File: {args.file}")
    print(f"Keys: {sorted(data.keys())}")
    print()

    if "metadata" in data:
        meta = data.pop("metadata")
        vals = meta.tolist()
        print(f"Metadata: {len(vals)} values = {vals}")
        print()

    print(f"{'Layer':<10} {'Shape':<20} {'Norm':<12} {'Mean':<12} {'Std':<12}")
    print("-" * 66)
    for key in sorted(data.keys()):
        vec = data[key]
        norm = mx.sqrt(mx.sum(vec ** 2)).item()
        mean = mx.mean(vec).item()
        std = mx.sqrt(mx.mean((vec - mean) ** 2)).item()
        print(f"{key:<10} {str(vec.shape):<20} {norm:<12.4f} {mean:<12.6f} {std:<12.6f}")


if __name__ == "__main__":
    main()
