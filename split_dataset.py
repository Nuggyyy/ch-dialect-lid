#!/usr/bin/env python3
"""
Split dataset into train/test by class folders.

Usage examples:
  python split_dataset.py             # uses ./data, 90/10 split, copies files
  python split_dataset.py --src data --ratio 0.9 --seed 42 --dry-run
  python split_dataset.py --move     # move instead of copy
"""

import argparse
import os
import random
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Split dataset into train/test by class folders")
    p.add_argument("--src", default="data", help="Source data dir containing class subfolders (default: data)")
    p.add_argument("--ratio", type=float, default=0.9, help="Fraction for training set (default: 0.9)")
    p.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    p.add_argument("--move", action="store_true", help="Move files instead of copying")
    p.add_argument("--dry-run", action="store_true", help="Show actions without copying/moving files")
    p.add_argument("--verbose", action="store_true", help="Print per-file actions")
    return p.parse_args()


def main():
    args = parse_args()
    src_dir = Path(args.src)
    if not src_dir.exists() or not src_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {src_dir}")

    train_root = src_dir / "train"
    test_root = src_dir / "test"

    # find class directories (skip existing train/test)
    class_dirs = [d for d in sorted(src_dir.iterdir()) if d.is_dir() and d.name not in ("train", "test")]
    if not class_dirs:
        raise SystemExit(f"No class subdirectories found in {src_dir}")

    rng = random.Random(args.seed)

    summary = {"total": 0, "train": 0, "test": 0}

    for cls in class_dirs:
        files = [p for p in sorted(cls.iterdir()) if p.is_file()]
        n = len(files)
        if n == 0:
            if args.verbose:
                print(f"Skipping empty class folder: {cls.name}")
            continue

        rng.shuffle(files)
        split_idx = int(n * args.ratio)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        dest_train = train_root / cls.name
        dest_test = test_root / cls.name

        if not args.dry_run:
            dest_train.mkdir(parents=True, exist_ok=True)
            dest_test.mkdir(parents=True, exist_ok=True)

        for p in train_files:
            summary["total"] += 1
            summary["train"] += 1
            dest = dest_train / p.name
            if args.dry_run:
                if args.verbose:
                    print(f"[DRY] train: {p} -> {dest}")
                continue
            if args.move:
                if args.verbose:
                    print(f"move: {p} -> {dest}")
                shutil.move(str(p), str(dest))
            else:
                if args.verbose:
                    print(f"copy: {p} -> {dest}")
                shutil.copy2(str(p), str(dest))

        for p in test_files:
            summary["total"] += 1
            summary["test"] += 1
            dest = dest_test / p.name
            if args.dry_run:
                if args.verbose:
                    print(f"[DRY] test: {p} -> {dest}")
                continue
            if args.move:
                if args.verbose:
                    print(f"move: {p} -> {dest}")
                shutil.move(str(p), str(dest))
            else:
                if args.verbose:
                    print(f"copy: {p} -> {dest}")
                shutil.copy2(str(p), str(dest))

        if args.verbose:
            print(f"Class '{cls.name}': total={n}, train={len(train_files)}, test={len(test_files)}")

    print(f"Done. Total files processed: {summary['total']} (train={summary['train']}, test={summary['test']})")


if __name__ == "__main__":
    main()
