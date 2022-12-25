import argparse
import json
import os
import random
import shutil
import sys
from typing import List, Tuple


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="split_data")
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("-f", action="store_true", help="Overwrite out_dir if it already exists.")
    args = parser.parse_args()

    assert 0.01 <= args.val_fraction <= 0.5

    if os.path.exists(args.out_dir):
        if not args.f:
            print(f"The out directory {args.out_dir} already exists. Use -f to overwrite it.")
            sys.exit(1)
        shutil.rmtree(args.out_dir)
    for split in "train", "val":
        os.makedirs(os.path.join(args.out_dir, split))

    datafiles: List[Tuple[os.DirEntry, int]] = []  # [(path, n_words), ...]
    for entry in os.scandir(args.data_dir):
        if entry.name.endswith(".json"):
            with open(entry, encoding="utf8") as f:
                length = json.load(f)["text"].split().__len__()
            datafiles.append((entry, length))

    total_length = sum(length for _, length in datafiles)
    val_length = int(total_length * args.val_fraction)

    random.shuffle(datafiles)
    current_val_length = 0
    while current_val_length < val_length:
        entry, length = datafiles.pop()
        dst_path = os.path.join(args.out_dir, "val", entry.name)
        shutil.copyfile(entry.path, dst_path)
        current_val_length += length

    for entry, length in datafiles:
        dst_path = os.path.join(args.out_dir, "train", entry.name)
        shutil.copyfile(entry.path, dst_path)

    print(f"Train: {total_length - current_val_length}\n"
          f"Val:   {current_val_length}")
