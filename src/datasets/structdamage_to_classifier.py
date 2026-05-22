"""
Convert StructDamage Balanced subset → binary classifier dataset (intact/cracked).

Label encoding: "CRACK" in filename = cracked, otherwise = intact.

Cracked sources  (heritage surfaces): stone, tile, cob, brick (crack files)
Intact sources   (closest proxies):   walls, brick (intact files)

Output structure:
  data/structdamage_classifier/
    train/{intact,cracked}/
    val/{intact,cracked}/
    test/{intact,cracked}/
"""

import random
import shutil
from pathlib import Path

STRUCT_ROOT = Path("data/structdamage/StructDamage/Balanced")
OUT_ROOT    = Path("data/structdamage_classifier")

# surfaces to pull cracked images from
CRACK_SURFACES  = ["stone", "tile", "cob", "brick"]
# surfaces to pull intact images from
INTACT_SURFACES = ["walls", "brick"]

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED   = 42


def collect(surfaces, label):
    """Return list of (src_path, label) for given surfaces and label."""
    paths = []
    for surf in surfaces:
        surf_dir = STRUCT_ROOT / surf
        if not surf_dir.exists():
            print(f"  WARNING: {surf_dir} not found, skipping")
            continue
        for p in surf_dir.iterdir():
            if not p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                continue
            is_crack = "CRACK" in p.name.upper()
            if label == "cracked" and is_crack:
                paths.append(p)
            elif label == "intact" and not is_crack:
                paths.append(p)
    return paths


def split_and_copy(paths, label):
    rng = random.Random(SEED)
    rng.shuffle(paths)
    n = len(paths)
    n_train = int(n * SPLITS["train"])
    n_val   = int(n * SPLITS["val"])

    buckets = {
        "train": paths[:n_train],
        "val":   paths[n_train : n_train + n_val],
        "test":  paths[n_train + n_val :],
    }

    for split, files in buckets.items():
        out_dir = OUT_ROOT / split / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in files:
            dst = out_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

    return {s: len(v) for s, v in buckets.items()}


if __name__ == "__main__":
    print("Collecting cracked images...")
    cracked = collect(CRACK_SURFACES, "cracked")
    print(f"  found {len(cracked)} cracked images")

    print("Collecting intact images...")
    intact = collect(INTACT_SURFACES, "intact")
    print(f"  found {len(intact)} intact images")

    # balance classes to smaller set
    n = min(len(cracked), len(intact))
    rng = random.Random(SEED)
    cracked = rng.sample(cracked, n)
    intact  = rng.sample(intact, n)
    print(f"  balanced to {n} per class")

    print("\nSplitting and copying...")
    c_stats = split_and_copy(cracked, "cracked")
    i_stats = split_and_copy(intact,  "intact")

    print(f"\n{'split':<8} {'cracked':>8} {'intact':>8}")
    print("-" * 26)
    for split in ["train", "val", "test"]:
        print(f"{split:<8} {c_stats[split]:>8} {i_stats[split]:>8}")

    total = sum(c_stats.values()) + sum(i_stats.values())
    print(f"\nTotal images written: {total}")
    print(f"Output: {OUT_ROOT.resolve()}")
