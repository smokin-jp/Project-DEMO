
from pathlib import Path

import numpy as np
import pandas as pd

# Dataset locations
BASE = Path("data/BoxingVI")

ANNOTATIONS = BASE / "Annotation_files"
SKELETONS = BASE / "Skeleton_data"

OUTPUT = Path("data/processed")

OUTPUT.mkdir(parents=True, exist_ok=True)

# Six boxing punch categories
CLASSES = [
    "cross",
    "jab",
    "lead hook",
    "lead uppercut",
    "rear hook",
    "rear uppercut"
]

LABEL_TO_ID = {
    label: index
    for index, label in enumerate(CLASSES)
}


def find_label_column(df):

    best_column = None
    best_count = 0

    for column in df.columns:

        values = (
            df[column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        count = values.isin(LABEL_TO_ID).sum()

        if count > best_count:
            best_count = count
            best_column = column

    if best_column is None:
        raise ValueError("Could not find punch labels")

    return best_column


all_sequences = []
all_labels = []
all_video_ids = []

print("\n========== PREPARING BOXINGVI ==========\n")

for excel_file in sorted(ANNOTATIONS.glob("*.xlsx")):

    video_id = excel_file.stem

    skeleton_file = SKELETONS / f"{video_id}.npy"

    # We initially use only the standard skeleton files.
    if not skeleton_file.exists():
        print(f"Skipping {video_id}: missing skeleton")
        continue

    poses = np.load(skeleton_file)

    # Check the standard skeleton structure.
    if poses.ndim != 4 or poses.shape[1:] != (25, 17, 2):

        print(
            f"Skipping {video_id}: "
            f"unexpected shape {poses.shape}"
        )

        continue

    # Load annotations
    df = pd.read_excel(excel_file)

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    label_column = find_label_column(df)

    # Clean class labels
    raw_labels = (
        df[label_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    valid_mask = raw_labels.isin(LABEL_TO_ID)

    labels = raw_labels[valid_mask]

    # Verify annotations and poses match
    if len(labels) != len(poses):

        raise ValueError(
            f"{video_id}: annotation count {len(labels)} "
            f"does not match skeleton count {len(poses)}"
        )

    # Convert labels into numerical classes
    encoded_labels = np.array(
        [LABEL_TO_ID[label] for label in labels],
        dtype=np.int64
    )

    # Validate pose data
    if not np.isfinite(poses).all():
        raise ValueError(
            f"{video_id}: non-finite pose coordinates"
        )

    # Convert to float32 to save memory
    poses = poses.astype(np.float32)

    all_sequences.append(poses)
    all_labels.append(encoded_labels)

    all_video_ids.extend(
        [video_id] * len(poses)
    )

    print(
        f"{video_id}: "
        f"{len(poses)} sequences prepared"
    )


# Combine all standard-format files
X = np.concatenate(all_sequences, axis=0)

y = np.concatenate(all_labels, axis=0)

video_ids = np.array(all_video_ids)

# Save processed dataset
output_file = OUTPUT / "boxingvi_skeleton.npz"

np.savez_compressed(
    output_file,
    sequences=X,
    labels=y,
    video_ids=video_ids,
    classes=np.array(CLASSES)
)

print("\n========== DATASET COMPLETE ==========\n")

print("Sequences:", X.shape)
print("Labels:", y.shape)

print("\nClass distribution:")

for class_id, class_name in enumerate(CLASSES):

    count = np.sum(y == class_id)

    print(f"{class_name}: {count}")

print("\nVideo distribution:")

for video_id in sorted(set(video_ids)):

    print(
        video_id,
        np.sum(video_ids == video_id)
    )

print("\nSaved to:", output_file)
print("\n========== SUCCESS ==========\n")
