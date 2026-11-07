from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================
# 1. SETTINGS
# ==========================================

SEED = 42
EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 6

# Keep the same temporary video-level split as Experiment 6.
# This is not confirmed to be the official subject-based BoxingVI split.
VALIDATION_VIDEOS = {"V1", "V4"}

DATA_FILE = Path("data/processed/boxingvi_skeleton.npz")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Unique Experiment 7 output paths; earlier experiments are preserved.
BEST_CHECKPOINT = (
    CHECKPOINT_DIR / "skeleton_gru_normalized_velocity_best.pth"
)
CONFUSION_MATRIX_FILE = (
    CHECKPOINT_DIR / "skeleton_gru_normalized_velocity_confusion_matrix.png"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Using device:", DEVICE)


# ==========================================
# 2. LOAD AND VALIDATE DATA
# ==========================================

if not DATA_FILE.is_file():
    raise FileNotFoundError(
        f"Could not find {DATA_FILE}. Run prepare_skeleton_dataset.py first."
    )

with np.load(DATA_FILE, allow_pickle=False) as data:
    X = data["sequences"].astype(np.float32)
    y = data["labels"].astype(np.int64)
    video_ids = data["video_ids"].astype(str)
    classes = [str(name) for name in data["classes"].tolist()]

print("\nDataset shape:", X.shape)
print("Labels shape:", y.shape)
print("Classes:", classes)

if X.ndim != 4 or X.shape[1:] != (25, 17, 2):
    raise ValueError(f"Expected (N, 25, 17, 2), received {X.shape}.")
if len(X) != len(y) or len(X) != len(video_ids):
    raise ValueError("Sequence, label, and video-ID counts do not match.")
if len(classes) != 6:
    raise ValueError(f"Expected 6 punch classes, found {len(classes)}.")
if not np.isfinite(X).all():
    raise ValueError("Skeleton coordinates contain NaN or infinity.")
if np.any((y < 0) | (y >= len(classes))):
    raise ValueError("Found a label ID outside the classes list.")


# ==========================================
# 3. REMOVE EMPTY SEQUENCES AND GET LENGTHS
# ==========================================

# A frame is valid if at least one keypoint has a non-zero coordinate.
valid_frames = np.any(np.abs(X) > 1e-8, axis=(2, 3))
has_pose = valid_frames.any(axis=1)
empty_indices = np.flatnonzero(~has_pose)

print("\n========== EMPTY SEQUENCE INSPECTION ==========")
print("Empty sequences found:", len(empty_indices))
for index in empty_indices:
    print(
        f"Index: {index} | Video: {video_ids[index]} | "
        f"Punch: {classes[y[index]]}"
    )

if len(empty_indices):
    X = X[has_pose]
    y = y[has_pose]
    video_ids = video_ids[has_pose]
    valid_frames = valid_frames[has_pose]
    print(f"Removed {len(empty_indices)} completely empty sequences.")

if len(X) == 0:
    raise ValueError("No non-empty skeleton sequences remain.")

leading_empty = int(np.sum(~valid_frames[:, 0]))
if leading_empty:
    raise ValueError(
        f"Found {leading_empty} sequences with leading empty frames. "
        "The packed-GRU setup expects valid frames to start at frame 0."
    )

last_valid_index = X.shape[1] - 1 - np.argmax(valid_frames[:, ::-1], axis=1)
lengths = (last_valid_index + 1).astype(np.int64)

if np.any(lengths < 1) or np.any(lengths > X.shape[1]):
    raise ValueError("Invalid sequence lengths detected.")

frame_positions = np.arange(X.shape[1])[None, :]
inside_sequence = frame_positions < lengths[:, None]
missing_inside = int(np.sum((~valid_frames) & inside_sequence))

print("\n========== SEQUENCE LENGTHS ==========")
print("Remaining sequences:", len(X))
print("Sequences with leading empty frames:", leading_empty)
print("Missing frames inside sequences:", missing_inside)
print("Minimum sequence length:", int(lengths.min()))
print("Maximum sequence length:", int(lengths.max()))
print("Average sequence length:", float(lengths.mean()))


# ==========================================
# 4. SPLIT BY VIDEO BEFORE FITTING VELOCITY SCALING
# ==========================================

validation_mask = np.isin(video_ids, list(VALIDATION_VIDEOS))
training_mask = ~validation_mask

if not validation_mask.any():
    raise ValueError(
        f"Validation videos {sorted(VALIDATION_VIDEOS)} were not found."
    )
if not training_mask.any():
    raise ValueError("No training videos remain after the split.")

X_train_raw = X[training_mask]
y_train_np = y[training_mask]
lengths_train_np = lengths[training_mask]
valid_train = valid_frames[training_mask]

X_val_raw = X[validation_mask]
y_val_np = y[validation_mask]
lengths_val_np = lengths[validation_mask]
valid_val = valid_frames[validation_mask]

train_videos = sorted(set(video_ids[training_mask].tolist()))
val_videos = sorted(set(video_ids[validation_mask].tolist()))

print("\nTraining videos:", train_videos)
print("Validation videos:", val_videos)
print("Training samples:", len(X_train_raw))
print("Validation samples:", len(X_val_raw))

for class_id, class_name in enumerate(classes):
    train_count = int(np.sum(y_train_np == class_id))
    val_count = int(np.sum(y_val_np == class_id))
    print(f"{class_name}: train={train_count}, validation={val_count}")
    if train_count == 0 or val_count == 0:
        raise ValueError(f"Class '{class_name}' is missing from a partition.")


# ==========================================
# 5. BODY-RELATIVE COORDINATE NORMALISATION
# ==========================================

# COCO 17-keypoint indices used by the BoxingVI pose format:
# left shoulder=5, right shoulder=6, left hip=11, right hip=12.
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


def normalise_sequences(
    sequences: np.ndarray,
    sequence_lengths: np.ndarray,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Use a fixed hip-centre origin and median torso scale per clip.

    Only valid frames are transformed. Trailing padded frames stay zero.
    Returns normalised sequences and one torso scale per sequence.
    """
    normalised = np.zeros_like(sequences, dtype=np.float32)
    scales = np.zeros(len(sequences), dtype=np.float32)

    for i, sequence_length in enumerate(sequence_lengths):
        sequence = sequences[i, :sequence_length]  # (T, 17, 2)

        hip_midpoint = (
            sequence[:, LEFT_HIP, :] + sequence[:, RIGHT_HIP, :]
        ) / 2.0
        shoulder_midpoint = (
            sequence[:, LEFT_SHOULDER, :] + sequence[:, RIGHT_SHOULDER, :]
        ) / 2.0

        # A fixed origin per clip preserves movement throughout the clip.
        origin = np.median(hip_midpoint, axis=0)
        torso_lengths = np.linalg.norm(
            shoulder_midpoint - hip_midpoint,
            axis=1,
        )
        scale = float(np.median(torso_lengths))

        if not np.isfinite(scale) or scale < 1e-6:
            raise ValueError(
                f"Invalid torso scale in {split_name} sequence {i}: {scale}. "
                "Check shoulder and hip keypoints."
            )

        normalised[i, :sequence_length] = (sequence - origin) / scale
        scales[i] = scale

    if not np.isfinite(normalised).all():
        raise ValueError(f"Normalised {split_name} data contains invalid values.")

    return normalised, scales


X_train_pos, train_torso_scales = normalise_sequences(
    X_train_raw, lengths_train_np, "training"
)
X_val_pos, val_torso_scales = normalise_sequences(
    X_val_raw, lengths_val_np, "validation"
)

print("\n========== BODY-RELATIVE NORMALISATION ==========")
print("Method: fixed per-sequence hip-centre / median torso length")
print("Median training torso scale:", float(np.median(train_torso_scales)))
print("Median validation torso scale:", float(np.median(val_torso_scales)))
print(
    "Normalised training coordinate range:",
    float(X_train_pos.min()),
    "to",
    float(X_train_pos.max()),
)


# ==========================================
# 6. CALCULATE VELOCITY FROM NORMALISED POSITIONS
# ==========================================

# Velocity is frame-to-frame displacement in normalised coordinates.
# The first frame has zero velocity because there is no previous frame.
velocity_train = np.zeros_like(X_train_pos, dtype=np.float32)
velocity_val = np.zeros_like(X_val_pos, dtype=np.float32)

velocity_train[:, 1:] = X_train_pos[:, 1:] - X_train_pos[:, :-1]
velocity_val[:, 1:] = X_val_pos[:, 1:] - X_val_pos[:, :-1]

# Keep trailing padded frames at zero.
velocity_train[~valid_train] = 0.0
velocity_val[~valid_val] = 0.0

# Fit per-keypoint/per-coordinate velocity scaling on TRAINING frames only.
train_velocity_values = velocity_train[valid_train]
velocity_mean = train_velocity_values.mean(axis=0).astype(np.float32)
velocity_std = train_velocity_values.std(axis=0).astype(np.float32)
velocity_std = np.where(velocity_std < 1e-6, 1.0, velocity_std).astype(np.float32)

velocity_train_scaled = (velocity_train - velocity_mean) / velocity_std
velocity_val_scaled = (velocity_val - velocity_mean) / velocity_std

# Keep padding and the first-frame velocity at exactly zero.
velocity_train_scaled[~valid_train] = 0.0
velocity_val_scaled[~valid_val] = 0.0
velocity_train_scaled[:, 0] = 0.0
velocity_val_scaled[:, 0] = 0.0

# Final representation: x', y', dx', dy' for each of 17 joints.
X_train_features = np.concatenate(
    [X_train_pos, velocity_train_scaled], axis=-1
).astype(np.float32)
X_val_features = np.concatenate(
    [X_val_pos, velocity_val_scaled], axis=-1
).astype(np.float32)

if not np.isfinite(X_train_features).all() or not np.isfinite(X_val_features).all():
    raise ValueError("Final Experiment 7 features contain NaN or infinity.")

print("\n========== NORMALISED POSITION + VELOCITY FEATURES ==========")
print("Training feature shape:", X_train_features.shape)
print("Validation feature shape:", X_val_features.shape)
print("Features per frame:", X_train_features.shape[2] * X_train_features.shape[3])
print("Velocity statistics fitted using training frames only.")
print(
    "Standardised velocity range (training):",
    float(velocity_train_scaled.min()),
    "to",
    float(velocity_train_scaled.max()),
)


# ==========================================
# 7. CONVERT TO PYTORCH AND CREATE LOADERS
# ==========================================

X_train_tensor = torch.from_numpy(X_train_features)
y_train_tensor = torch.from_numpy(y_train_np)
lengths_train_tensor = torch.from_numpy(lengths_train_np).long()

X_val_tensor = torch.from_numpy(X_val_features)
y_val_tensor = torch.from_numpy(y_val_np)
lengths_val_tensor = torch.from_numpy(lengths_val_np).long()

train_dataset = TensorDataset(
    X_train_tensor, y_train_tensor, lengths_train_tensor
)
val_dataset = TensorDataset(
    X_val_tensor, y_val_tensor, lengths_val_tensor
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)


# ==========================================
# 8. PACKED GRU + MEAN POOLING
# ==========================================

class BoxingGRU(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=68,  # 17 joints x (x', y', dx', dy')
            hidden_size=96,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25,
        )
        self.classifier = nn.Sequential(
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(96, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 25, 68)

        packed = pack_padded_sequence(
            x,
            sequence_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=25,
        )

        positions = torch.arange(25, device=output.device).unsqueeze(0)
        mask = positions < sequence_lengths.to(output.device).unsqueeze(1)
        mask = mask.unsqueeze(-1).to(output.dtype)

        pooled = (output * mask).sum(dim=1)
        denominator = sequence_lengths.to(output.device).unsqueeze(1).to(output.dtype)
        pooled = pooled / denominator.clamp_min(1.0)
        return self.classifier(pooled)


model = BoxingGRU(num_classes=len(classes)).to(DEVICE)


# ==========================================
# 9. CLASS-WEIGHTED LOSS (MATCH EXPERIMENT 6)
# ==========================================

class_counts = np.bincount(y_train_np, minlength=len(classes))
if np.any(class_counts == 0):
    raise ValueError("At least one class has no training examples.")

class_weights_np = len(y_train_np) / (len(classes) * class_counts)
class_weights = torch.tensor(
    class_weights_np, dtype=torch.float32, device=DEVICE
)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001,
)


# ==========================================
# 10. EVALUATION
# ==========================================

def evaluate():
    model.eval()
    predictions = []
    actual = []

    with torch.no_grad():
        for frames, labels, batch_lengths in val_loader:
            frames = frames.to(DEVICE)
            outputs = model(frames, batch_lengths)
            predicted = outputs.argmax(dim=1)
            predictions.extend(predicted.cpu().numpy().tolist())
            actual.extend(labels.numpy().tolist())

    accuracy = accuracy_score(actual, predictions)
    macro_f1 = f1_score(
        actual, predictions, average="macro", zero_division=0
    )
    return accuracy, macro_f1, actual, predictions


# ==========================================
# 11. TRAINING LOOP
# ==========================================

best_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0

print("\n========== STARTING EXPERIMENT 7 ==========")
print("Experiment: body-relative position + standardised velocity")
print("Loss: class-weighted cross entropy (same as Experiment 6)")
print("Checkpoint:", BEST_CHECKPOINT)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for frames, labels, batch_lengths in train_loader:
        frames = frames.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(frames, batch_lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / max(1, len(train_loader))
    accuracy, macro_f1, _, _ = evaluate()

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {average_loss:.4f} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Macro F1: {macro_f1:.4f}"
    )

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "validation_videos": sorted(VALIDATION_VIDEOS),
                "best_macro_f1": best_f1,
                "best_epoch": best_epoch,
                "architecture": "packed_gru_mean_pooling_normalized_position_velocity",
                "input_shape": [25, 17, 4],
                "normalisation_method": "fixed_per_sequence_hip_centre_median_torso_length",
                "keypoint_indices": {
                    "left_shoulder": LEFT_SHOULDER,
                    "right_shoulder": RIGHT_SHOULDER,
                    "left_hip": LEFT_HIP,
                    "right_hip": RIGHT_HIP,
                },
                "velocity_mean": torch.from_numpy(velocity_mean),
                "velocity_std": torch.from_numpy(velocity_std),
            },
            BEST_CHECKPOINT,
        )
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# ==========================================
# 12. FINAL VALIDATION REPORT
# ==========================================

if not BEST_CHECKPOINT.is_file():
    raise RuntimeError("Training ended without creating a checkpoint.")

checkpoint = torch.load(
    BEST_CHECKPOINT,
    map_location=DEVICE,
    weights_only=True,
)
model.load_state_dict(checkpoint["model_state_dict"])

accuracy, macro_f1, actual, predictions = evaluate()

print("\n========== EXPERIMENT 7 RESULTS ==========")
print("Best epoch:", checkpoint["best_epoch"])
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print("\nClassification report:\n")
print(
    classification_report(
        actual,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
    )
)


# ==========================================
# 13. CONFUSION MATRIX
# ==========================================

fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    actual,
    predictions,
    labels=list(range(len(classes))),
    display_labels=classes,
    xticks_rotation=45,
    cmap="Blues",
    ax=ax,
)
ax.set_title("Boxing Punch Classification - Normalised Position + Velocity GRU")
fig.tight_layout()
fig.savefig(CONFUSION_MATRIX_FILE, dpi=150)
plt.close(fig)

print("\nModel saved:")
print(BEST_CHECKPOINT)
print("\nConfusion matrix saved:")
print(CONFUSION_MATRIX_FILE)
print("\n========== EXPERIMENT 7 COMPLETE ==========")
