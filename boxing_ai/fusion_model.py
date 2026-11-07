
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import clip

from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)


# ==========================================
# 1. SETTINGS
# ==========================================

DEVICE = torch.device("cpu")

CHECKPOINT_PATH = Path(
    "checkpoints/skeleton_gru_normalized_velocity_best.pth"
)

NUM_CLASSES = 6
CLIP_FEATURES = 512
SKELETON_FEATURES = 192

CLASSES = [
    "cross",
    "jab",
    "lead hook",
    "lead uppercut",
    "rear hook",
    "rear uppercut",
]


# ==========================================
# 2. SKELETON FEATURE EXTRACTOR
# ==========================================

class BoxingGRU(nn.Module):

    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=68,
            hidden_size=96,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25,
        )

        # This is the same classifier structure as
        # the original trained model.
        self.classifier = nn.Sequential(
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(96, NUM_CLASSES),
        )

    def extract_features(self, x, sequence_lengths):

        batch_size = x.shape[0]

        # (batch, 25, 17, 4) -> (batch, 25, 68)
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

        # Mask padded frames before mean pooling.
        positions = torch.arange(
            25,
            device=output.device,
        ).unsqueeze(0)

        mask = (
            positions
            < sequence_lengths.to(output.device).unsqueeze(1)
        )

        mask = mask.unsqueeze(-1).to(output.dtype)

        pooled = (output * mask).sum(dim=1)

        denominator = (
            sequence_lengths
            .to(output.device)
            .unsqueeze(1)
            .to(output.dtype)
        )

        pooled = pooled / denominator.clamp_min(1.0)

        return pooled

    def forward(self, x, sequence_lengths):

        features = self.extract_features(
            x,
            sequence_lengths,
        )

        return self.classifier(features)



# ==========================================
# 3. CLIP ZERO-SHOT PUNCH CLASSIFIER
# ==========================================

class CLIPExtractor:

    def __init__(self):

        self.model, self.preprocess = clip.load(
            "ViT-B/32",
            device=DEVICE,
        )

        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        # Descriptions must follow the exact GRU class order.
        descriptions = [
        "a straight rear-hand cross punch with the rear shoulder and hip rotating forward",
        "a straight lead-hand jab with the front arm extending forward and rear hand guarding the face",
        "a lead-hand hook with the elbow bent and fist travelling horizontally in an arc",
        "a lead-hand uppercut with the elbow bent and fist travelling upwards from below",
        "a rear-hand hook with the elbow bent and fist travelling horizontally in an arc",
        "a rear-hand uppercut with the elbow bent and fist travelling upwards from below",
        ]

        templates = [
            "a photo of a boxer performing a {}.",
            "a frame from a boxing match showing a {}.",
            "a boxer throwing a {}.",
        ]

        prompts = [
            template.format(description)
            for description in descriptions
            for template in templates
        ]

        tokens = clip.tokenize(prompts).to(DEVICE)

        with torch.no_grad():

            text_features = self.model.encode_text(tokens).float()

            text_features = text_features / text_features.norm(
                dim=-1,
                keepdim=True,
            )

            # Average the three descriptions for each punch.
            class_features = []

            for i in range(NUM_CLASSES):

                start = i * len(templates)
                end = start + len(templates)

                class_feature = text_features[start:end].mean(dim=0)

                class_feature = class_feature / class_feature.norm()

                class_features.append(class_feature)

            self.text_features = torch.stack(class_features)

    @torch.no_grad()
    def classify_images(self, images):

        """
        images: List of RGB PIL images.

        Returns:
            CLIP scores with shape (1, 6).
        """

        image_tensors = torch.stack([
            self.preprocess(image)
            for image in images
        ]).to(DEVICE)

        image_features = self.model.encode_image(
            image_tensors
        ).float()

        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True,
        )

        # Compare each image against the six punch descriptions.
        similarities = 100.0 * (
            image_features @ self.text_features.T
        )

        # Average the scores across the supplied images.
        return similarities.mean(dim=0, keepdim=True)



# ==========================================
# 4. BOXING FUSION MODEL
# ==========================================

class BoxingFusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.skeleton_model = BoxingGRU()

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=True,
        )

        # Reuse the original GRU training statistics.
        self.velocity_mean = (
            checkpoint["velocity_mean"].cpu().numpy()
        )

        self.velocity_std = (
            checkpoint["velocity_std"].cpu().numpy()
        )

        self.keypoint_indices = checkpoint["keypoint_indices"]

        self.classes = checkpoint["classes"]

        if self.classes != CLASSES:
            raise ValueError(
                "Checkpoint class order does not match."
            )

        self.skeleton_model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # Freeze the existing trained GRU.
        for parameter in self.skeleton_model.parameters():
            parameter.requires_grad = False

        self.skeleton_model.eval()

        # Initial fusion weights.
        self.gru_weight = 0.80
        self.clip_weight = 0.20


    def preprocess_skeleton(self, raw_sequence):

        """
        Converts one original BoxingVI skeleton sequence
        into the normalised format expected by the GRU.

        Input:  (25, 17, 2)
        Output: (1, 25, 17, 4), plus sequence length
        """

        sequence = np.asarray(
            raw_sequence,
            dtype=np.float32,
        )

        if sequence.shape != (25, 17, 2):
            raise ValueError(
                f"Unexpected skeleton shape: {sequence.shape}"
            )

        if not np.isfinite(sequence).all():
            raise ValueError("Skeleton contains invalid values.")

        # Identify the valid frames.
        frame_valid = np.any(
            np.abs(sequence) > 1e-8,
            axis=(1, 2),
        )

        valid_indices = np.flatnonzero(frame_valid)

        if len(valid_indices) == 0:
            raise ValueError("Skeleton sequence is empty.")

        if valid_indices[0] != 0:
            raise ValueError(
                "Sequence contains leading empty frames."
            )

        length = int(valid_indices[-1] + 1)

        # Obtain keypoint indices from the checkpoint.
        indices = self.keypoint_indices

        left_shoulder = indices["left_shoulder"]
        right_shoulder = indices["right_shoulder"]
        left_hip = indices["left_hip"]
        right_hip = indices["right_hip"]

        valid_sequence = sequence[:length]

        # Calculate hip and shoulder centres.
        hip_midpoint = (
            valid_sequence[:, left_hip]
            + valid_sequence[:, right_hip]
        ) / 2.0

        shoulder_midpoint = (
            valid_sequence[:, left_shoulder]
            + valid_sequence[:, right_shoulder]
        ) / 2.0

        # Calculate the hip-centred origin.
        origin = np.median(
            hip_midpoint,
            axis=0,
        )

        # Calculate torso-length scaling.
        torso_lengths = np.linalg.norm(
            shoulder_midpoint - hip_midpoint,
            axis=1,
        )

        scale = float(np.median(torso_lengths))

        if not np.isfinite(scale) or scale < 1e-6:
            raise ValueError("Invalid torso length.")

        # Normalise positions.
        positions = np.zeros(
            (25, 17, 2),
            dtype=np.float32,
        )

        positions[:length] = (
            valid_sequence - origin
        ) / scale

        # Calculate frame-to-frame velocities.
        velocities = np.zeros_like(positions)

        velocities[1:length] = (
            positions[1:length]
            - positions[:length - 1]
        )

        # Standardise velocities using saved training statistics.
        std = np.where(
            np.abs(self.velocity_std) < 1e-6,
            1.0,
            self.velocity_std,
        )

        velocities = (
            velocities - self.velocity_mean
        ) / std

        # Match the original training pipeline.
        velocities[0] = 0.0
        velocities[length:] = 0.0

        # Combine position and velocity features.
        features = np.concatenate(
            [positions, velocities],
            axis=-1,
        )

        features = torch.from_numpy(
            features
        ).unsqueeze(0).to(DEVICE)

        sequence_lengths = torch.tensor(
            [length],
            dtype=torch.long,
            device=DEVICE,
        )

        return features, sequence_lengths

    def forward(
        self,
        clip_logits,
        skeleton_sequences,
        sequence_lengths,
    ):

        # Obtain predictions from the trained GRU.
        with torch.no_grad():

            skeleton_logits = self.skeleton_model(
                skeleton_sequences,
                sequence_lengths,
            )

        # Convert both outputs into probabilities.
        gru_probabilities = torch.softmax(
            skeleton_logits,
            dim=1,
        )

        clip_probabilities = torch.softmax(
            clip_logits,
            dim=1,
        )

        # Combine the two models.
        predictions = (
            self.gru_weight * gru_probabilities
            + self.clip_weight * clip_probabilities
        )

        return predictions


# ==========================================
# 5. INITIALISE THE MODELS
# ==========================================

if __name__ == "__main__":

    print("Initialising CLIP...")

    clip_extractor = CLIPExtractor()

    print("CLIP loaded successfully.")

    print("\nInitialising skeleton model...")

    fusion_model = BoxingFusionModel()


    print("Skeleton checkpoint loaded successfully.")

    print("\nFusion architecture:")
    print(fusion_model)

    print("\nCLIP features:", CLIP_FEATURES)
    print("Skeleton features:", SKELETON_FEATURES)
    print(
        "Combined features:",
        CLIP_FEATURES + SKELETON_FEATURES,
    )

    print("\nAll model components initialised successfully.")


    print("\n=== TESTING SCORE-LEVEL FUSION ===")

    # Dummy CLIP class scores (six classes).
    dummy_clip_logits = torch.randn(1, NUM_CLASSES)

    # Dummy skeleton input.
    dummy_skeleton = torch.randn(1, 25, 17, 4)

    dummy_lengths = torch.tensor([25])

    fusion_model.eval()

    with torch.no_grad():

        predictions = fusion_model(
            dummy_clip_logits,
            dummy_skeleton,
            dummy_lengths,
        )

    print("CLIP scores:", tuple(dummy_clip_logits.shape))
    print("Skeleton input:", tuple(dummy_skeleton.shape))
    print("Prediction output:", tuple(predictions.shape))

    print(
        "Probability total:",
        predictions.sum(dim=1).item(),
    )

    print("\nScore-level fusion successful!")
