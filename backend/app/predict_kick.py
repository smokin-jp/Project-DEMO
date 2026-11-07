
from pathlib import Path

import clip
import joblib
import torch
from PIL import Image


# Load the saved classifier and threshold.
MODEL_PATH = Path("./models/kick_detector.joblib")

saved_model = joblib.load(MODEL_PATH)

classifier = saved_model["classifier"]
threshold = saved_model["threshold"]


# Load CLIP.
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()


def predict_kick(image_path: str) -> dict:
    """
    Predict whether an image contains a kick being delivered.
    """

    image = Image.open(image_path).convert("RGB")

    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = clip_model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    embedding = embedding.cpu().numpy()

    kick_score = float(classifier.predict_proba(embedding)[0][1])

    kick_detected = kick_score >= threshold

    return {
        "kick_detected": kick_detected,
        "confidence": round(kick_score, 4),
        "threshold": float(threshold),
        "decision": "FOUL" if kick_detected else "NO FOUL"
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python ./predict_kick.py <image_path>")
        raise SystemExit(1)

    result = predict_kick(sys.argv[1])

    print("\nPREDICTION")
    print("----------")
    for key, value in result.items():
        print(f"{key}: {value}")
