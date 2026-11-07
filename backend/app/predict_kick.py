"""Command-line check of the trained kick detector on one picture:   python -m app.predict_kick <image_path>   (from ./backend)

(The original version used the `clip` package and a working-directory-relative model path; this one shares the backend's
transformers-based ViT-B/32 encoder and finds the model file itself, so it works from anywhere.)
"""
import sys

try:
    from .b32_clip import B32Embedder
    from .kick_clf import KickClassifier
except ImportError:  # run as a plain script:  python backend/app/predict_kick.py <image>
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.b32_clip import B32Embedder
    from app.kick_clf import KickClassifier


def predict_kick(image_path: str, _cache: dict = {}) -> dict:
    """Predict whether an image contains a kick being delivered."""
    if "m" not in _cache:
        _cache["m"] = KickClassifier.load(B32Embedder().load())
    return _cache["m"].predict_image(image_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m app.predict_kick <image_path>")
        raise SystemExit(1)
    print("\nPREDICTION\n----------")
    for key, value in predict_kick(sys.argv[1]).items():
        print(f"{key}: {value}")
