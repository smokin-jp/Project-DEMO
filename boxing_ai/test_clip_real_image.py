import sys
from pathlib import Path

from PIL import Image

from fusion_model import CLIPExtractor, CLASSES

if len(sys.argv) != 2:
    print("Usage: python test_clip_real_image.py IMAGE_PATH")
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.is_file():
    print("Image not found:", image_path)
    sys.exit(1)

print("Loading CLIP...")
model = CLIPExtractor()

image = Image.open(image_path).convert("RGB")

print("Processing:", image_path)

scores = model.classify_images([image])[0]

ranking = scores.argsort(descending=True)

print("\n=== CLIP PUNCH RANKINGS ===")

for index in ranking.tolist():
    print(f"{CLASSES[index]}: {scores[index].item():.4f}")

print("\nCLIP real-image test completed.")