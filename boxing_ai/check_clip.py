import torch
import clip

# Select GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# Load pretrained CLIP
model, preprocess = clip.load(
    "ViT-B/32",
    device=device
)

model.eval()

# Test image encoding
dummy_image = torch.randn(
    1, 3, 224, 224
).to(device)

with torch.no_grad():
    features = model.encode_image(dummy_image)

print("CLIP successfully loaded!")
print("Feature dimensions:", features.shape)