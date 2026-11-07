# Boxing AI - CLIP + GRU

Prototype boxing punch recognition using OpenAI CLIP
and a trained bidirectional GRU skeleton model.

## Six punch classes

1. Cross
2. Jab
3. Lead Hook
4. Lead Uppercut
5. Rear Hook
6. Rear Uppercut

## Included

- CLIP ViT-B/32 zero-shot image classification.
- Existing trained skeleton GRU.
- Skeleton preprocessing using saved checkpoint statistics.
- Initial CLIP and GRU score-level fusion.
- Real-image CLIP test.
- Selected GRU training and dataset preparation scripts.

## Model details

CLIP is currently frozen and compares RGB images with
six punch descriptions.

The GRU uses 17 keypoints, 25 frames and normalised
position and velocity features.

The fusion currently uses initial weights of:
- GRU: 80%
- CLIP: 20%

These weights have not been calibrated against a paired
RGB and skeleton validation dataset.

## Installation

Python 3.12 is the development environment used.

Create a virtual environment:

    python -m venv .venv
    source .venv/bin/activate

Install PyTorch and torchvision using the official
installation selector:

    https://pytorch.org/get-started/locally/

Install the remaining dependencies:

    pip install -r requirements.txt

CLIP downloads its pretrained weights on first use.

## Run the model test

    python fusion_model.py

## Test an RGB image

    python test_clip_real_image.py /path/to/boxing_image.jpg

## Current limitations

This repository contains a model prototype, not a
completed video processing application.

Automatic video pose extraction, RGB/skeleton alignment,
FastAPI video upload and comprehensive six-class accuracy
evaluation have not yet been completed.

CLIP has passed software and individual image tests,
but its general boxing recognition accuracy is unverified.

The GRU checkpoint recorded a preliminary validation
macro F1 of approximately 0.428 on the V1/V4 split.

## Dataset

BoxingVI data and annotations are not included.
Obtain them separately through an authorised source.

Check dataset terms before redistributing the trained
checkpoint or dataset-derived material.
