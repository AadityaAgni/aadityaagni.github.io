# Project 1: Colorizing the Prokudin-Gorskii Photos

Aligns the blue, green, and red glass-plate exposures into a single color image.

## Setup

Requires numpy, opencv, torch, and torchvision

## Usage

```bash
# Multi-scale pyramid alignment (use for the full-size .tif plates)
python colorize.py --imname data/church.tif --out results/church.jpg

# Single-scale search over [-15, 15]
python colorize.py --imname data/cathedral.jpg --out results/cathedral.jpg --single-scale
```


Website: https://aadityaagni.github.io/proj1/