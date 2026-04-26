# Image Blurring Using Convolution

A NumPy-based image blurring model built from scratch using core mathematical concepts — matrices, weighted sums, and 2D discrete convolution. No black-box CV libraries; every step is transparent and auditable.

**Group 8** · Pragnaya Priyadarshini & Reeti Shah

---

## Overview

This project implements two convolution-based blurring filters — Box and Gaussian — applied to RGB images. The goal is to demonstrate the mathematical foundations of image filtering while measuring the quality impact of each approach using MSE and sharpness loss metrics.

---

## Features

- Box kernel and Gaussian kernel generation from scratch
- 2D discrete convolution engine with edge padding
- Full RGB image support via per-channel convolution
- Quantitative analysis: Mean Squared Error and Sharpness Loss
- Side-by-side visualization of original vs blurred outputs

---

## Tech Stack

| Library | Purpose |
|---|---|
| NumPy | All mathematical operations |
| Matplotlib | Visualization and output rendering |
| PIL (Pillow) | Image loading and saving |

---

## Project Structure

```
├── MathProject.ipynb       # Main notebook — full pipeline
├── blur_results.png        # Output: side-by-side comparison image
└── README.md
```

---

## How It Works

### 1. Image Representation
A digital image is loaded as an `(H × W × 3)` NumPy array. Pixel values are normalized from `[0, 255]` to `[0.0, 1.0]` for numerical stability.

### 2. Kernel Types

**Box Kernel** — equal weight for every neighbor:
```
kernel[i,j] = 1/k²   for all i, j
```

**Gaussian Kernel** — center-weighted, bell-curve distribution:
```
G(x,y) = exp( -(x² + y²) / (2σ²) )
```
Larger σ → wider blur. The kernel is normalized so all weights sum to 1.

### 3. Convolution
For every pixel `(r, c)`, a `k×k` region is extracted from the padded image and multiplied element-wise with the kernel. The sum of that product becomes the output pixel:

```
output[r,c] = Σᵢ Σⱼ image[r+i, c+j] × kernel[i,j]
```

Edge padding (`mode='edge'`) replicates border pixels to avoid darkening artifacts.

### 4. RGB Handling
Convolution operates on 2D arrays. The `convolve_rgb()` function splits the image into R, G, B channels, blurs each independently, then recombines them. Output is clipped to `[0.0, 1.0]`.

---

## Results

Tested on a 7×7 kernel:

| Metric | Box Blur 7×7 | Gaussian 7×7 (σ=2) |
|---|---|---|
| Sharpness Loss | 10.01% | 7.91% |
| MSE | 0.0129 | 0.0084 |

Gaussian consistently outperforms Box blur — ~35% lower pixel error and ~2% less sharpness loss — because center-weighted averaging better preserves edge information.

---

## Usage

1. Clone the repo and install dependencies:
```bash
pip install numpy matplotlib pillow
```

2. Place your image in the project directory and update the filename in `main()`:
```python
image = load_image("your_image.jpg")
```

3. Run the notebook or execute the script. Results are printed to console and saved as `blur_results.png`.

---

## Key Concepts

- **Normalization** — kernel weights must sum to 1 to conserve pixel energy (prevent brightness shifts)
- **Odd kernel size** — ensures a well-defined center pixel for symmetric convolution
- **σ (sigma)** — controls Gaussian spread; larger σ = softer, more diffuse blur
- **Sharpness Loss** — measured as percentage drop in pixel intensity standard deviation after blurring

---

## Real-World Applications

- Privacy protection (face/license plate blurring)
- Portrait mode background defocus
- Medical image noise reduction
- Pre-processing for computer vision pipelines
- Skin smoothing in beauty filters
vrd-xvay-sfw
