# Gaussian Blur & FFT Convolution

This project demonstrates image processing techniques using Fast Fourier Transforms (FFT) in Python. It implements 2D convolution to apply various blur kernels (Box, Gaussian, Linear) to images.

## Features

- **FFT-based Convolution:** Efficiently computes convolution using the Convolution Theorem ($f * g \iff \mathcal{F}^{-1}(\mathcal{F}(f) \cdot \mathcal{F}(g))$).
- **Multiple Kernels:**
  - **Box (Mean):** Standard averaging blur.
  - **Gaussian:** Smooth, bell-curve blur based on sigma.
- **Batch Processing:** `main_refactored.py` automatically iterates over different kernel sizes and types.

## Prerequisites

- **Python:** >= 3.13
- **uv:** A fast Python package and project manager.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd gaussian_blur
    ```

2.  **Install dependencies:**
    This project uses `uv` for dependency management.
    ```bash
    uv sync
    ```

## Usage

### 1. Refactored Script (Recommended)

The refactored script (`main_refactored.py`) allows you to loop over multiple kernel sizes and types automatically.

**Run the script:**
```bash
uv run main_refactored.py
```

**What it does:**
1.  Loads `photos/nice_dog.JPG`.
2.  Iterates through kernel sizes $p = [10, 20, 50]$.
3.  Iterates through kernel types: `box`, `gaussian`, `linear`.
4.  Applies the convolution using FFT.
5.  Saves the output images as `output_<type>_p<size>.png` in the project root.

### 2. Original Script

The original script (`main.py`) performs a single convolution with a fixed box kernel.

```bash
uv run main.py
```

## Project Structure

- `main_refactored.py`: Main entry point with functions for kernel generation and batch processing.
- `main.py`: Original script (single execution).
- `photos/`: Directory containing input images.
- `pyproject.toml`: Dependency configuration.

## Theory

The convolution is performed in the frequency domain for efficiency:

1.  **Pad** the image and kernel to the same size (typically the next power of 2 or combined dimensions).
2.  **Transform** both to the frequency domain using `fft2`.
3.  **Multiply** the transformed matrices element-wise.
4.  **Inverse Transform** (`ifft2`) the result back to the spatial domain.
5.  **Clip** values to the 0-255 range to handle floating-point noise.
