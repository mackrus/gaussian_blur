import numpy as np
from PIL import Image


def load_image(path):
    """Loads an image and converts it to grayscale."""
    return Image.open(path).convert("L")


def get_kernel(size, kernel_type="box", sigma=None):
    """
    Generates a kernel matrix of the specified size and type.

    Args:
        size (int): The size of the kernel (size x size).
        kernel_type (str): "box" or "gaussian".
        sigma (float): Sigma for gaussian kernel. Defaults to size/4 if None.

    Returns:
        np.ndarray: The normalized kernel matrix.
    """
    if kernel_type == "box":
        return np.ones((size, size)) / (size**2)

    elif kernel_type == "gaussian":
        if sigma is None:
            sigma = size / 4.0  # Arbitrary standard deviation

        # Create a grid of values centered around 0
        ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
        xx, yy = np.meshgrid(ax, ax)

        # Apply Gaussian distribution to the grid values
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        return kernel / np.sum(kernel)

    else:
        return None


def apply_convolution(img_array, kernel):
    """
    Applies convolution between an image and a kernel using FFT.
    Pads the image and kernel to the square size of max(img_dims).
    """
    rows, cols = img_array.shape
    max_dim = max(rows, cols)

    # 1. Pad the image to match the largest dimension
    pad_rows_img = max_dim - rows
    pad_cols_img = max_dim - cols
    img_padded = np.pad(
        img_array, ((0, pad_rows_img), (0, pad_cols_img)), "constant", constant_values=0
    )

    # 2. Pad the kernel to match the image dimensions
    k_rows, k_cols = kernel.shape
    pad_rows_k = max_dim - k_rows
    pad_cols_k = max_dim - k_cols
    kernel_padded = np.pad(
        kernel, ((0, pad_rows_k), (0, pad_cols_k)), "constant", constant_values=0
    )

    # 3. Compute FFT
    img_f = np.fft.fft2(img_padded)
    k_f = np.fft.fft2(kernel_padded)

    # 4. Multiply in frequency domain and compute inverse FFT
    output_f = img_f * k_f
    output = np.real(np.fft.ifft2(output_f))

    return output


def main():
    # Note: Ensure the image path is correct relative to the script execution
    # Changed extension to .JPG based on file listing, assuming case sensitivity on some systems
    image_path = "photos/nice_dog.JPG"
    
    try:
        img = load_image(image_path)
    except FileNotFoundError:
        print(f"Error: Could not find image at {image_path}")
        return

    # Convert image to numpy array
    image_matrix = np.asarray(img)

    # Define kernel sizes (p)
    p_values = [i for i in range(1, 20)]
    kernel_types = ["box", "gaussian"]

    print(f"Image: {image_path} (Size: {image_matrix.shape})")

    for k_type in kernel_types:
        for p in p_values:
            print(f"Applying {k_type} convolution with p={p}...")

            # Generate a kernel of size p x p
            kernel = get_kernel(p, kernel_type=k_type)

            # Apply convolution
            output_matrix = apply_convolution(image_matrix, kernel)

            # Clip values to [0, 255] and convert to image
            output_matrix_clipped = np.clip(output_matrix, 0, 255).astype(np.uint8)
            output_img = Image.fromarray(output_matrix_clipped)

            # Save the result
            output_filename = f"output/output_{k_type}_p{p}.png"
            output_img.save(output_filename)
            print(f"  -> Saved to {output_filename}")

    print("Done.")


if __name__ == "__main__":
    main()