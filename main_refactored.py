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
        kernel_type (str): "box", "gaussian", or "linear".
        sigma (float): Sigma for gaussian kernel. Defaults to size/6 if None.

    Returns:
        np.ndarray: The normalized kernel matrix.
    """
    if kernel_type == "box":
        return np.ones((size, size)) / (size**2)

    elif kernel_type == "gaussian":
        if sigma is None:
            sigma = size / 4.0  # godtycklig std

        # Skapa ett rutnät av värden som är linjärt fördelade mellan -1 till 1
        ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
        xx, yy = np.meshgrid(ax, ax)

        # Gör värdena normalfördelade
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

    # 1. Anpassa padding till den största sidan av bilden
    pad_rows_img = max_dim - rows
    pad_cols_img = max_dim - cols
    img_padded = np.pad(
        img_array, ((0, pad_rows_img), (0, pad_cols_img)), "constant", constant_values=0
    )

    # 2. Fixa padding för kerneln, så att dimensionerna matchar
    k_rows, k_cols = kernel.shape
    pad_rows_k = max_dim - k_rows
    pad_cols_k = max_dim - k_cols
    kernel_padded = np.pad(
        kernel, ((0, pad_rows_k), (0, pad_cols_k)), "constant", constant_values=0
    )

    # 3. FFT
    img_f = np.fft.fft2(img_padded)
    k_f = np.fft.fft2(kernel_padded)

    # 4. Multiplicera och applicera FFT-invers
    output_f = img_f * k_f
    output = np.real(np.fft.ifft2(output_f))

    return output


def main():
    image_path = "photos/nice_dog.JPG"
    img = load_image(image_path)

    # B är bildmatrisen
    B = np.asarray(img)

    # p-värden är olika linjära omskalningar av initalvärdena i kärnan (enhetsmatris)
    p_values = [i for i in range(1, 20)]
    kernel_types = ["box", "gaussian"]

    print(f"Bild: {image_path} (Storlek: {B.shape})")

    for k_type in kernel_types:
        for p in p_values:
            print(f"Applicerar {k_type}-konvolution med p={p}...")

            # gör en kärna
            kernel = get_kernel(p, kernel_type=k_type)

            # kör konvolutionsfunktionen
            B_output = apply_convolution(B, kernel)

            # clipping gör att värdena håller sig i intervallet [0,255]
            # konvertera tillbaka till bild
            B_output_clipped = np.clip(B_output, 0, 255).astype(np.uint8)
            output_img = Image.fromarray(B_output_clipped)

            # Save the result
            output_filename = f"output/output_{k_type}_p{p}.png"
            output_img.save(output_filename)
            print(f"  -> sparat till {output_filename}")

    print("Klart.")


if __name__ == "__main__":
    main()
