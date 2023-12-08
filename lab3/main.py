import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
from matplotlib import pyplot as plt
import pywt


def read_image(file_path):
    image = Image.open(file_path).convert('L')  # Convert to grayscale
    image = image.resize((512, 512))  # Resize to 512x512 or another power of 2
    return np.array(image)


def apply_dct(image, quantize=True):
    height, width = image.shape
    dct_image = np.zeros((height, width), dtype=np.float32)

    # Define a simple quantization matrix
    quantization_matrix = np.ones((8, 8)) * 50

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = image[i:i + 8, j:j + 8]
            # Apply DCT
            dct_block = dct_2d(block)

            # Quantize (simulate lossy compression)
            if quantize:
                dct_block = np.round(dct_block / quantization_matrix)

            # Dequantize (simulate decompression)
            dct_block = dct_block * quantization_matrix

            # Apply IDCT
            idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
            dct_image[i:i + 8, j:j + 8] = idct_block

    # Clip values to the 0-255 range
    dct_image = np.clip(dct_image, 0, 255)

    return dct_image.astype(np.uint8)


def dct_1d(signal):
    N = len(signal)
    result = np.zeros(N)
    factor = np.pi / N
    for k in range(N):
        sum_val = 0.0
        for n in range(N):
            sum_val += signal[n] * np.cos(factor * (n + 0.5) * k)
        result[k] = sum_val
    # Weighting factors
    result[0] = result[0] / np.sqrt(N)
    result[1:] = result[1:] * np.sqrt(2 / N)
    return result


def dct_2d(block):
    height, width = block.shape
    # Apply the DCT to each row
    dct_temp = np.zeros_like(block, dtype=float)
    for i in range(height):
        dct_temp[i, :] = dct_1d(block[i, :])
    # Apply the DCT to each column
    dct_block = np.zeros_like(block, dtype=float)
    for i in range(width):
        dct_block[:, i] = dct_1d(dct_temp[:, i])
    return dct_block


def apply_dwt(image):
    coeffs2 = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs2

    # Reconstructing the image using the approximate coefficients
    dwt_image = pywt.idwt2((cA, (None, None, None)), 'haar')

    return dwt_image


def mse(imageA, imageB):
    # The 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images.
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err


def calculate_psnr(original_image, compressed_image):
    # Compute MSE between the images
    err = mse(original_image, compressed_image)

    # Avoid division by zero
    if err == 0:
        return 100

    # Calculate PSNR
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(err))

    return psnr


def display_results(original, dct_compressed, haar_compressed):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(dct_compressed, cmap='gray')
    plt.title('DCT Compressed')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(haar_compressed, cmap='gray')
    plt.title('Haar Compressed')
    plt.axis('off')

    plt.show()


def main():
    image_path = 'image/boat.png'
    original_image = read_image(image_path)

    dct_image = apply_dct(original_image)
    dwt_image = apply_dwt(original_image)

    psnr_dct = calculate_psnr(original_image, dct_image)
    psnr_haar = calculate_psnr(original_image, dwt_image)

    print(f"PSNR for DCT: {psnr_dct} dB")
    print(f"PSNR for Haar: {psnr_haar} dB")

    # Display images and PSNR values
    display_results(original_image, dct_image, dwt_image)


if __name__ == "__main__":
    main()
