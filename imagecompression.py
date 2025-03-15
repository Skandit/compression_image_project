import os
import numpy as np
from PIL import Image
from lzwproject import LZWCode

class ImageCompression:
    def __init__(self, filename):
        self.filename = filename
        self.lzw = LZWCode(filename, "image")

    def compress_image(self):
        """Compress an image using LZW algorithm and save the binary file."""
        input_path = self.filename + '.bmp'  # Ensure PNG format
        output_path = self.filename + '.bin'
        shape_path = self.filename + '_shape.npy'  # Save original shape

        # Load image and convert to grayscale
        img = Image.open(input_path).convert('L')
        img_data = np.array(img)
        original_shape = img_data.shape  # Store original shape
        flattened_data = img_data.flatten()

        # Encode image data using LZW
        compressed_data = self.lzw.encode_pic(flattened_data.tolist())

        # Save compressed binary file and original shape
        with open(output_path, "wb") as f:
            for code in compressed_data:
                f.write(code.to_bytes(2, byteorder="big"))  # Store as 2-byte values

        # Calculate compression metrics
        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(output_path)
        entropy = self.calculate_entropy(flattened_data)
        average_code_length = np.mean([len(bin(code)[2:]) for code in compressed_data])
        compression_ratio = compressed_size / original_size if original_size > 0 else 0

        print(f"Image {input_path} is compressed into {output_path}.")
        print("\nCompression Statistics:")
        print(f"Original Size: {original_size} bytes")
        print(f"Compressed Size: {compressed_size} bytes")
        print(f"Entropy: {entropy:.4f}")
        print(f"Average Code Length: {average_code_length:.2f} bits")
        print(f"Compression Ratio: {compression_ratio:.4f}")

        return output_path

    def decompress_image(self):
        """Decompress an LZW-compressed binary file and restore the image."""
        input_path = self.filename + '.bin'
        shape_path = self.filename + '_shape.npy'
        output_path = self.filename + '_restored.bmp'

        # Load compressed data and original shape
        with open(input_path, "rb") as f:
            compressed_data = [int.from_bytes(f.read(2), byteorder="big") for _ in
                               range(os.path.getsize(input_path) // 2)]

        # Decode compressed data
        decompressed_data = self.lzw.decode_pic(compressed_data)

        # Restore image using the original shape
        restored_img = Image.fromarray(decompressed_data.astype(np.uint8), mode='L')
        restored_img.save(output_path)

        print(f"Image {input_path} is decompressed into {output_path}.")

        return output_path

    def calculate_entropy(self, data):
        """Calculate entropy of an image."""
        hist, _ = np.histogram(data, bins=256, range=(0, 255))
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

if __name__ == "__main__":
    filename = "sample_image"  # Change this to your image file name without extension
    compressor = ImageCompression(filename)
    compressed_file = compressor.compress_image()
    decompressed_file = compressor.decompress_image()

    # Compare original and decompressed images
    original_image = Image.open(filename + ".bmp").convert('L')
    decompressed_image = Image.open(decompressed_file).convert('L')

    original_array = np.array(original_image)
    decompressed_array = np.array(decompressed_image)

    if np.array_equal(original_array, decompressed_array):
        print("\nVerification: SUCCESS - The decompressed image matches the original.")
    else:
        print("\nVerification: FAILURE - The decompressed image does NOT match the original.")