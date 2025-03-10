from PIL import Image
import os
from lzwproject import LZWCode  # Using the existing LZW class


class LZWImageCompression:
    def __init__(self, filename):
        self.filename = filename  # Image filename without extension
        self.image_path = filename + ".png"  # PNG image file
        self.compressed_path = filename + ".lzw"  # Compressed file
        self.decompressed_path = filename + "_decompressed.png"  # Decompressed output

    def compress_image(self):
        # Load image
        img = Image.open(self.image_path).convert("L")  # Convert to grayscale
        pixels = list(img.getdata())  # Extract pixel values

        # Apply LZW Compression
        lzw = LZWCode(self.filename, "image")
        compressed_data = lzw.encode(pixels)

        # Save compressed data
        with open(self.compressed_path, "w") as f:
            f.write(" ".join(map(str, compressed_data)))

        print(f"Image compressed and saved as {self.compressed_path}")

    def decompress_image(self):
        # Read compressed data
        with open(self.compressed_path, "r") as f:
            compressed_data = list(map(int, f.read().split()))

        # Apply LZW Decompression
        lzw = LZWCode(self.filename, "image")
        decompressed_pixels = lzw.decode(compressed_data)

        # Convert pixel values back to an image
        img = Image.new("L", (256, int(len(decompressed_pixels) / 256)))  # Assuming width=256
        img.putdata(list(map(ord, decompressed_pixels)))
        img.save(self.decompressed_path)

        print(f"Image decompressed and saved as {self.decompressed_path}")


# Run the image compression
if __name__ == "__main__":
    filename = input("Enter the image filename (without extension): ")
    lzw_img = LZWImageCompression(filename)
    lzw_img.compress_image()
    lzw_img.decompress_image()
