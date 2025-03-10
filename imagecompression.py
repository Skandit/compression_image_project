import os
import numpy as np
from PIL import Image


class ImageCompression:
    def __init__(self, filename):
        self.filename = filename

    def compress_image(self):
        """Compress an image using LZW algorithm and save the binary file."""
        input_path = self.filename + '.png'  # Ensure PNG format
        output_path = self.filename + '.bin'

        # Load image and convert to grayscale
        img = Image.open(input_path).convert('L')
        img_data = np.array(img).flatten()

        # Encode image data using LZW
        compressed_data = self.lzw_encode(img_data.tolist())

        # Save compressed binary file
        with open(output_path, 'wb') as out_file:
            np.save(out_file, compressed_data)

        # Calculate compression metrics
        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(output_path)
        entropy = self.calculate_entropy(img_data)
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
        output_path = self.filename + '_restored.png'

        # Load compressed data
        with open(input_path, 'rb') as in_file:
            compressed_data = np.load(in_file, allow_pickle=True)

        # Decode compressed data
        decompressed_data = self.lzw_decode(compressed_data)

        # Restore image
        img_size = int(np.sqrt(len(decompressed_data)))  # Assuming square images
        img_array = np.array(decompressed_data, dtype=np.uint8).reshape((img_size, img_size))
        restored_img = Image.fromarray(img_array, mode='L')
        restored_img.save(output_path)

        print(f"Image {input_path} is decompressed into {output_path}.")

        return output_path

    def lzw_encode(self, data):
        """LZW compression algorithm."""
        dict_size = 256
        dictionary = {i: [i] for i in range(dict_size)}
        w = []
        result = []
        for c in data:
            wc = w + [c]
            if tuple(wc) in dictionary:
                w = wc
            else:
                result.append(dictionary[tuple(w)])
                dictionary[tuple(wc)] = dict_size
                dict_size += 1
                w = [c]
        if w:
            result.append(dictionary[tuple(w)])
        return result

    def lzw_decode(self, compressed):
        """LZW decompression algorithm."""
        dict_size = 256
        dictionary = {i: [i] for i in range(dict_size)}
        w = dictionary[compressed.pop(0)]
        result = w[:]
        for k in compressed:
            if k in dictionary:
                entry = dictionary[k]
            elif k == dict_size:
                entry = w + [w[0]]
            else:
                raise ValueError("Bad compressed k: %s" % k)
            result.extend(entry)
            dictionary[dict_size] = w + [entry[0]]
            dict_size += 1
            w = entry
        return result

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
    original_image = Image.open(filename + ".png").convert('L')
    decompressed_image = Image.open(decompressed_file).convert('L')

    original_array = np.array(original_image)
    decompressed_array = np.array(decompressed_image)

    if np.array_equal(original_array, decompressed_array):
        print("\nVerification: SUCCESS - The decompressed image matches the original.")
    else:
        print("\nVerification: FAILURE - The decompressed image does NOT match the original.")
