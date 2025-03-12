import numpy as np
import cv2
import pickle
import os


def get_file_size(file_path):
    """Returns file size in bytes"""
    return os.path.getsize(file_path) if os.path.exists(file_path) else 0


class LZWCompression:
    def __init__(self):
        self.dictionary_size = 256
        self.dictionary = {chr(i): i for i in range(self.dictionary_size)}
        self.reverse_dictionary = {i: chr(i) for i in range(self.dictionary_size)}

    def compress(self, data):
        """Compress the input data using LZW"""
        w = ""
        compressed_data = []
        for c in data:
            wc = w + c
            if wc in self.dictionary:
                w = wc
            else:
                compressed_data.append(self.dictionary[w])
                self.dictionary[wc] = self.dictionary_size
                self.dictionary_size += 1
                w = c
        if w:
            compressed_data.append(self.dictionary[w])
        return compressed_data

    def decompress(self, compressed_data):
        """Decompress the LZW encoded data"""
        w = self.reverse_dictionary[compressed_data.pop(0)]
        decompressed_data = w
        for k in compressed_data:
            if k in self.reverse_dictionary:
                entry = self.reverse_dictionary[k]
            elif k == self.dictionary_size:
                entry = w + w[0]
            else:
                raise ValueError("Bad compressed k: %s" % k)
            decompressed_data += entry
            self.reverse_dictionary[self.dictionary_size] = w + entry[0]
            self.dictionary_size += 1
            w = entry
        return decompressed_data


def compute_difference_image(image):
    """Compute the difference image"""
    diff_image = np.zeros_like(image, dtype=np.int16)
    diff_image[:, 1:] = image[:, 1:] - image[:, :-1]  # Row-wise difference
    diff_image[1:, 0] = image[1:, 0] - image[:-1, 0]  # Column-wise difference
    return diff_image


def restore_original_image(diff_image):
    """Restore original image from difference image"""
    restored_image = np.zeros_like(diff_image, dtype=np.uint8)
    restored_image[:, 0] = diff_image[:, 0]  # First column
    for j in range(1, diff_image.shape[1]):
        restored_image[:, j] = restored_image[:, j - 1] + diff_image[:, j]  # Row-wise restoration
    for i in range(1, diff_image.shape[0]):
        restored_image[i, 0] = restored_image[i - 1, 0] + diff_image[i, 0]  # Column-wise restoration
    return restored_image


def save_compressed_file(filename, compressed_data):
    with open(filename, 'wb') as f:
        pickle.dump(compressed_data, f)


def load_compressed_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def compress_image(image_path, output_path):
    """Compress an image using LZW and difference encoding"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image file '{image_path}' not found.")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Error: Unable to read image at {image_path}. Check the file path and integrity.")

    # Reduce bit depth to 4-bit to improve compression efficiency
    image = (image // 16) * 16

        # Normalize difference values to avoid large variations
    diff_image = compute_difference_image(image)
    diff_string = ''.join([chr((val + 256) % 256) for val in diff_image.flatten()])

    lzw = LZWCompression()
    compressed_data = lzw.compress(diff_string)

    save_compressed_file(output_path, compressed_data)
    print(f"Image compressed and saved to {output_path}")


def decompress_image(compressed_path, output_path, original_shape):
    """Decompress LZW compressed image and reconstruct the original image"""
    if not os.path.exists(compressed_path):
        raise FileNotFoundError(f"Error: Compressed file '{compressed_path}' not found.")

    compressed_data = load_compressed_file(compressed_path)
    lzw = LZWCompression()
    decompressed_string = lzw.decompress(compressed_data)

    # Convert back to integer array with proper normalization
    diff_image = np.array([(ord(c) - 256) % 256 for c in decompressed_string], dtype=np.int16)
    diff_image = diff_image.reshape(original_shape)

    restored_image = restore_original_image(diff_image)
    cv2.imwrite(output_path, restored_image)
    print(f"Decompressed image saved to {output_path}")


if __name__ == "__main__":
    input_image = "sample_image.png"
    compressed_file = "compressed.lzw"
    decompressed_image = "restored.png"

    compress_image(input_image, compressed_file)
    decompress_image(compressed_file, decompressed_image, (256, 256))

    # Display file size statistics
    original_size = get_file_size(input_image)
    compressed_size = get_file_size(compressed_file)
    decompressed_size = get_file_size(decompressed_image)

    print("\nCompression Results:")
    print(f"Original Image Size: {original_size} bytes")
    print(f"Compressed File Size: {compressed_size} bytes")
    print(f"Decompressed Image Size: {decompressed_size} bytes")
    print(f"Compression Ratio: {compressed_size / original_size:.4f}")
    print(f"Space Savings: {100 * (1 - compressed_size / original_size):.2f}%")
