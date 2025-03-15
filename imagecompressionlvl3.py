import numpy as np
import cv2
import pickle
import os

def get_file_size(file_path):
    """Returns file size in bytes."""
    return os.path.getsize(file_path) if os.path.exists(file_path) else 0

class LZWCompression:
    def __init__(self):
        # For reference; not heavily used in this design approach
        self.dictionary_size = 256
        self.dictionary = {chr(i): i for i in range(self.dictionary_size)}
        self.reverse_dictionary = {i: chr(i) for i in range(self.dictionary_size)}

    def encode(self, data):
        """
        Encode a bytes-like object using LZW, returning a list of integer codes.
        """
        dict_size = 256
        dictionary = {bytes([i]): i for i in range(dict_size)}
        w = b""
        result = []

        for byte in data:
            wk = w + bytes([byte])
            if wk in dictionary:
                w = wk
            else:
                result.append(dictionary[w])
                dictionary[wk] = dict_size
                dict_size += 1
                w = bytes([byte])

        # Output the code for the last w
        if w:
            result.append(dictionary[w])

        return result

    def decode(self, compressed):
        """
        Decode a list of integer codes (from LZW) back to a bytes object.
        """
        dict_size = 256
        dictionary = {i: bytes([i]) for i in range(dict_size)}
        result = []

        # First code
        w = dictionary[compressed[0]]
        result.append(w)
        compressed = compressed[1:]  # consume the first code

        for k in compressed:
            if k in dictionary:
                entry = dictionary[k]
            elif k == dict_size:
                # Special case: "w + w[0]" for the 'KWK' scenario
                entry = w + w[:1]
            else:
                raise ValueError(f"Bad compressed k: {k}")

            result.append(entry)
            # Add w + entry's first byte to the dictionary
            dictionary[dict_size] = w + entry[:1]
            dict_size += 1
            w = entry

        return b"".join(result)

def compute_difference_image(image):
    """
    Compute the difference-encoded image:
      - For col > 0: horizontal difference from the left pixel
      - For row > 0 and col=0: vertical difference from the above pixel
      - diff[0,0] = image[0,0]
    """
    rows, cols = image.shape
    diff_image = np.zeros((rows, cols), dtype=np.int16)

    # First pixel
    diff_image[0, 0] = image[0, 0]

    # First row: horizontal differences
    for j in range(1, cols):
        diff_image[0, j] = image[0, j] - image[0, j - 1]

    # Remaining rows
    for i in range(1, rows):
        # First column: vertical difference
        diff_image[i, 0] = image[i, 0] - image[i - 1, 0]
        # Remaining columns: horizontal difference
        for j in range(1, cols):
            diff_image[i, j] = image[i, j] - image[i, j - 1]

    return diff_image

def restore_original_image(diff_image):
    """
    Reverse the difference-encoding:
      - rec[0,0] = diff[0,0]
      - rec[0,j] = rec[0,j-1] + diff[0,j]
      - rec[i,0] = rec[i-1,0] + diff[i,0]
      - rec[i,j] = rec[i,j-1] + diff[i,j]
    """
    diff = diff_image.astype(np.int16)
    rows, cols = diff.shape
    rec = np.zeros((rows, cols), dtype=np.int16)

    # First pixel
    rec[0, 0] = diff[0, 0]

    # First row
    for j in range(1, cols):
        rec[0, j] = rec[0, j - 1] + diff[0, j]

    # Remaining rows
    for i in range(1, rows):
        # First column in each row
        rec[i, 0] = rec[i - 1, 0] + diff[i, 0]
        # Remaining columns
        for j in range(1, cols):
            rec[i, j] = rec[i, j - 1] + diff[i, j]

    # Clip to [0..255] to ensure valid gray pixel range
    return np.clip(rec, 0, 255).astype(np.uint8)

def save_compressed_file(filename, shape, compressed_data):
    """
    Save (image shape, compressed_data) as a pickle file for easy retrieval.
    """
    with open(filename, 'wb') as f:
        pickle.dump((shape, compressed_data), f)

def load_compressed_file(filename):
    """
    Load (image shape, compressed_data) from a pickle file.
    """
    with open(filename, 'rb') as f:
        shape, compressed_data = pickle.load(f)
    return shape, compressed_data

def compress_image(input_image_path, output_path):
    """
    Compress an image using difference encoding + LZW.
    The shape is stored so we can reconstruct the original size on decompression.
    """
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Error: Image file '{input_image_path}' not found.")

    # Read image in grayscale
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Error: Unable to read image at {input_image_path}.")

    # Create a difference-encoded version
    diff_image = compute_difference_image(image)

    # Convert the difference image to raw bytes
    diff_bytes = diff_image.tobytes()

    # Compress the difference bytes using LZW
    lzw = LZWCompression()
    compressed_data = lzw.encode(diff_bytes)

    # Save the shape + compressed data to file
    save_compressed_file(output_path, image.shape, compressed_data)
    print(f"Image compressed and saved to {output_path}.")

def decompress_image(compressed_file_path, output_image_path):
    """
    Decompress an LZW + difference-encoded file and restore the original image.
    """
    if not os.path.exists(compressed_file_path):
        raise FileNotFoundError(f"Error: Compressed file '{compressed_file_path}' not found.")

    # Load shape + compressed data
    shape, compressed_data = load_compressed_file(compressed_file_path)

    # Decompress the difference bytes
    lzw = LZWCompression()
    diff_bytes = lzw.decode(compressed_data)

    # Convert back to a NumPy array
    diff_image = np.frombuffer(diff_bytes, dtype=np.int16).reshape(shape)

    # Reconstruct the original image
    restored_image = restore_original_image(diff_image)

    # Save the final image
    cv2.imwrite(output_image_path, restored_image)
    print(f"Decompressed image saved to {output_image_path}.")

if __name__ == "__main__":
    # Update these paths/names as desired or necessary
    input_image = "sample_image.bmp"
    compressed_file = "compressed.lzw"
    decompressed_image = "sample_image_decompressed.bmp"

    # Compress
    compress_image(input_image, compressed_file)

    # Decompress
    decompress_image(compressed_file, decompressed_image)

    # Display file sizes
    original_size = get_file_size(input_image)
    compressed_size = get_file_size(compressed_file)
    decompressed_size = get_file_size(decompressed_image)

    print("\nCompression Results:")
    print(f"Original Image Size:      {original_size} bytes")
    print(f"Compressed File Size:     {compressed_size} bytes")
    print(f"Decompressed Image Size:  {decompressed_size} bytes")

    if original_size > 0:
        ratio = compressed_size / original_size
        print(f"Compression Ratio:        {ratio:.4f}")
        print(f"Space Savings:            {100 * (1 - ratio):.2f}%")