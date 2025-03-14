import numpy as np
import cv2
import os

class LZWImageCompression:
    def __init__(self, image_path):
        """
        Initializes the LZW compression object with the image path.
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Read color image with unchanged format
        
        if self.image is None:
            raise FileNotFoundError(f"Error: Unable to read image file '{image_path}'. Check file path and format.")
        
        self.channels = cv2.split(self.image)  # Split into R, G, B channels
        self.height, self.width, _ = self.image.shape

    def compute_differences(self, channel):
        """
        Computes the difference image for a given color channel.
        """
        diff_image = np.zeros_like(channel, dtype=np.int16)
        diff_image[:, 1:] = channel[:, 1:] - channel[:, :-1]  # Row-wise differences
        diff_image[1:, 0] = channel[1:, 0] - channel[:-1, 0]  # Column-wise differences
        return diff_image.flatten()

    def restore_from_differences(self, diff_data):
        """
        Restores the original color channel from the difference image.
        """
        diff_image = diff_data.reshape((self.height, self.width))
        restored_image = np.zeros_like(diff_image, dtype=np.uint8)
        restored_image[:, 0] = diff_image[:, 0]  # Restore first column
        for j in range(1, self.width):
            restored_image[:, j] = np.clip(restored_image[:, j - 1] + diff_image[:, j], 0, 255)
        for i in range(1, self.height):
            restored_image[i, 0] = np.clip(restored_image[i - 1, 0] + diff_image[i, 0], 0, 255)
        return restored_image

    def lzw_compress(self, data):
        """
        Applies LZW compression to a given data array.
        """
        dictionary = {bytes([i]): i for i in range(256)}
        dict_size = 256
        w = b''
        compressed_data = []
        
        for k in data:
            wk = w + bytes([k & 0xFF])
            if wk in dictionary:
                w = wk
            else:
                compressed_data.append(dictionary[w])
                dictionary[wk] = dict_size
                dict_size += 1
                w = bytes([k & 0xFF])
        
        if w:
            compressed_data.append(dictionary[w])
        return compressed_data

    def lzw_decompress(self, compressed_data):
        """
        Applies LZW decompression to a given compressed data array.
        """
        dictionary = {i: bytes([i]) for i in range(256)}
        dict_size = 256
        w = bytes([compressed_data.pop(0)])
        decompressed_data = [w]
        
        for k in compressed_data:
            if k in dictionary:
                entry = dictionary[k]
            elif k == dict_size:
                entry = w + w[:1]
            else:
                raise ValueError("Invalid compressed k: %s" % k)
            decompressed_data.append(entry)
            dictionary[dict_size] = w + entry[:1]
            dict_size += 1
            w = entry
        
        return np.frombuffer(b''.join(decompressed_data), dtype=np.int16)

    def compress_image(self, output_path):
        """
        Compresses an RGB BMP image using LZW on difference images.
        """
        compressed_channels = []
        for channel in self.channels:
            diff_image = self.compute_differences(channel)
            compressed_data = self.lzw_compress(diff_image)
            compressed_channels.append(compressed_data)
        
        np.savez_compressed(output_path, r=compressed_channels[0], g=compressed_channels[1], b=compressed_channels[2])
        print(f"BMP Image successfully compressed using difference encoding and saved to {output_path}")

    def decompress_image(self, compressed_path, output_path):
        """
        Decompresses an LZW compressed BMP image using difference restoration.
        """
        data = np.load(compressed_path)
        decompressed_channels = []
        
        for key in ['r', 'g', 'b']:
            compressed_data = data[key].tolist()
            decompressed_data = self.lzw_decompress(compressed_data)
            
            if decompressed_data.size != self.height * self.width:
                print(f"Warning: Decompressed data size {decompressed_data.size} does not match expected size {self.height * self.width}")
                return
            
            decompressed_channels.append(decompressed_data)
        
        # Reshape decompressed data back into image shape
        r_channel = self.restore_from_differences(decompressed_channels[0])
        g_channel = self.restore_from_differences(decompressed_channels[1])
        b_channel = self.restore_from_differences(decompressed_channels[2])
        
        decompressed_image = cv2.merge([r_channel, g_channel, b_channel])
        cv2.imwrite(output_path, decompressed_image, [cv2.IMWRITE_BMP])
        print(f"BMP Image successfully decompressed and restored using difference encoding, saved to {output_path}")

# Example usage
if __name__ == "__main__":
    image_path = "thumbs_up.bmp"  # Change this to your input BMP image
    compressed_output = "compressed_image.npz"
    decompressed_output = "thumbs_updecompressedforlvl5.bmp"
    
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' does not exist. Please check the file path.")
    else:
        compressor = LZWImageCompression(image_path)
        compressor.compress_image(compressed_output)
        compressor.decompress_image(compressed_output, decompressed_output)
