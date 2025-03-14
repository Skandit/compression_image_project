import numpy as np
import cv2
import os

class LZWImageCompression:
    def __init__(self, image_path):
        """
        Initializes the LZW compression object with the image path.
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)  # Read color image
        self.channels = cv2.split(self.image)  # Split into R, G, B channels

    def lzw_compress(self, data):
        """
        Applies LZW compression to a given data array.
        """
        dictionary = {bytes([i]): i for i in range(256)}
        dict_size = 256
        w = b''
        compressed_data = []
        
        for k in data:
            wk = w + bytes([k])
            if wk in dictionary:
                w = wk
            else:
                compressed_data.append(dictionary[w])
                dictionary[wk] = dict_size
                dict_size += 1
                w = bytes([k])
        
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
        
        return b''.join(decompressed_data)

    def compress_image(self, output_path):
        """
        Compresses an RGB image using LZW on each color channel separately.
        """
        compressed_channels = []
        for channel in self.channels:
            flat_channel = channel.flatten()
            compressed_data = self.lzw_compress(flat_channel)
            compressed_channels.append(compressed_data)
        
        np.savez_compressed(output_path, r=compressed_channels[0], g=compressed_channels[1], b=compressed_channels[2])
        print(f"Image successfully compressed and saved to {output_path}")

    def decompress_image(self, compressed_path, output_path):
        """
        Decompresses an LZW compressed image and reconstructs the original color image.
        """
        data = np.load(compressed_path)
        decompressed_channels = []
        
        for key in ['r', 'g', 'b']:
            compressed_data = data[key].tolist()
            decompressed_data = self.lzw_decompress(compressed_data)
            decompressed_channels.append(np.frombuffer(decompressed_data, dtype=np.uint8))
        
        # Reshape decompressed data back into image shape
        height, width, _ = self.image.shape
        r_channel = decompressed_channels[0].reshape((height, width))
        g_channel = decompressed_channels[1].reshape((height, width))
        b_channel = decompressed_channels[2].reshape((height, width))
        
        decompressed_image = cv2.merge([r_channel, g_channel, b_channel])
        cv2.imwrite(output_path, decompressed_image)
        print(f"Image successfully decompressed and saved to {output_path}")

# Example usage
if __name__ == "__main__":
    image_path = "thumbs_up.bmp"  # Change this to your input image
    compressed_output = "compressed_image.npz"
    decompressed_output = "thumbs_updecompressedforlevel4.bmp"
    
    compressor = LZWImageCompression(image_path)
    compressor.compress_image(compressed_output)
    compressor.decompress_image(compressed_output, decompressed_output)
