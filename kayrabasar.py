import os
import math
import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
from PIL import Image, ImageTk

#############################
# LZW Core Compression Class
#############################
class LZWCoding:
    def __init__(self, data, codelength=9):
        """
        data: can be a text string or a byte array (for images)
        codelength: initial number of bits per code
        """
        self.data = data
        self.codelength = codelength  # starting code length
    
    def encode(self):
        """Encode text or byte data using LZW."""
        dict_size = 256
        if isinstance(self.data, str):
            dictionary = {chr(i): i for i in range(dict_size)}
            w = ""
        else:
            dictionary = {bytes([i]): i for i in range(dict_size)}
            w = b""
        result = []

        for symbol in (self.data if isinstance(self.data, str) else list(self.data)):
            wk = w + symbol if isinstance(self.data, str) else w + bytes([symbol])
            if wk in dictionary:
                w = wk
            else:
                result.append(dictionary[w])
                dictionary[wk] = dict_size
                dict_size += 1
                w = symbol if isinstance(self.data, str) else bytes([symbol])
        if w:
            result.append(dictionary[w])

        self.codelength = math.ceil(math.log2(dict_size))
        return result

    def decode(self, compressed):
        """Decode a list of integer codes using LZW."""
        dict_size = 256
        if isinstance(self.data, str):
            dictionary = {i: chr(i) for i in range(dict_size)}
        else:
            dictionary = {i: bytes([i]) for i in range(dict_size)}
        result = []

        w = dictionary[compressed.pop(0)]
        result.append(w)
        for k in compressed:
            if k in dictionary:
                entry = dictionary[k]
            elif k == dict_size:
                entry = w + (w[0:1] if isinstance(w, bytes) else w[0])
            else:
                raise ValueError('Bad compressed k: %s' % k)
            result.append(entry)
            dictionary[dict_size] = w + (entry[0:1] if isinstance(entry, bytes) else entry[0])
            dict_size += 1
            w = entry

        if isinstance(result[0], bytes):
            return b"".join(result)
        else:
            return "".join(result)

    def int_list_to_binary_string(self, int_list):
        bitstring = ''
        for num in int_list:
            for n in range(self.codelength):
                bitstring += '1' if (num & (1 << (self.codelength - 1 - n))) else '0'
        return bitstring

    def add_code_length_info(self, bitstring):
        codelength_info = '{0:08b}'.format(self.codelength)
        return codelength_info + bitstring

    def pad_encoded_data(self, encoded_data):
        extra_bits = (8 - len(encoded_data) % 8) % 8
        encoded_data += '0' * extra_bits
        padding_info = '{0:08b}'.format(extra_bits)
        return padding_info + encoded_data

    def get_byte_array(self, padded_encoded_data):
        if len(padded_encoded_data) % 8 != 0:
            raise ValueError("Encoded data not padded to a multiple of 8 bits.")
        b = bytearray()
        for i in range(0, len(padded_encoded_data), 8):
            byte_str = padded_encoded_data[i:i+8]
            b.append(int(byte_str, 2))
        return b

    def remove_padding(self, padded_encoded_data):
        padding_info = padded_encoded_data[:8]
        extra_padding = int(padding_info, 2)
        encoded_data = padded_encoded_data[8:]
        if extra_padding != 0:
            encoded_data = encoded_data[:-extra_padding]
        return encoded_data

    def binary_string_to_int_list(self, bitstring):
        int_codes = []
        for i in range(0, len(bitstring), self.codelength):
            slice_bits = bitstring[i:i+self.codelength]
            int_codes.append(int(slice_bits, 2))
        return int_codes

    def extract_code_length_info(self, bitstring):
        codelength_info = bitstring[:8]
        self.codelength = int(codelength_info, 2)
        return bitstring[8:]

#############################
# Image Processing Helpers
#############################
def compute_difference_image(gray_arr):
    arr = gray_arr.astype(np.int16)
    rows, cols = arr.shape
    diff = np.zeros((rows, cols), dtype=np.int16)

    print(gray_arr)
    
    for i in range(rows):
        diff[i, 0] = arr[i, 0]  # Leave the first pixel unchanged.
        for j in range(1, cols):
            diff[i, j] = arr[i, j] - arr[i, j-1]

    print(diff)
    
    return diff

def reconstruct_from_difference(diff_arr):
    diff = diff_arr.astype(np.int16)
    rows, cols = diff.shape
    rec = np.zeros((rows, cols), dtype=np.int16)

    for i in range(rows):
        rec[i, 0] = diff[i, 0]
        for j in range(1, cols):
            rec[i, j] = diff[i, j] + rec[i, j-1]

    rec = np.clip(rec, 0, 255).astype(np.uint8)
    return rec

#############################
# Unified Compression Class
#############################
class LZWProject:
    def __init__(self, filepath, data_type, method="raw"):
        self.filepath = filepath
        self.data_type = data_type
        self.method = method
        self.filename, self.ext = os.path.splitext(os.path.basename(filepath))

        if data_type == "text":
            with open(filepath, 'r') as f:
                self.text = f.read()
        elif data_type == "image":
            self.image = Image.open(filepath)
        else:
            raise ValueError("Unsupported data type.")

    ##################
    # Helper: build stats
    ##################
    def build_stats(self, uncompressed_size, code_length, compressed_size):
        """
        Return (uncompressed_size, code_length, compressed_size, ratio).
        """
        ratio = float('inf')
        if compressed_size != 0:
            ratio = uncompressed_size / compressed_size
        return (uncompressed_size, code_length, compressed_size, ratio)

    ##################
    # Level 1: Text
    ##################
    def process_text(self):
        # Uncompressed size = number of characters
        uncompressed_size = len(self.text)
        
        # Encode
        lzw = LZWCoding(self.text)
        codes = lzw.encode()
        code_length = lzw.codelength  # bits

        # Convert to bit string
        bitstr = lzw.int_list_to_binary_string(codes)
        bitstr = lzw.add_code_length_info(bitstr)
        padded = lzw.pad_encoded_data(bitstr)
        byte_array = lzw.get_byte_array(padded)

        compressed_size = len(byte_array)  # bytes
        stats = self.build_stats(uncompressed_size, code_length, compressed_size)

        comp_file = self.filename + "_compressed.bin"
        with open(comp_file, 'wb') as f:
            f.write(byte_array)
        print(f"Text compressed to {comp_file}")

        # Decompress
        with open(comp_file, 'rb') as f:
            file_bytes = f.read()
        bit_string = ''.join(bin(byte)[2:].rjust(8, '0') for byte in file_bytes)
        
        lzw_decomp = LZWCoding("")
        bit_string = lzw_decomp.remove_padding(bit_string)
        bit_string = lzw_decomp.extract_code_length_info(bit_string)
        codes = lzw_decomp.binary_string_to_int_list(bit_string)
        decompressed = lzw_decomp.decode(codes)

        decomp_file = self.filename + "_decompressed.txt"
        with open(decomp_file, 'w') as f:
            f.write(decompressed)
        print(f"Text decompressed to {decomp_file}")

        return comp_file, decomp_file, stats

    ##################
    # Level 2: Grayscale RAW
    ##################
    def process_image_gray(self):
        gray = self.image.convert("L")
        gray_arr = np.array(gray, dtype=np.uint8)
        uncompressed_size = gray_arr.size  # number of pixels (1 byte each)

        gray_bytes = gray_arr.tobytes()
        lzw = LZWCoding(gray_bytes)
        codes = lzw.encode()
        code_length = lzw.codelength

        bitstr = lzw.int_list_to_binary_string(codes)
        bitstr = lzw.add_code_length_info(bitstr)
        padded = lzw.pad_encoded_data(bitstr)
        byte_array = lzw.get_byte_array(padded)

        compressed_size = len(byte_array)
        stats = self.build_stats(uncompressed_size, code_length, compressed_size)

        comp_file = self.filename + "_gray_compressed.bin"
        with open(comp_file, 'wb') as f:
            f.write(byte_array)
        print(f"Grayscale image compressed to {comp_file}")

        # Decompress
        with open(comp_file, 'rb') as f:
            file_bytes = f.read()
        bit_string = ''.join(bin(byte)[2:].rjust(8, '0') for byte in file_bytes)

        lzw_decomp = LZWCoding(b"")
        bit_string = lzw_decomp.remove_padding(bit_string)
        bit_string = lzw_decomp.extract_code_length_info(bit_string)
        codes = lzw_decomp.binary_string_to_int_list(bit_string)
        decompressed_bytes = lzw_decomp.decode(codes)

        arr = np.frombuffer(decompressed_bytes, dtype=np.uint8).reshape(gray_arr.shape)
        image = Image.fromarray(arr)

        decomp_file = self.filename + "_gray_decompressed.bmp"
        image.save(decomp_file)
        print(f"Grayscale image decompressed to {decomp_file}")
        return comp_file, decomp_file, stats

    ##################
    # Level 3: Grayscale Diff
    ##################
    def process_image_gray_diff(self):
        gray = self.image.convert("L")
        gray_arr = np.array(gray, dtype=np.uint8)
        uncompressed_size = gray_arr.size  # each pixel is 1 byte

        diff_arr = compute_difference_image(gray_arr)
        diff_bytes = diff_arr.tobytes()

        lzw = LZWCoding(diff_bytes)
        codes = lzw.encode()
        code_length = lzw.codelength

        bitstr = lzw.int_list_to_binary_string(codes)
        bitstr = lzw.add_code_length_info(bitstr)
        padded = lzw.pad_encoded_data(bitstr)
        byte_array = lzw.get_byte_array(padded)

        compressed_size = len(byte_array)
        stats = self.build_stats(uncompressed_size, code_length, compressed_size)

        comp_file = self.filename + "_gray_diff_compressed.bin"
        with open(comp_file, 'wb') as f:
            f.write(byte_array)
        print(f"Difference image compressed to {comp_file}")

        # Decompress
        with open(comp_file, 'rb') as f:
            file_bytes = f.read()
        bit_string = ''.join(bin(byte)[2:].rjust(8, '0') for byte in file_bytes)

        lzw_decomp = LZWCoding(b"")
        bit_string = lzw_decomp.remove_padding(bit_string)
        bit_string = lzw_decomp.extract_code_length_info(bit_string)
        codes = lzw_decomp.binary_string_to_int_list(bit_string)
        diff_dec = lzw_decomp.decode(codes)

        diff_arr2 = np.frombuffer(diff_dec, dtype=np.int16).reshape(gray_arr.shape)
        rec_arr = reconstruct_from_difference(diff_arr2)
        image = Image.fromarray(rec_arr)

        decomp_file = self.filename + "_gray_diff_decompressed.bmp"
        image.save(decomp_file)
        print(f"Difference image decompressed to {decomp_file}")
        return comp_file, decomp_file, stats

    ##################
    # Level 4: Color (Per Channel)
    ##################
    def process_color_image(self):
        image_rgb = self.image.convert("RGB")
        r, g, b = image_rgb.split()

        channels = {'R': np.array(r, dtype=np.uint8),
                    'G': np.array(g, dtype=np.uint8),
                    'B': np.array(b, dtype=np.uint8)}

        # For stats
        total_uncompressed = 0
        total_compressed = 0
        code_lengths = []

        comp_files = {}
        decomp_channels = {}

        for channel_name, arr in channels.items():
            data = arr.tobytes()
            # Uncompressed size is arr.size bytes for this channel
            total_uncompressed += arr.size

            lzw = LZWCoding(data)
            codes = lzw.encode()
            code_lengths.append(lzw.codelength)

            bitstr = lzw.int_list_to_binary_string(codes)
            bitstr = lzw.add_code_length_info(bitstr)
            padded = lzw.pad_encoded_data(bitstr)
            byte_array = lzw.get_byte_array(padded)

            # Accumulate compressed size
            total_compressed += len(byte_array)

            comp_file = f"{self.filename}_{channel_name}_compressed.bin"
            with open(comp_file, 'wb') as f:
                f.write(byte_array)
            comp_files[channel_name] = comp_file
            print(f"Channel {channel_name} compressed to {comp_file}")

            # Decompress
            with open(comp_file, 'rb') as f:
                file_bytes = f.read()
            bit_string = ''.join(bin(byte)[2:].rjust(8, '0') for byte in file_bytes)

            lzw_decomp = LZWCoding(b"")
            bit_string = lzw_decomp.remove_padding(bit_string)
            bit_string = lzw_decomp.extract_code_length_info(bit_string)
            codes = lzw_decomp.binary_string_to_int_list(bit_string)
            decompressed_data = lzw_decomp.decode(codes)

            decomp_arr = np.frombuffer(decompressed_data, dtype=np.uint8).reshape(arr.shape)
            decomp_channels[channel_name] = Image.fromarray(decomp_arr)

        # Build overall stats
        avg_code_length = round(sum(code_lengths) / len(code_lengths), 2)
        stats = self.build_stats(total_uncompressed, avg_code_length, total_compressed)

        # Merge channels
        image_merged = Image.merge("RGB", (
            decomp_channels['R'],
            decomp_channels['G'],
            decomp_channels['B']
        ))
        decomp_file = self.filename + "_color_decompressed.bmp"
        image_merged.save(decomp_file)
        print(f"Color image decompressed to {decomp_file}")

        return comp_files, decomp_file, stats

    ##################
    # Level 5: Color Diff (Per Channel)
    ##################
    def process_color_image_diff(self):
        image_rgb = self.image.convert("RGB")
        r, g, b = image_rgb.split()

        channels = {'R': np.array(r, dtype=np.uint8),
                    'G': np.array(g, dtype=np.uint8),
                    'B': np.array(b, dtype=np.uint8)}

        # For stats
        total_uncompressed = 0
        total_compressed = 0
        code_lengths = {}

        comp_files = {}
        decomp_channels = {}

        for channel_name, arr in channels.items():
            total_uncompressed += arr.size

            # Compute difference on this channel
            diff_arr = compute_difference_image(arr)
            diff_bytes = diff_arr.tobytes()

            lzw = LZWCoding(diff_bytes)
            codes = lzw.encode()
            code_lengths[channel_name] = lzw.codelength

            bitstr = lzw.int_list_to_binary_string(codes)
            bitstr = lzw.add_code_length_info(bitstr)
            padded = lzw.pad_encoded_data(bitstr)
            byte_array = lzw.get_byte_array(padded)
            total_compressed += len(byte_array)

            comp_file = f"{self.filename}_{channel_name}_diff_compressed.bin"
            with open(comp_file, 'wb') as f:
                f.write(byte_array)
            comp_files[channel_name] = comp_file
            print(f"Channel {channel_name} difference compressed to {comp_file}")

            # Decompress
            with open(comp_file, 'rb') as f:
                file_bytes = f.read()
            bit_string = ''.join(bin(byte)[2:].rjust(8, '0') for byte in file_bytes)

            lzw_decomp = LZWCoding(b"")
            bit_string = lzw_decomp.remove_padding(bit_string)
            bit_string = lzw_decomp.extract_code_length_info(bit_string)
            codes = lzw_decomp.binary_string_to_int_list(bit_string)
            diff_dec = lzw_decomp.decode(codes)

            diff_arr2 = np.frombuffer(diff_dec, dtype=np.int16).reshape(arr.shape)
            rec_arr = reconstruct_from_difference(diff_arr2)
            decomp_channels[channel_name] = Image.fromarray(rec_arr)

        # Build overall stats
        # We'll take average code length across channels for display
        avg_code_len = round(sum(code_lengths.values()) / len(code_lengths), 2)
        stats = self.build_stats(total_uncompressed, avg_code_len, total_compressed)

        # Merge channels
        image_merged = Image.merge("RGB", (
            decomp_channels['R'],
            decomp_channels['G'],
            decomp_channels['B']
        ))
        decomp_file = self.filename + "_color_diff_decompressed.bmp"
        image_merged.save(decomp_file)
        print(f"Color image (differences) decompressed to {decomp_file}")

        return comp_files, decomp_file, stats

#############################
# Level 6: GUI using Tkinter
#############################
class LZW_GUI:
    def __init__(self, master):
        self.master = master
        master.title("LZW Compression Project")

        self.label = tk.Label(master, text="Select a file:")
        self.label.pack(pady=5)

        self.file_button = tk.Button(master, text="Browse", command=self.select_file)
        self.file_button.pack(pady=5)

        self.file_type = tk.StringVar(value="image")
        self.rb_text = tk.Radiobutton(master, text="Text", variable=self.file_type, value="text")
        self.rb_image = tk.Radiobutton(master, text="Image", variable=self.file_type, value="image")
        self.rb_text.pack()
        self.rb_image.pack()

        self.method = tk.StringVar(value="raw")
        self.method_label = tk.Label(master, text="Method (for images):")
        self.method_label.pack(pady=5)
        self.method_combo = ttk.Combobox(
            master, 
            textvariable=self.method, 
            values=["raw", "diff", "color", "color_diff"]
        )
        self.method_combo.pack(pady=5)

        # Button for compression & immediate decompression
        self.process_button = tk.Button(master, text="Process File", command=self.process_file)
        self.process_button.pack(pady=5)

        # Labels to show the original and the decompressed images
        self.original_image_label = tk.Label(master, text="Original Image Preview")
        self.original_image_label.pack(pady=10)

        self.decompressed_image_label = tk.Label(master, text="Decompressed Image Preview")
        self.decompressed_image_label.pack(pady=10)

        self.status = tk.Label(master, text="Status: Waiting for file selection")
        self.status.pack(pady=10)

        self.filepath = None

    def select_file(self):
        self.filepath = filedialog.askopenfilename()
        if self.filepath:
            self.status.config(text=f"Selected: {self.filepath}")
            # If file_type is image, display a preview
            if self.file_type.get() == "image":
                self.show_image(self.filepath, self.original_image_label)

    def show_image(self, path, label):
        """Load the image from 'path' and display it in 'label'."""
        try:
            img = Image.open(path)
            # Optionally resize if large
            img.thumbnail((300, 300))  # Keep within 300x300
            tk_img = ImageTk.PhotoImage(img)
            label.config(image=tk_img, text="")
            label.image = tk_img  # Keep a reference to avoid garbage-collection
        except Exception as e:
            label.config(text=f"Cannot open image: {e}", image="")

    def process_file(self):
        if not self.filepath:
            self.status.config(text="No file selected!")
            return
        file_type = self.file_type.get()
        method = self.method.get()
        proj = LZWProject(self.filepath, file_type, method)

        try:
            if file_type == "text":
                comp_file, decomp_file, stats = proj.process_text()
                msg = (f"Text compressed to {comp_file} and decompressed to {decomp_file}.\n"
                       f"{self.format_stats(stats)}")
            elif file_type == "image":
                if method == "raw":
                    comp_file, decomp_file, stats = proj.process_image_gray()
                    msg = (f"Grayscale image compressed to {comp_file} and decompressed to {decomp_file}.\n"
                           f"{self.format_stats(stats)}")
                elif method == "diff":
                    comp_file, decomp_file, stats = proj.process_image_gray_diff()
                    msg = (f"Difference image compressed to {comp_file} and decompressed to {decomp_file}.\n"
                           f"{self.format_stats(stats)}")
                elif method == "color":
                    comp_files, decomp_file, stats = proj.process_color_image()
                    # comp_files is a dict of channel files
                    msg = (f"Color image channels compressed; merged into {decomp_file}.\n"
                           f"{self.format_stats(stats)}")
                elif method == "color_diff":
                    comp_files, decomp_file, stats = proj.process_color_image_diff()
                    msg = (f"Color difference channels compressed; merged into {decomp_file}.\n"
                           f"{self.format_stats(stats)}")
                else:
                    msg = "Invalid method selected!"
                # Show the decompressed image
                if "Invalid" not in msg:
                    self.show_image(decomp_file, self.decompressed_image_label)
            else:
                msg = "Invalid file type selected!"

            self.status.config(text=msg)

        except Exception as e:
            self.status.config(text=f"Error: {e}")
            print("Processing error:", e)
            self.decompressed_image_label.config(text=f"Error decompressing: {e}", image="")

    def format_stats(self, stats):
        """
        stats is a tuple: (uncompressed_size, code_length, compressed_size, ratio)
        Return a nicely formatted string for display.
        """
        uncompressed_size, code_length, compressed_size, ratio = stats
        return (f"Uncompressed Size: {uncompressed_size} bytes\n"
                f"Code Length: {code_length} bits\n"
                f"Compressed Size: {compressed_size} bytes\n"
                f"Compression Ratio: {ratio:.2f}")

if __name__ == "__main__":
    root = tk.Tk()
    gui = LZW_GUI(root)
    root.minsize(width=450, height=600)
    root.mainloop()