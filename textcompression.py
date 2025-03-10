
import os
from lzwproject import LZWCode  # Import your LZW class

def main():
    # Ask user for file name (without extension)
    filename = input("Enter the name of the file (without extension): ")

    # Ensure the file exists
    if not os.path.exists(filename + ".txt"):
        print("Error: File not found!")
        return

    # Create an instance of LZWCoding
    lzw = LZWCode(filename, "text")

    # Compress the file
    compressed_file = lzw.compress_text_file()
    print(f"Compression complete: {compressed_file}")

    # Calculate original and compressed file sizes
    original_size = os.path.getsize(filename + ".txt")
    compressed_size = os.path.getsize(compressed_file)

    # Calculate compression ratio
    compression_ratio = compressed_size / original_size if original_size > 0 else 0
    code_length = len(bin(compressed_size)[2:])  # Approximate code length in bits

    print("\nCompression Statistics:")
    print(f"Original File Size: {original_size} bytes")
    print(f"Compressed File Size: {compressed_size} bytes")
    print(f"Compression Ratio: {compression_ratio:.4f}")
    print(f"Calculated Code Length: {code_length} bits")

    # Decompress the file
    decompressed_file = lzw.decompress_text_file()
    print(f"Decompression complete: {decompressed_file}")

    # Compare file sizes
    decompressed_size = os.path.getsize(decompressed_file)
    print("\nFile Size Report:")
    print(f"Decompressed File Size: {decompressed_size} bytes")

    # Compare original and decompressed files to verify correctness
    with open(filename + ".txt", 'r') as orig_file, open(decompressed_file, 'r') as decomp_file:
        original_content = orig_file.read()
        decompressed_content = decomp_file.read()
        if original_content == decompressed_content:
            print("\nVerification: SUCCESS - The decompressed file matches the original.")
        else:
            print("\nVerification: FAILURE - The decompressed file does NOT match the original.")


if __name__ == "__main__":
    main()
