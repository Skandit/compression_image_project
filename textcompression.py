
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

    # Decompress the file
    decompressed_file = lzw.decompress_text_file()
    print(f"Decompression complete: {decompressed_file}")

    # Compare file sizes
    original_size = os.path.getsize(filename + ".txt")
    compressed_size = os.path.getsize(compressed_file)
    decompressed_size = os.path.getsize(decompressed_file)

    print("\nFile Size Report:")
    print(f"Original file size: {original_size} bytes")
    print(f"Compressed file size: {compressed_size} bytes")
    print(f"Decompressed file size: {decompressed_size} bytes")

    # Check if decompression was successful
    with open(filename + ".txt", "r") as f1, open(decompressed_file, "r") as f2:
        if f1.read() == f2.read():
            print("Decompression successful! File matches original.")
        else:
            print("Decompression failed! Files do not match.")

if __name__ == "__main__":
    main()
