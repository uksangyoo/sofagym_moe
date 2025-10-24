import numpy as np
import sys

def view_npz(file_path):
    data = np.load(file_path)
    
    print(f"Arrays in {file_path}:")
    for name in data.files:
        array = data[name]
        print(f"\n{name}:")
        print(f"  Shape: {array.shape}")
        print(f"  Data: {array}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python view_npz.py <file.npz>")
    else:
        view_npz(sys.argv[1])
