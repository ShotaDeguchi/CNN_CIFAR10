"""
test to check PyTorch is working with MPS (Metal Performance Shader)
"""

import torch

def main():
    print(f"PyTorch version: {torch.__version__}")

    # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
    print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")

    # Set the device      
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create data and send it to the device
    x = torch.rand(size=(3, 4)).to(device)
    print(x)



if __name__ == "__main__":
    main()

