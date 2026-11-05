import torch

def main():
    print("=" * 50)
    print("LiDAR AI/ML Project - GPU System Check")
    print("=" * 50)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

        # Run a small computation on the GPU
        a = torch.randn(1000, 1000, device="cuda")
        b = torch.randn(1000, 1000, device="cuda")
        result = torch.matmul(a, b)

        print(f"Computation device: {result.device}")
        print(f"Result shape: {result.shape}")
        print("GPU computation successful!")
    else:
        print("GPU unavailable. Check the PyTorch installation.")

    print("=" * 50)

if __name__ == "__main__":
    main()