"""Click entry point for the independent advection-diffusion architecture."""
from train_convex_kernel import main as shared_main


if __name__ == "__main__":
    import sys

    shared_main.main(args=["--architecture", "advection-diffusion", "--wind-mode", "directional", *sys.argv[1:]])
