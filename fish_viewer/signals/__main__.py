"""Dispatch signal-center inference to the configured PyTorch worker."""
import sys
from ..spots.runtime import dispatch

def main():
    dispatch(sys.argv[1:])

if __name__ == "__main__":
    main()
