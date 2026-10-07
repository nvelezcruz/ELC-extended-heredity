from pathlib import Path

from src.publication_supplement import make_figures


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    make_figures(root, root / "outputs" / "publication_figures")
