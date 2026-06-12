"""Application entrypoint for the current foundation milestone."""

from backend.config import settings


def main() -> None:
    """Print a deterministic startup message for smoke verification."""
    print(f"{settings.app_name} backend foundation ready")


if __name__ == "__main__":
    main()
