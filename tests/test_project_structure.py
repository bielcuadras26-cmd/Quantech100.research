from pathlib import Path


def test_required_foundation_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    required_paths = [
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "docs/architecture.md",
        "docs/roadmap.md",
        "docs/research_framework.md",
        "backend/main.py",
        "backend/config.py",
    ]

    missing = [path for path in required_paths if not (root / path).exists()]

    assert missing == []
