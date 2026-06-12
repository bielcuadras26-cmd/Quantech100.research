"""Application entrypoint for QuantTech100."""


def main() -> None:
    """Run the local FastAPI application."""
    import uvicorn

    uvicorn.run("backend.api.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
