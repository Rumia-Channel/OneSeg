"""Console entry point for the OneSeg desktop application."""

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OneSeg SDR / experimental 1seg GUI (continuous live TV unverified)"
    )
    parser.parse_args()
    from .gui import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
