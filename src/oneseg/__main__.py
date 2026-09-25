"""Console entry point for the OneSeg desktop application."""

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OneSeg SDR / 1seg research GUI (1seg video decoding not implemented)"
    )
    parser.parse_args()
    from .gui import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
