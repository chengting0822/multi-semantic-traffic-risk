#!/usr/bin/env python3
"""Compatibility launcher for the compact temporal annotation GUI."""

from traffic_risk.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["annotate", *__import__("sys").argv[1:]]))
