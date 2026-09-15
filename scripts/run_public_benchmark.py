#!/usr/bin/env python3
"""Convenience wrapper for the benchmark command."""

import sys

from molprop_fusion.cli import main

if __name__ == "__main__":
    main(["benchmark", *sys.argv[1:]])
