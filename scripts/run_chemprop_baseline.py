#!/usr/bin/env python3
"""Convenience wrapper for the Chemprop command."""

import sys

from molprop_fusion.cli import main

if __name__ == "__main__":
    main(["chemprop", *sys.argv[1:]])
