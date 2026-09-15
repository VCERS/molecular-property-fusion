#!/usr/bin/env python3
"""Convenience wrapper for the AqSolDB transfer command."""

import sys

from molprop_fusion.cli import main

if __name__ == "__main__":
    main(["transfer", *sys.argv[1:]])
