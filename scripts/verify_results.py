#!/usr/bin/env python3
"""Convenience wrapper for aggregate-result verification."""

import sys

from molprop_fusion.cli import main

if __name__ == "__main__":
    main(["verify-results", *sys.argv[1:]])
