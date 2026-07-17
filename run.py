#!/usr/bin/env python3
"""Convenience entry point so you can run ``python run.py`` from the repo root.

Forwards to the same CLI as the installed console script ``humanizer-bench``,
so all its flags work here too (e.g. ``python run.py --list``).
"""

import sys

from humanizer_bench.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
