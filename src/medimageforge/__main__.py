"""Enables `python -m medimageforge ...` by delegating to the CLI."""

from medimageforge.cli import main

raise SystemExit(main())
