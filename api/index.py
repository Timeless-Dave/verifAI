"""Vercel entry point. Vercel runs this file as one Python function.

The package lives in src/, so it is put on the import path here. Settings
come from the project's environment variables (see docs/deploy.md).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verifai.main import app

__all__ = ["app"]
