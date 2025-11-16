# Package initialization
# This allows absolute imports to work when running scripts from this directory

import sys
import os

# Add current directory to Python path if not already there
# This enables absolute imports like: from model import SignSTGCNModel
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
