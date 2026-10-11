"""OmniRetail AI Suite — Standalone Master Desktop GUI Launcher
(Subfolder Proxy Launcher for Attendence System)
"""

import sys
from pathlib import Path

# Resolve Parent Workspace Root
PARENT_ROOT = Path(__file__).resolve().parent.parent
if str(PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_ROOT))

import unified_app

if __name__ == "__main__":
    unified_app.main()
