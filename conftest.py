"""Project-wide pytest conftest.

Adds MinGW's bin/ directory to the DLL search path so the
optional C++ extension (built with -DNFLOW_BUILD_PYBIND=ON)
can be imported by the cross-language parity tests.
"""

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    _mingw_bin = Path(r"C:\Program Files\JetBrains\CLion 2026.1.2\bin\mingw\bin")
    if _mingw_bin.exists():
        os.add_dll_directory(str(_mingw_bin))
