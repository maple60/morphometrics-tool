# pyi_rth_torchpath.py
import os
import sys


def _add(path: str):
    try:
        if os.path.isdir(path):
            os.add_dll_directory(path)
    except Exception:
        pass


# Onefile 時は _MEIPASS、Onefolder 時は exe 直下
base = (
    getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    if getattr(sys, "frozen", False)
    else None
)

if base:
    # torch の DLL 置き場
    _add(os.path.join(base, "torch", "lib"))
    # Conda/科学計算系が入ることがあるパス（NumPy/SciPy/OpenMP 等）
    _add(os.path.join(base, "Library", "bin"))
    # 念のため exe 直下も
    _add(base)
