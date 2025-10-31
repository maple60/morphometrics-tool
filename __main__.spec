# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules, collect_all, collect_dynamic_libs

datas = []
binaries = []
hiddenimports = []

# --- Add Visual C++ runtime DLLs manually ---
import glob
import sys
import os

# get the path to System32 (64bit only)
system32 = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32")

# find required VC runtime DLLs
vcruntime_dlls = [
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
]

for dll in vcruntime_dlls:
    src = os.path.join(system32, dll)
    if os.path.exists(src):
        binaries.append((src, "."))

# Collect from core dependencies
for pkg in ["PyQt6", "napari", "napari_builtins", "vispy", "magicgui", "imageio", "PIL", "tifffile", "torch", "torchvision"]:
    tmp = collect_all(pkg)
    datas += tmp[0]
    binaries += tmp[1]
    hiddenimports += tmp[2]

# Also include MSVC-dependent dynamic libraries for torch
binaries += collect_dynamic_libs("torch")

# Additional hidden imports for readers
hiddenimports += [
    "imageio.plugins.pillow",
    "imageio.plugins.tifffile",
    "PIL.JpegImagePlugin",
    "PIL.PngImagePlugin",
    "PIL.TiffImagePlugin",
    "PIL.BmpImagePlugin",
    "napari_builtins",  # built-in readers
]

# Add napari plugin modules
hiddenimports += collect_submodules('napari.plugins')
hiddenimports += collect_submodules('napari.plugins.io')
hiddenimports += collect_submodules('napari.plugins._builtins')

# Safety: ensure all imageio plugins are bundled
hiddenimports += collect_submodules('imageio.plugins')

a = Analysis(
    ['src\\leaf_shape_tool\\__main__.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["runtime_hooks\\pyi_rth_torchpath.py"],
    excludes=[
        "OpenGL",
        "torch.distributed",
        "torch.testing",
        "torch.distributed.elastic",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='LeafShapeTool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
)


coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='LeafShapeTool'
)
