# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec-fil för PlankInspection Windows-tjänst.

Kör med:
    pyinstaller build.spec --noconfirm

VIKTIGT: Bygg EFTER att CPU-only torch installerats (se build.bat).
"""

import os
import sys
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_dynamic_libs,
)

block_cipher = None

# Bas-mapp (där build.spec ligger)
BASE_DIR = os.getcwd()

# Samla ultralytics data-filer (YOLO behöver config-filer)
ultralytics_datas = collect_data_files('ultralytics')

# Samla torch DLL-filer korrekt (löser shm.dll-problemet)
torch_binaries = collect_dynamic_libs('torch')

# Samla även explicit från torch/lib (backup om collect_dynamic_libs missar dem)
import torch as _torch
_torch_lib_dir = os.path.join(os.path.dirname(_torch.__file__), 'lib')
if os.path.isdir(_torch_lib_dir):
    for dll in os.listdir(_torch_lib_dir):
        if dll.lower().endswith('.dll'):
            full = os.path.join(_torch_lib_dir, dll)
            # Lägg bara till om den inte redan finns
            if not any(b[1] == full for b in torch_binaries):
                torch_binaries.append((full, os.path.join('torch', 'lib')))

a = Analysis(
    ['service.py'],
    pathex=[BASE_DIR],
    binaries=torch_binaries,
    datas=[
        # Webbfiler
        (os.path.join(BASE_DIR, 'templates'), 'templates'),
        (os.path.join(BASE_DIR, 'static'), 'static'),

        # YOLO-modeller
        (os.path.join(BASE_DIR, '..', 'best.pt'), '.'),
        (os.path.join(BASE_DIR, '..', 'best_cracks.pt'), '.'),
        (os.path.join(BASE_DIR, '..', 'Best_Corners.pt'), '.'),

        # Config
        (os.path.join(BASE_DIR, 'presets.json'), '.'),
        (os.path.join(BASE_DIR, 'settings.ini'), '.'),
    ] + ultralytics_datas,
    hiddenimports=[
        # Windows service
        'win32timezone',

        # Uvicorn / FastAPI
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',

        # Era moduler
        'main',
        'pipeline_web',
        'config',

        # OpenCV
        'cv2',
        'opencv_python',

        # Torch
        'torch',
        'torch.utils',
    ] + collect_submodules('ultralytics'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(BASE_DIR, 'rthook_torch.py')],
    excludes=[
        # Säkra att exkludera — används aldrig av YOLO/CV pipeline
        'tkinter', 'matplotlib', 'scipy',
        'PyQt5', 'PySide2',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ============================================================
# Med CPU-only torch finns inga CUDA-DLL:er att filtrera.
# Om du av misstag bygger med CUDA-torch, filtrera bort dem:
# ============================================================
CUDA_DLLS = (
    'cublas', 'cudnn', 'cufft', 'curand', 'cusolver',
    'cusparse', 'nccl', 'nvrtc', 'cudart', 'cupti',
    'nvJitLink', 'nvToolsExt', 'torch_cuda', 'c10_cuda',
)
a.binaries = [
    b for b in a.binaries
    if not any(cuda in b[0].lower() for cuda in CUDA_DLLS)
]

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='service',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PlankInspection',
)
