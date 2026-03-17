"""
PyInstaller runtime hook - fixar torch DLL-laddning.

Kör FÖRE alla imports och lägger till torch/lib i DLL-sökvägen
så att shm.dll (och alla andra torch-DLL:er) hittas korrekt.
"""
import os
import sys

if getattr(sys, 'frozen', False):
    base = sys._MEIPASS

    # Alla platser där torch kan lägga DLL-filer
    dll_dirs = [
        os.path.join(base, 'torch', 'lib'),
        os.path.join(base, 'torch', 'bin'),
        base,
    ]

    for d in dll_dirs:
        if os.path.isdir(d):
            # Python 3.8+ : os.add_dll_directory() krävs på Windows
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(d)
            # Lägg även i PATH som fallback
            os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')
