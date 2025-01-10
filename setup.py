import sys
from cx_Freeze import setup, Executable

# Define options for the build
build_exe_options = {
    "packages": ["tkinter", "pyaudio", "numpy", "scipy.io", "sqlite3", "matplotlib", "joblib"],
    "include_files": ["songs.db", "trainedModel.pkl"],
    "excludes": ["unittest", "email", "pdb", "doctest", "distutils", "pytest", "test"],
}

# GUI applications on Windows
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="WAVRecognizer",
    version="1.0",
    description="Fourier Transform Audio Recognizer",
    options={"build_exe": build_exe_options},
    executables=[Executable("WAVRecognizer.py", base=base, target_name="WAVRecognizer.exe", icon="app_icon.ico")]
)
