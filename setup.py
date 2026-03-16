"""PyInstaller build configuration for LiftReview."""

import PyInstaller.__main__
import sys

def build():
    args = [
        'main.py',
        '--name=LiftReview',
        '--onedir',
        '--windowed',
        '--noconfirm',
        '--clean',
        # Include mediapipe model files
        '--collect-data=mediapipe',
        # Hidden imports that PyInstaller may miss
        '--hidden-import=cv2',
        '--hidden-import=numpy',
        '--hidden-import=scipy',
        '--hidden-import=sounddevice',
        '--hidden-import=pyqtgraph',
        '--hidden-import=mediapipe',
        # Add the app package
        '--add-data=app:app',
    ]

    PyInstaller.__main__.run(args)


if __name__ == '__main__':
    build()
