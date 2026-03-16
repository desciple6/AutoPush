"""LiftReview - Lifting Footage Review Software

Entry point for the application.
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from app.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("LiftReview")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
