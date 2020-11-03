import sys
import AudioResolutionApp
from PyQt5.QtWidgets import QApplication

# import moviepy.editor as mpe
import os


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = AudioResolutionApp.AudioResolutionApp()
    sys.exit(app.exec_())

