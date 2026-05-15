import sys

from MixerMainWindow import MixerMainWindow
from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)
window = MixerMainWindow()
window.show()
sys.exit(app.exec_())
