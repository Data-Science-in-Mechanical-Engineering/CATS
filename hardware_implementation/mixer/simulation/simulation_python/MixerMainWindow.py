# import sys
import logging

import matplotlib as mpl
from generateTestbedDialog import Ui_GenerateTestbedDialog

# If you use the use() function, this must be done before importing matplotlib.pyplot.
mpl.use("Qt5Agg")
import numpy as np
import scipy.optimize
import yaml
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from MixerMainWindowUI import Ui_MainWindow
from PyQt5.QtCore import (
    QRectF,
    Qt,
)
from PyQt5.QtGui import (
    QFont,
    QPainter,
    QPen,
)
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)
from TestbedConfiguration import LinkModel, TestbedConfiguration

# logging format
fmt = "%(asctime)s %(filename)-15.15s:%(lineno)-5d %(levelname)-8s %(message)s"
dfmt = "%H:%M:%S"
logging.basicConfig(format=fmt, datefmt=dfmt)
logger = logging.getLogger("MixerMainWindow")
logger.setLevel(logging.INFO)


# This canvas object is also a QWidget and so can be embedded straight into an application as any other Qt widget.
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)


class LinkModelWindow(QWidget):
    def __init__(self, testbed):
        super().__init__(parent=None)

        self.setWindowTitle("Link Model")
        # self.tc = testbedConfiguration

        self.plot = MplCanvas(width=10, height=10, dpi=100)
        x = np.arange(1, (np.max(testbed.nodes) + 1) * np.sqrt(2) * 1.1, dtype=int)
        r = testbed.linkmodel.Ptx_dBm + 10 * np.log10(testbed.linkmodel.d2g(x))
        n = testbed.linkmodel.Pnoise_dBm

        ax = self.plot.fig.add_subplot(221)
        ax.plot(x, r)
        ax.plot(x, [n] * len(x), label="Noise floor")
        ax.set_xlim(x[0], x[-1])
        # ax.set_title('Rx power vs. distance')
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Rx power [dBm]")
        ax.legend(frameon=False)

        ax = self.plot.fig.add_subplot(222)
        ax.semilogx(x, r)
        ax.semilogx(x, [n] * len(x), label="Noise floor")
        ax.set_xlim(x[0], x[-1])
        # ax.set_title('Rx power vs. distance')
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Rx power [dBm]")
        ax.legend(frameon=False)

        sinr = np.arange(-10, 21, dtype=int)
        ax = self.plot.fig.add_subplot(223)
        ax.plot(sinr, testbed.linkmodel.SINR2p(sinr))
        ax.set_ylim(0, 1)
        # ax.set_title('Rx probability vs. SINR')
        ax.set_xlabel("SINR [dB]")
        ax.set_ylabel("Rx probability")

        ax = self.plot.fig.add_subplot(224)
        ax.plot(x, testbed.linkmodel.SINR2p(r - n))
        ax.set_ylim(0, 1)
        # ax.set_title('Rx probability vs. distance w/o interference')
        ax.set_xlabel("distance [m]")
        ax.set_ylabel("Rx probability")

        self.toolbar = NavigationToolbar(self.plot, parent=None)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.plot)
        self.setLayout(self.layout)

        # deletes widget on close event
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.show()

    # Alternative to seperate class:

    # dialog = QDialog(parent=None)
    # dialog.ui = Ui_GenerateTestbedDialog()
    # dialog.ui.setupUi(dialog)
    # dialog.setAttribute(Qt.WA_DeleteOnClose)
    # ret = dialog.exec()


class GenerateTestbedDialog(QDialog, Ui_GenerateTestbedDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)
        self.minPRRLe.editingFinished.connect(self.lineEditFinished)
        self.noisefloorLe.editingFinished.connect(self.lineEditFinished)
        self.txpowerLe.editingFinished.connect(self.lineEditFinished)
        self.wavelengthLe.editingFinished.connect(self.lineEditFinished)
        self.fsplExpLe.editingFinished.connect(self.lineEditFinished)
        self.config = None
        self.lineEditFinished()  # init "resulting properties" fields

    # Update configs before exiting.
    def accept(self):
        self.lineEditFinished()
        QDialog.accept(self)

    def lineEditFinished(self):
        # logger.info(self.sender())

        self.configTestbed = {
            "areaW": int(self.areaWidthLe.text()),
            "areaH": int(self.areaHeightLe.text()),
            "numNodes": int(self.numNodesLe.text()),
            "minDist": float(self.minDistLe.text()),
            "minPRR": float(self.minPRRLe.text()),
        }
        self.configLinkmodel = {
            "noise": float(self.noisefloorLe.text()),
            "txpwr": int(self.txpowerLe.text()),
            "wavelen": float(self.wavelengthLe.text()),
            "attenuation": int(self.fsplExpLe.text()),
        }

        lm = LinkModel(self.configLinkmodel)

        dmax_SNR = scipy.optimize.brentq(
            lambda s: lm.SINR2p(s) - self.configTestbed["minPRR"], -20, 50
        )
        linkmin_dB = (
            self.configLinkmodel["noise"] + dmax_SNR - self.configLinkmodel["txpwr"]
        )
        dmax = lm.g2d(10 ** (linkmin_dB / 10))
        self.configTestbed["maxDist"] = dmax

        self.snrLbl.setText(str(round(dmax_SNR, 3)))
        self.rxsenLbl.setText(str(round(linkmin_dB, 3)))
        self.maxDistLbl.setText(str(round(dmax, 3)))


class MixerMainWindow(QMainWindow, Ui_MainWindow):
    # This defines a signal called 'linkToAdd' that takes two
    # integer arguments.
    # linkToAdd = pyqtSignal(int, int, name='linkToAdd')

    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)

        # Connect self defined signals
        # self.ui.pushButton_6.clicked.connect(lambda: self.addNode(int(self.ui.lineEdit.text())))
        self.testbedGenerateBtn.clicked.connect(self.testbedGenerateBtnClicked)
        self.testbedSaveBtn.clicked.connect(self.testbedSaveBtnClicked)
        self.testbedLoadBtn.clicked.connect(self.testbedLoadBtnClicked)
        self.linkModelBtn.clicked.connect(self.linkModelBtnClicked)

        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(Qt.white)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setDragMode(QGraphicsView.ScrollHandDrag)
        self.graphicsView.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        self.testbed = None

    def testbedSaveBtnClicked(self):
        fileName = QFileDialog.getSaveFileName(self, "Save Testbed")[0]
        if not fileName:
            return
        else:
            with open(fileName, "w") as f:
                f.write(yaml.dump(self.testbed))

    def testbedLoadBtnClicked(self):
        fileName = QFileDialog.getOpenFileName(self, "Open Testbed")[0]
        if not fileName:
            return
        else:
            with open(fileName, "r") as f:
                self.testbed = yaml.load(f.read(), Loader=yaml.Loader)
                self.updateTestbed()

    def testbedGenerateBtnClicked(self):
        dialog = GenerateTestbedDialog()
        if dialog.exec() == QDialog.Accepted:
            self.testbed = TestbedConfiguration(
                dialog.configTestbed, dialog.configLinkmodel
            )
            self.updateTestbed()

    def updateTestbed(self):
        # Cleanup old testbed.
        for item in self.scene.items():
            self.scene.removeItem(item)

        # Add nodes to scene.
        for id, pos in enumerate(self.testbed.nodes, 1):
            node = Node(id)
            # logger.info(f'Node {id} at x={pos[0]} and y={pos[1]}')
            node.setPos(pos[0], pos[1])
            self.scene.addItem(node)

        # Resize the view to show all scene items.
        self.graphicsView.setSceneRect(
            self.scene.itemsBoundingRect().adjusted(-20, -20, 0, 0)
        )
        self.graphicsView.fitInView(self.graphicsView.sceneRect())

        # Update stats.
        stats = (
            f"Area: {self.testbed.areaW}x{self.testbed.areaH}\n"
            f"Nodes: {self.testbed.numNodes}\n"
            f"minPRR: {self.testbed.minPRR}\n"
            f"minDist: {round(self.testbed.minDist, 3)}\n"
            f"maxDist: {round(self.testbed.maxDist, 3)}\n"
        )
        self.statsLbl.setText(stats)

    def linkModelBtnClicked(self):
        # If we don't save the object into a variable, the window closes immediately (I guess garbage collected).
        self.linkModelWin = LinkModelWindow(self.testbed)


class Node(QGraphicsItem):
    def __init__(self, id):
        super(Node, self).__init__()
        # self.setCursor(Qt.OpenHandCursor)
        # self.setAcceptedMouseButtons(Qt.LeftButton)
        # self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setZValue(1)

        self.id = id

        self.penColor = Qt.black
        self.penWidth = 2

        self.baseBoundRect = QRectF(0, 0, 25, 25)

        # Nodes should always appear in the same way and ignore scalings etc.
        # NOTE: With this, the bounding box does not fit to the node sizes anymore.
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, enabled=True)

    def boundingRect(self):
        return self.baseBoundRect

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing, True)

        painter.setPen(QPen(self.penColor, self.penWidth))
        painter.setBrush(Qt.white)

        painter.setFont(QFont("Consolas", 12, QFont.Normal))
        painter.drawEllipse(0, 0, 25, 25)
        painter.drawText(0, 0, 25, 25, Qt.AlignCenter, str(self.id))
