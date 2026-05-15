import json

from PyQt5.QtWidgets import QComboBox, QDialog, QFileDialog, QLineEdit
from TestcasesDialogUI import Ui_TestcasesDialog


class TestcasesDialog(QDialog, Ui_TestcasesDialog):
    def __init__(self, config):
        QDialog.__init__(self)
        self.setupUi(self)
        self.config = config

        self.openTestbedBtn.clicked.connect(self.openTestbedDialog)
        self.saveConfigBtn.clicked.connect(self.saveConfig)
        self.loadConfigBtn.clicked.connect(self.loadConfig)
        self.closeConfigBtn.clicked.connect(self.closeConfig)

        # default items in combo boxes
        self.simulationModeBox.insertItems(0, ["default", "replayTrace"])
        self.logLevelBox.insertItems(
            0, ["default", "packets", "node states", "packets and node states"]
        )
        self.logSaveDetailsBox.insertItems(
            0, ["drop", "log structure", "log file", "extra files"]
        )
        self.smartShutdownBox.insertItems(0, ["yes", "no"])
        self.immediateEliminationBox.insertItems(0, ["yes", "no"])
        self.recursiveNeighborhoodBox.insertItems(0, ["yes", "no"])
        self.mulOnRxBox.insertItems(0, ["yes", "no"])
        self.mulOnRequestBox.insertItems(0, ["yes", "no"])
        self.mulOnOwnBox.insertItems(0, ["yes", "no"])
        self.emptyPacketStrategyBox.insertItems(0, ["own", "first", "random"])
        self.coordinatedSlottingBox.insertItems(0, ["yes", "no"])
        self.requestModeBox.insertItems(0, ["column", "pivot", "column and pivot"])
        self.columnSearchModeBox.insertItems(0, ["pivot", "all"])
        self.rxSnoopBox.insertItems(0, ["yes", "no"])

        # mapping of config properties to GUI elements
        self.configGUIMapping = [
            ("labelLine", "label"),
            ("testbedLine", "testbed"),
            ("numRoundsLine", "numRounds"),
            ("numSlotsLine", "numSlots"),
            ("payloadSizeLine", "payloadSize"),
            ("payloadDistributionLine", "payloadDistribution"),
            ("simulationModeBox", "simulationMode"),
            ("logLevelBox", "logLevel"),
            ("logSaveDetailsBox", "logSaveDetails"),
            ("fieldSizeLine", "fieldSize"),
            ("smartShutdownBox", "smartShutdown"),
            ("immediateEliminationBox", "immediateElimination"),
            ("recursiveNeighborhoodBox", "recursiveNeighborhood"),
            ("historyWindowLengthLine", "historyWindowLength"),
            ("fInitiatorLine", "fInitiator"),
            ("fTimeoutLine", "fTimeout"),
            ("fTxCurveLine", "fTxCurve"),
            ("exchangeTriggerSparsityLine", "exchangeTriggerSparsity"),
            ("allowedUpdatesLine", "allowedUpdates"),
            ("mulOnRxBox", "mulOnRx"),
            ("mulOnRequestBox", "mulOnRequest"),
            ("mulOnOwnBox", "mulOnOwn"),
            ("includeOwnLine", "includeOwn"),
            ("fAgeToPLine", "fAgeToP"),
            ("emptyPacketStrategyBox", "emptyPacketStrategy"),
            ("coordinatedSlottingBox", "coordinatedSlotting"),
            ("fpOwnLine", "fpOwn"),
            ("fpForeignLine", "fpForeign"),
            ("fpInitLine", "fpInit"),
            ("requestModeBox", "requestMode"),
            ("columnSearchModeBox", "columnSearchMode"),
            ("rxSnoopBox", "rxSnoop"),
            ("fTxColumnYesNoLine", "fTxColumnYesNo"),
            ("fTxPivotYesNoLine", "fTxPivotYesNo"),
            ("fpHelplessLine", "fpHelpless"),
        ]

        # load default config
        self.configToGUI()

    def configToGUI(self):
        for g, c in self.configGUIMapping:
            if type(getattr(self, g)) == QLineEdit:
                getattr(self, g).setText(getattr(self.config, c))
            if type(getattr(self, g)) == QComboBox:
                getattr(self, g).setCurrentIndex(
                    getattr(self, g).findText(getattr(self.config, c))
                )

    def GUItoConfig(self):
        for g, c in self.configGUIMapping:
            if type(getattr(self, g)) == QLineEdit:
                setattr(self.config, c, getattr(self, g).text())
            if type(getattr(self, g)) == QComboBox:
                setattr(self.config, c, getattr(self, g).currentText())

    def saveConfig(self):
        self.GUItoConfig()

        fileName = QFileDialog.getSaveFileName(
            None, "Save Configuration", "", "Mixer Configurations (*.mc)"
        )[0]
        if fileName:
            with open(fileName, "w") as f:
                json.dump(self.config.__dict__, f, indent=4)

    def loadConfig(self):
        fileName = QFileDialog.getOpenFileName(
            None, "Open Configuration", "", "Mixer Configurations (*.mc)"
        )[0]
        if fileName:
            with open(fileName, "r") as f:
                self.config.__dict__ = json.load(f)
                self.configToGUI()

    def closeConfig(self):
        self.close()

    def openTestbedDialog(self):
        fileName = QFileDialog.getOpenFileName(
            None, "Open Testbed File", "", "Testbed Files (*)"
        )[0]
        if fileName:
            self.testbedLine.setText(fileName)
