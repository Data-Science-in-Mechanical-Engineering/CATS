from PyQt5.QtWidgets import QGraphicsView


class MyQGraphicsView(QGraphicsView):
    def wheelEvent(self, event):
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        zoomFactor = 1.1

        if event.angleDelta().y() > 0:
            self.scale(zoomFactor, zoomFactor)
        else:
            self.scale(1 / zoomFactor, 1 / zoomFactor)
