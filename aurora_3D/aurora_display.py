# File: aurora_display.py

import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt6.Qt3DCore import QEntity, QTransform
from PyQt6.Qt3DRender import QCamera, QMesh
from PyQt6.Qt3DExtras import Qt3DWindow, QOrbitCameraController, QPhongMaterial
from PyQt6.QtGui import QVector3D, QColor
from PyQt6.QtCore import QUrl

class ModelViewer(QWidget):
    def __init__(self):
        super().__init__()

        # 3D Window
        self.view = Qt3DWindow()
        self.container = QWidget.createWindowContainer(self.view)
        self.setMinimumSize(800, 600)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.container)
        self.setLayout(layout)

        # Root Entity
        self.root_entity = QEntity()
        self.view.setRootEntity(self.root_entity)

        # Camera
        self.camera = self.view.camera()
        self.camera.lens().setPerspectiveProjection(45.0, 16.0 / 9.0, 0.1, 1000.0)
        self.camera.setPosition(QVector3D(0, 0, 10))
        self.camera.setViewCenter(QVector3D(0, 0, 0))

        # Camera Controller
        self.camera_controller = QOrbitCameraController(self.root_entity)
        self.camera_controller.setLinearSpeed(50)
        self.camera_controller.setLookSpeed(180)
        self.camera_controller.setCamera(self.camera)

        # Model Entity
        self.model_entity = QEntity(self.root_entity)

        self.model_mesh = QMesh()
        self.model_mesh.setSource(QUrl.fromLocalFile("E:/AURORA/aurora_3d/model/aurora.fbx"))  # <- Fix your path here
        self.model_entity.addComponent(self.model_mesh)

        self.model_transform = QTransform()
        self.model_transform.setScale3D(QVector3D(1, 1, 1))
        self.model_entity.addComponent(self.model_transform)

        self.default_material = QPhongMaterial(self.root_entity)
        self.default_material.setDiffuse(QColor(255, 255, 255))
        self.model_entity.addComponent(self.default_material)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ModelViewer()
    viewer.show()
    sys.exit(app.exec())
