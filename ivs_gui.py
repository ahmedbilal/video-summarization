import sys
import random
import subprocess

from pathlib import Path

from PySide2.QtWidgets import (QApplication, QLabel, QPushButton,
                               QCheckBox, QRadioButton, QWidget,
                               QFileDialog, QFormLayout,
                               QFrame, QGroupBox, QVBoxLayout)
from PySide2.QtCore import Slot, Qt
from PySide2.QtGui import QFont, QPixmap, QIcon


class MyWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        self.setStyleSheet("background-color:white; font-size:24px;")

        # Icon
        self.top_icon = QPixmap("ivs.jpg")
        self.top_icon_label = QLabel()
        self.top_icon_label.setAlignment(Qt.AlignCenter)
        self.top_icon_label.setPixmap(self.top_icon.scaled(256,256,Qt.KeepAspectRatio))
        
        # Title
        title_font = QFont("Helvatica", 24, QFont.Bold)
        self.title = QLabel("Intelligent Video Summarization")
        self.title.setFont(title_font)
        

        # Desired Objects
        self.desired_objects_checkbox = [QCheckBox("car", self), QCheckBox("motor-bike", self),
                                         QCheckBox("rickshaw", self), QCheckBox("cycle", self),
                                         QCheckBox("person", self)]
        self.desired_objects_label = QLabel("Select the desired objects")
        self.desired_objects = QGroupBox()
        self.desired_objects.setTitle("Select Desired Objects")

        # Add Options (Checkboxes) in Desired Objects
        self.desired_objects_layout = QVBoxLayout()
        self.desired_objects.setStyleSheet("font-weight:bold;")
        for checkbox in self.desired_objects_checkbox:
            self.desired_objects_layout.addWidget(checkbox)
        self.desired_objects.setLayout(self.desired_objects_layout)

        self.button = QPushButton("Select Video")
        self.button.setStyleSheet("background-color:blue; color:white;")

        self.start_button = QPushButton("Start")
        self.start_button.setStyleSheet("background-color:red; color:white;")

        # Detector
        self.detectors_list = [QRadioButton("yolo", self), QRadioButton("frcc", self),
                                         QRadioButton("pseudo", self)]
        self.detectors = QGroupBox()
        self.detectors.setTitle("Select Detection Algorithm")
        self.detectors_layout = QVBoxLayout()
        self.detectors.setStyleSheet("font-weight:bold;")

        for radio_button in self.detectors_list:
            self.detectors_layout.addWidget(radio_button)
        
        self.detectors.setLayout(self.detectors_layout)

        # Main Layout
        self.layout = QFormLayout()
        self.layout.addRow(self.top_icon_label)
        self.layout.addRow(self.title)
        self.layout.addRow(self.button)
        self.layout.addRow(self.desired_objects)
        self.layout.addRow(self.detectors)
        self.layout.addRow(self.start_button)
        self.setLayout(self.layout)

        # Connecting the signal
        self.button.clicked.connect(self.add_video_path)
        self.start_button.clicked.connect(self.start_summarizing)

    @Slot()
    def add_video_path(self):
        self.video_file_path = QFileDialog.getOpenFileName(self, "Select Video", str(Path.home()),
                                                            "Video File (*.mp4, *.m4v)")
        # Update the text of button only if the selected video path is not empty
        if self.video_file_path[0]:
            self.button.setText(self.video_file_path[0])

    @Slot()
    def start_summarizing(self):
        _desired_objects = []
        _detection_algorithm = ""

        # get desired objects
        for child in self.desired_objects.children():
            if hasattr(child, "isChecked") and child.isChecked():
                _desired_objects.append(child)
                print(child.text())

        # get detection algorithm
        for child in self.detectors.children():
            if hasattr(child, "isChecked") and child.isChecked():
                _detection_algorithm = child.text()
        
        print(dir(self.video_file_path[0]))
        subprocess.run(["python3", "main.py", "--video", self.video_file_path[0],
                        "--detector", _detection_algorithm])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Intelligent Video Summarization")
    # icon = QIcon("ivs.jpg")

    # app.setWindowIcon(icon)
    widget = MyWidget()
    widget.show()

    sys.exit(app.exec_())