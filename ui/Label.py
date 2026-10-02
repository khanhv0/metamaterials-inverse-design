from PySide6 import QtWidgets
from .ObservableValue import ObservableValue

class Label(QtWidgets.QWidget):
    def __init__(self, text):
        super().__init__()

        self.value_text = ObservableValue("")
        layout = QtWidgets.QHBoxLayout(self)

        label = QtWidgets.QLabel(text)
        layout.addWidget(label)

        value_input = QtWidgets.QLabel()
        layout.addWidget(value_input)

        self.value_text.hasChangedSignal.connect(value_input.setText)