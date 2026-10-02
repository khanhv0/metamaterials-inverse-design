from PySide6 import QtWidgets, QtCore, QtGui
from .ObservableValue import ObservableValue

class DataframeList(QtWidgets.QListWidget):
    def __init__(self, canvas):
        super().__init__()

        self.canvas = canvas
        self.loaded_df = ObservableValue(None)
        self.loaded_df.hasChangedSignal.connect(self.show_dataframe)
        self.itemClicked.connect(self.on_click)

        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.on_right_click)

    @QtCore.Slot(object)
    def show_dataframe(self, df):
        self.clear()
        self.clear_info()
        if df is not None:
            new_items = [f"Design {i}" for i in range(df.shape[0])]
            self.addItems(new_items)

            n = self.count()

            for i in range(n):
                if df["possibly_problematic"][i]:
                    self.item(i).setBackground(QtGui.QColor("orange"))

            if n > 0:
                self.on_click(self.item(0))

    @QtCore.Slot(object)
    def on_click(self, item):
        row = self.row(item)
        self.canvas.draw_mesh_from_parameters(self.loaded_df.value.iloc[row, :8].to_numpy())
        parent = self.parent()
        for index, value in enumerate(self.loaded_df.value.iloc[self.row(item), 8:-1].to_numpy()):
            parent.labels[index].value_text.value = str(value)

    @QtCore.Slot(QtCore.QPoint)
    def on_right_click(self, pos):
        item = self.itemAt(pos)

        if not item:
            return

        isCurrent = item == self.currentItem()
        index = self.row(item)
        self.takeItem(index)
        size = self.count()

        self.loaded_df.value.drop(index=index, inplace=True)
        self.loaded_df.value.reset_index(drop=True, inplace=True)

        if size == 0:
            self.loaded_df.value = None
        elif isCurrent:
            self.on_click(self.item(min(size, index)))
        

    def clear_info(self):
        self.canvas.clear()
        for label in self.parent().labels:
            label.value_text.value = ""


    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
        else:
            event.ignore()