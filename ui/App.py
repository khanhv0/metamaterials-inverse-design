from ml_workflow.design_predictor import DesignPredictor
from pipeline.mesh_generator import MeshGenerator
from .DataframeList import DataframeList
from .PlotWidget import PlotWidget
from .Label import Label
from PySide6 import QtCore, QtWidgets, QtGui
import pandas as pd

class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.generator = MeshGenerator()
        self.predictor = DesignPredictor("./models_pth/forward_model_with_scaler.pth", 
                                         "./models_pth/inverse_model_with_scaler.pth")
        
        self.central_widget = MainWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.df_list.loaded_df.hasChangedSignal.connect(self.update_save_action)

        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")

        open_csv_action = QtGui.QAction("Predict from CSV", self)
        open_csv_action.triggered.connect(self.open_csv_context)

        self.save_action = QtGui.QAction("Save results", self)
        self.save_action.triggered.connect(self.save_result)
        self.save_action.setEnabled(False)

        load_action = QtGui.QAction("Load results", self)
        load_action.triggered.connect(self.load_result)

        self.save_mesh_action = QtGui.QAction("Save predicted designs", self)
        self.save_mesh_action.triggered.connect(self.save_meshes)
        self.save_mesh_action.setEnabled(False)

        exit_action = QtGui.QAction("Exit", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(open_csv_action)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_mesh_action)
        file_menu.addAction(load_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)


    @QtCore.Slot()
    def open_csv_context(self):
        csv_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Open File",
            "./inputs",
            "CSV Files (*.csv)"
        )

        if(not csv_path):
            return
        
        print(f"Opening {csv_path}")

        self.central_widget.df_list.loaded_df.value = self.predictor.predict_from_csv(csv_path)

    @QtCore.Slot()
    def save_result(self):
        csv_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "Save File",
            "./outputs",
            "CSV Files (*.csv)"
        )

        if(not csv_path):
            return

        print(f"Saving results into {csv_path}")

        self.central_widget.df_list.loaded_df.value.to_csv(csv_path, index=False)

    @QtCore.Slot()
    def load_result(self):
        csv_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Open File",
            "./outputs",
            "CSV Files (*.csv)"
        )

        if(not csv_path):
            return
        
        print(f"Opening {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            if "possibly_problematic" not in df.columns:
                print("The CSV file is not a prediction results file")
                df = None
        except Exception as e:
            print(e)
            df = None

        self.central_widget.df_list.loaded_df.value = df

    
    @QtCore.Slot(object)
    def update_save_action(self, df):
        is_defined = df is not None
        self.save_action.setEnabled(is_defined)
        self.save_mesh_action.setEnabled(is_defined)

    @QtCore.Slot()
    def save_meshes(self):
        designs = tuple(tuple(row) for row in self.central_widget.df_list.loaded_df.value.iloc[:, :8].to_numpy())
        self.generator.generateMeshesBatch(designs, output_path="./outputs/batch_results")
    



class MainWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        layout = QtWidgets.QVBoxLayout(self)
        canvas_layout = QtWidgets.QHBoxLayout()

        self.canvas = PlotWidget()
        self.df_list = DataframeList(self.canvas)

        canvas_layout.addWidget(self.canvas, stretch = 3)
        canvas_layout.addWidget(self.df_list, stretch = 1)

        layout.addLayout(canvas_layout, stretch=4)

        label_layout = QtWidgets.QVBoxLayout()
        self.labels = [Label("Target Poisson:"), Label("Target Young:"),
                       Label("Target Volume fraction:"), Label("Predicted Poisson:"),
                       Label("Predicted Young:"), Label("Predicted Volume fraction:")]
        
        for i in range(0, len(self.labels), 3):
            horizontal_layout = QtWidgets.QHBoxLayout()
            horizontal_layout.addWidget(self.labels[i])
            horizontal_layout.addWidget(self.labels[i + 1])
            horizontal_layout.addWidget(self.labels[i + 2])
            label_layout.addLayout(horizontal_layout)

        layout.addLayout(label_layout, stretch=1)

