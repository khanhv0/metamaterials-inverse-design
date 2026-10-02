from PySide6 import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from pipeline.mesh_generator import MeshGenerator
import numpy as np

class PlotWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        layout = QtWidgets.QVBoxLayout(self)

        self.fig = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.fig)
        
        layout.addWidget(self.canvas)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.mesh_generator = MeshGenerator(boundary_steps=5)

        layout.addWidget(self.toolbar)

    def draw_mesh(self, all_vertices, inner_segments, outer_segments):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(all_vertices[inner_segments, 0], all_vertices[inner_segments, 1], "orange")
        ax.plot(all_vertices[outer_segments, 0], all_vertices[outer_segments, 1], "orange")
        ax.plot([-10, -10, 10, 10, -10], [10, -10, -10, 10, 10])
        ax.set_aspect('equal', adjustable='box')
        self.canvas.draw()

    def clear(self):
        self.fig.clear()
        self.canvas.draw()

    def draw_mesh_from_parameters(self, args):
        self.mesh_generator.set_design_parameters(*args)
        inner_segments, outer_segments, all_vertices = self.mesh_generator.create_petal_curves()
        self.draw_mesh(all_vertices, inner_segments, outer_segments)

    def draw_inp_mesh(self, file_path):
        with open(file_path) as file:
            vertices = []
            segments = []
            collecting_vertices = True

            file.readline()
            for line in file:
                if collecting_vertices:
                    if "*ELEMENT" in line:
                        collecting_vertices = False
                        continue
                    
                    split_vals = line.split(",")
                    vertices.append([float(split_vals[1]), float(split_vals[2])])
                else:
                    if "*NSET" in line:
                        break

                    split_vals = [int(val) - 1 for val in line.split(",")]
                    segments.append(split_vals[1:])

        vertices = np.array(vertices)
        segments = np.array(segments)

        self.draw_mesh(vertices, segments)
