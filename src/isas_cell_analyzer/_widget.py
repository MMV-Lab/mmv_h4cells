from qtpy.QtWidgets import (
    QLabel,
    QGridLayout,
    QVBoxLayout,
    QPushButton,
    QWidget,
    QScrollArea,
    QLineEdit,
    QComboBox,
    QSizePolicy,
    QMessageBox,
)
import napari
import numpy as np
from typing import List, Tuple
from pathlib import Path
from isas_cell_analyzer._reader import open_dialog, read
from isas_cell_analyzer._writer import save_dialog, write
from napari.layers.labels.labels import Labels
from scipy import ndimage


class CellAnalyzer(QWidget):
    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__()
        self.viewer = viewer

        self.label_layer: Labels = None  # label layer to evaluate
        self.accepted_cells: np.ndarray  # label layer esque for all accepted cells
        self.current_cell_layer: Labels = (
            None  # label layer consisting of the current cell to evaluate
        )
        self.metric_data: List[
            Tuple[int, int]
        ] = (
            []
        )  # list of tuples holding cell-id and metric data (adjust if more metrics need to be saved)
        self.mean_size: float = 0  # mean size of all selected cells
        self.std_size: float = (
            0  # standard deviation of size of all selected cells
        )
        # self.metric_value: datatype = 0
        self.amount_included: int = 0
        self.amount_excluded: int = 0
        self.amount_remaining: int = 0
        self.remaining_ids: List[int]

        ### QObjects
        # objects that can be updated are attributes of the class
        # for ease of access

        # Labels
        title = QLabel("ISAS Cell Analyzer")
        label_start_id = QLabel("Start analysis at:")
        label_included = QLabel("Included:")
        label_excluded = QLabel("Excluded:")
        label_remaining = QLabel("Remaining:")
        label_mean = QLabel("Mean size:")
        label_std = QLabel("Mean std:")
        # label_matric = QLabel("Metric name:")
        label_conversion = QLabel("1 pixel equals:")
        self.label_amount_included = QLabel("0")
        self.label_amount_excluded = QLabel("0")
        self.label_amount_remaining = QLabel("0")
        self.label_mean_included = QLabel("0")
        self.label_std_included = QLabel("0")
        # self.label_metric_included = QLabel("0")

        label_mean.setToolTip(
            "Only accounting for cells which have been included"
        )
        label_std.setToolTip(
            "Only accounting for cells which have been included"
        )

        # Buttons
        self.btn_start_analysis = QPushButton("Start analysis")
        self.btn_export = QPushButton("Export")
        self.btn_import = QPushButton("Import")
        self.btn_include = QPushButton("Include")
        self.btn_exclude = QPushButton("Exclude")
        btn_undo = QPushButton("Undo")

        self.btn_export.setToolTip("Export tooltip")
        self.btn_import.setToolTip("Import tooltip")
        self.btn_include.setToolTip("J")
        self.btn_include.setToolTip("F")
        btn_undo.setToolTip("B")

        self.btn_start_analysis.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_include.setEnabled(False)
        self.btn_exclude.setEnabled(False)

        # LineEdits
        self.lineedit_start_id = QLineEdit()
        self.lineedit_conversion_rate = QLineEdit()

        # Comboboxes
        self.combobox_conversion_unit = QComboBox()

        self.combobox_conversion_unit.addItems(["mm", "µm", "nm"])

        # Horizontal lines
        spacer = QWidget()
        spacer.setFixedHeight(4)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        line1 = QWidget()
        line1.setFixedHeight(4)
        line1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line1.setStyleSheet("background-color: #c0c0c0")

        line2 = QWidget()
        line2.setFixedHeight(4)
        line2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line2.setStyleSheet("background-color: #c0c0c0")

        ### GUI
        content = QWidget()
        content.setLayout(QGridLayout())
        content.layout().addWidget(title, 0, 0, 1, -1)

        content.layout().addWidget(spacer, 1, 0, 1, -1)

        content.layout().addWidget(self.btn_import, 2, 0, 1, 1)
        content.layout().addWidget(self.btn_export, 2, 2, 1, 1)

        content.layout().addWidget(self.btn_start_analysis, 3, 0, 1, 1)
        content.layout().addWidget(label_start_id, 3, 1, 1, 1)
        content.layout().addWidget(self.lineedit_start_id, 3, 2, 1, 1)

        content.layout().addWidget(line1, 4, 0, 1, -1)

        content.layout().addWidget(self.btn_exclude, 5, 0, 1, 1)
        content.layout().addWidget(btn_undo, 5, 1, 1, 1)
        content.layout().addWidget(self.btn_include, 5, 2, 1, 1)

        content.layout().addWidget(label_included, 6, 0, 1, 1)
        content.layout().addWidget(self.label_amount_included, 6, 2, 1, 1)

        content.layout().addWidget(label_mean, 7, 0, 1, 1)
        content.layout().addWidget(self.label_mean_included, 7, 2, 1, 1)

        content.layout().addWidget(label_std, 8, 0, 1, 1)
        content.layout().addWidget(self.label_std_included, 8, 2, 1, 1)

        # content.layout().addWidget(self.label_metric_included, 9, 0, 1, 1)
        # content.layout().addWidget(self.label_metric_included, adjust, the, rows, below)

        content.layout().addWidget(label_excluded, 9, 0, 1, 1)
        content.layout().addWidget(self.label_amount_excluded, 9, 2, 1, 1)

        content.layout().addWidget(label_remaining, 10, 0, 1, 1)
        content.layout().addWidget(self.label_amount_remaining, 10, 2, 1, 1)

        content.layout().addWidget(line2, 11, 0, 1, -1)

        content.layout().addWidget(label_conversion, 12, 0, 1, 1)
        content.layout().addWidget(self.lineedit_conversion_rate, 12, 1, 1, 1)
        content.layout().addWidget(self.combobox_conversion_unit, 12, 2, 1, 1)

        scroll_area = QScrollArea()
        scroll_area.setWidget(content)
        scroll_area.setWidgetResizable(True)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)

        # Hotkeys

        viewer.bind_key("j", self.on_hotkey_include)
        viewer.bind_key("f", self.on_hotkey_exclude)
        viewer.bind_key("b", self.on_hotkey_undo)

        self.viewer.layers.events.inserted.connect(self.get_label_layer)

    def get_label_layer(self, event):
        if not (self.label_layer is None and isinstance(event.value, Labels)):
            return
        self.label_layer = event.value
        if len(self.metric_data) > 0:
            unique_ids = np.unique(self.label_layer.data)
            max_id = max(self.accepted_cells)
            self.remaining_ids = [
                value for value in unique_ids if value > max_id
            ]
            self.amount_remaining = len(self.remaining_ids)
            self.amount_excluded = len(
                unique_ids - self.amount_included - self.amount_remaining
            )
            self.lineedit_start_id.setText(str(min(self.remaining_ids)))
            self.btn_start_analysis.setEnabled(True)

        self.update_labels()

    def update_labels(self):
        self.label_amount_excluded.setText(str(self.amount_excluded))
        self.label_amount_included.setText(str(self.amount_included))
        self.label_amount_remaining.setText(str(self.amount_remaining))
        self.label_mean_included.setText(
            f"{str(self.mean_size)*self.lineedit_conversion_rate} {self.combobox_conversion_unit.currentText()}"
        )
        self.label_std_included.setText(
            f"{str(self.std_size)*self.lineedit_conversion_rate} {self.combobox_conversion_unit.currentText()}"
        )
        # self.label_metric.included.setText(new value)

    def start_analysis_on_click(self):
        self.btn_exclude.setEnabled(True)
        self.btn_include.setEnabled(True)
        self.btn_import.setEnabled(False)
        self.btn_export.setEnabled(True)
        self.label_layer.opacity = 0.3
        self.current_cell_layer = self.viewer.add_labels(
            np.zeros_like(self.label_layer.data), name="Current Cell"
        )
        # start iterating through ids to create label layer for and zoom into centroid of label

    def display_cell(self):
        if len(self.remaining_ids) > 0:
            # display cell with id self.remaining_ids[0]
            self.current_cell_layer.data[:] = 0
            indices = np.where(self.label_layer.data == self.remaining_ids[0])
            self.current_cell_layer[indices] = self.remaining_ids[0]
            centroid = ndimage.center_of_mass(
                self.current_cell_layer,
                labels=self.current_cell_layer,
                index=self.remaining_ids[0],
            )
            self.viewer.dims.set_point(centroid)
            self.viewer.camera.zoom = 2.0

    def export_on_click(self):
        csv_filepath = Path(save_dialog())
        tiff_filepath = csv_filepath.with_suffix(".tiff")
        write(
            csv_filepath,
            (
                self.metric_data,
                (self.mean_size, self.std_size),
                (
                    int(self.lineedit_conversion_rate.text()),
                    self.combobox_conversion_unit.currentText(),
                ),
            ),
        )
        write(tiff_filepath, self.accepted_cells)

    def import_on_click(self):
        csv_filepath = Path(open_dialog())
        tiff_filepath = csv_filepath.with_suffix(".tiff")
        self.metric_data, metrics = read(csv_filepath)
        self.mean_size, self.std_size = metrics #, self.metric_value = ...
        self.accepted_cells = read(tiff_filepath)
        self.btn_export.setEnabled(True)
        max_id = max(self.accepted_cells)
        self.amount_included = len(np.unique(self.accepted_cells))

        if not self.label_layer is None:
            unique_ids = np.unique(self.label_layer.data)
            self.remaining_ids = [
                value for value in unique_ids if value > max_id
            ]
            self.amount_remaining = len(self.remaining_ids)
            self.amount_excluded = len(
                unique_ids - self.amount_included - self.amount_remaining
            )
            self.lineedit_start_id.setText(str(min(self.remaining_ids)))
            self.btn_start_analysis.setEnabled(True)

        self.update_labels()

    def on_hotkey_include(self, _):
        if self.btn_include.isEnabled():
            print("hotkey for include:", end=" ")
            self.include_on_click()

    def include_on_click(self):
        overlap_indices = np.intersect1d(
            np.nonzero(self.accepted_cells, self.current_cell_layer)
        )
        if len(overlap_indices) > 0:
            self.current_cell_layer.opacity = 0.3
            overlap_layer = self.viewer.add_labels(
                np.zeros_like(self.label_layer.data), name="Overlap"
            )
            overlap_layer[overlap_indices] = self.remaining_ids[0] + 1
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            msg.setText(
                "Overlap detected and highlighted. Please remove the overlap!"
            )
            msg.exec()
            self.current_cell_layer.opacity = 0.7
            self.viewer.remove(overlap_layer)
            return

        self.accepted_cells += self.current_cell_layer.data
        self.amount_included += 1
        current_id = self.remaining_ids.pop(0)
        self.amount_remaining = len(self.remaining_ids)
        self.metric_data.append(
            (current_id, np.count_nonzero(self.current_cell_layer.data))
        )

        self.calculate_metrics()
        self.update_labels()
        print("included")

        if len(self.remaining_ids) > 0:
            self.display_cell()

    def on_hotkey_exclude(self, _):
        if self.btn_exclude.isEnabled():
            print("hotkey for exclude:", end=" ")
            self.exclude_on_click()

    def exclude_on_click(self):
        self.amount_excluded += 1
        self.remaining_ids.pop(0)
        self.amount_remaining = len(self.remaining_ids)

        self.update_labels()
        print("excluded")

        if len(self.remaining_ids) > 0:
            self.display_cell()

    def calculate_metrics(self):
        sizes = [t[1] for t in self.metric_data]
        self.mean_size = np.mean(sizes)
        self.std_size = np.std(sizes)
        # self.metric = metric_calculation

    def on_hotkey_undo(self, _):
        self.undo()

    def undo(self):
        if len(self.metric_data) > 0:
            max_included, _ = self.metric_data[-1]
        else:
            max_included = -1
        unique_ids = [np.unique(self.label_layer.data)]
        excluded_ids = [cell_id for cell_id in unique_ids if cell_id not in self.remaining_ids and cell_id not in self.metric_data[:,0]]
        if len(excluded_ids) > 0:
            max_excluded = excluded_ids[-1]
        else:
            max_excluded = -1

        id_to_undo = max(max_excluded, max_included)
        if id_to_undo == -1:
            return

        if max_included > max_excluded:
            if len(self.metric_data) > 1:
                self.metric_data.pop(-1)
                indices = np.where(self.accepted_cells == id_to_undo)
                self.accepted_cells[indices] = 0
                self.remaining_ids.insert(0, id_to_undo)
                self.amount_included -= 1
        else:
            self.remaining_ids.insert(0, id_to_undo)
            self.amount_excluded -=1

        self.label_amount_remaining = len(self.remaining_ids)
        self.update_labels()
