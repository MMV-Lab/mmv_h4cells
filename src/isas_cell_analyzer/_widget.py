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
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtCore import Qt
import cv2
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
        self.included_layer: Labels = None  # label layer of all included cells
        self.excluded_layer: Labels = None  # label layer of all excluded cells
        self.remaining_layer: Labels = (
            None  # label layer of all remaining cells
        )
        self.metric_data: List[
            Tuple[int, int, Tuple(int, int)]
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
        self.evaluated_ids: List[int] = []

        ### QObjects
        # objects that can be updated are attributes of the class
        # for ease of access

        # Labels
        title = QLabel("<h1>MMV-Cell_Analyzer</h1>")
        self.label_start_id = QLabel("Start analysis at:")
        label_included = QLabel("Included:")
        label_excluded = QLabel("Excluded:")
        label_remaining = QLabel("Remaining:")
        label_mean = QLabel("Mean size:")
        label_std = QLabel("Std size:")
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
        self.btn_show_included = QPushButton("Show Included")
        self.btn_show_excluded = QPushButton("Show Excluded")
        self.btn_show_remaining = QPushButton("Show Remaining")
        self.btn_segment = QPushButton("Draw own cell")

        self.btn_start_analysis.clicked.connect(self.start_analysis_on_click)
        self.btn_export.clicked.connect(self.export_on_click)
        self.btn_import.clicked.connect(self.import_on_click)
        self.btn_include.clicked.connect(self.include_on_click)
        self.btn_exclude.clicked.connect(self.exclude_on_click)
        btn_undo.clicked.connect(self.undo_on_click)
        self.btn_show_included.clicked.connect(self.show_included_on_click)
        self.btn_show_excluded.clicked.connect(self.show_excluded_on_click)
        self.btn_show_remaining.clicked.connect(self.show_remaining_on_click)
        self.btn_segment.clicked.connect(self.draw_own_cell)

        self.btn_export.setToolTip("Export tooltip")
        self.btn_import.setToolTip("Import tooltip")
        self.btn_include.setToolTip("J")
        self.btn_exclude.setToolTip("F")
        btn_undo.setToolTip("B")

        self.btn_start_analysis.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_include.setEnabled(False)
        self.btn_exclude.setEnabled(False)
        self.btn_show_included.setEnabled(False)
        self.btn_show_excluded.setEnabled(False)
        self.btn_show_remaining.setEnabled(False)
        self.btn_segment.setEnabled(False)

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
        content.layout().addWidget(self.btn_segment, 2, 1, 1, 1)
        content.layout().addWidget(self.btn_export, 2, 2, 1, 1)

        content.layout().addWidget(self.btn_start_analysis, 3, 0, 1, 1)
        content.layout().addWidget(self.label_start_id, 3, 1, 1, 1)
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

        content.layout().addWidget(self.btn_show_included, 13, 0, 1, 1)
        content.layout().addWidget(self.btn_show_excluded, 13, 1, 1, 1)
        content.layout().addWidget(self.btn_show_remaining, 13, 2, 1, 1)

        scroll_area = QScrollArea()
        scroll_area.setWidget(content)
        scroll_area.setWidgetResizable(True)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)

        # Hotkeys

        hotkeys = self.viewer.keymap.keys()
        custom_binds = [
            ("J", self.on_hotkey_include),
            ("F", self.on_hotkey_exclude),
            ("B", self.on_hotkey_undo),
            ("V", self.toggle_visibility_label_layers_hotkey),
        ]
        for custom_bind in custom_binds:
            if not custom_bind[0] in hotkeys:
                viewer.bind_key(*custom_bind)
        # viewer.bind_key("j", self.on_hotkey_include)
        # viewer.bind_key("f", self.on_hotkey_exclude)
        # viewer.bind_key("b", self.on_hotkey_undo)
        # viewer.bind_key("v", self.toggle_visibility_label_layers)

        self.viewer.layers.events.inserted.connect(self.get_label_layer)
        for layer in self.viewer.layers:
            if isinstance(layer, Labels):
                self.set_label_layer(layer)
                break

    def draw_own_cell(self):
        if self.btn_segment.text() == "Draw own cell":
            self.btn_segment.setText("Confirm")
            # Display empty current cell layer
            self.current_cell_layer.data[:] = 0
            self.current_cell_layer.refresh()
            # Select current cell layer, set mode to paint
            self.viewer.layers.select_all()
            self.viewer.layers.selection.select_only(self.current_cell_layer)
            self.current_cell_layer.mode = "paint"
            # Select unique id
            self.current_cell_layer.selected_label = (
                max(np.max(self.accepted_cells), max(self.remaining_ids)) + 1
            )
        else:
            if self.include_on_click(True) is None:
                self.btn_segment.setText("Draw own cell")
                self.current_cell_layer.mode = "pan_zoom"

    def get_label_layer(self, event):
        if not (self.label_layer is None and isinstance(event.value, Labels)):
            return
        self.set_label_layer(event.value)

    def set_label_layer(self, layer):
        self.label_layer = layer
        self.btn_start_analysis.setEnabled(True)
        unique_ids = np.unique(self.label_layer.data)
        if len(self.metric_data) > 0:
            max_id = np.max(self.accepted_cells)
        else:
            max_id = 0
            self.accepted_cells = np.zeros_like(self.label_layer.data)
        self.remaining_ids = [value for value in unique_ids if value > max_id]
        self.amount_remaining = len(self.remaining_ids)
        self.amount_excluded = (
            len(unique_ids) - self.amount_included - self.amount_remaining - 1
        )

        self.lineedit_start_id.setText(str(min(self.remaining_ids)))

        self.update_labels()

    def update_labels(self):
        self.label_amount_excluded.setText(str(self.amount_excluded))
        self.label_amount_included.setText(str(self.amount_included))
        self.label_amount_remaining.setText(str(self.amount_remaining))
        if self.lineedit_conversion_rate.text() == "":
            unit = "pixels"
            factor = 1
        else:
            unit = self.combobox_conversion_unit.currentText() + "²"
            factor = float(self.lineedit_conversion_rate.text())
        self.label_mean_included.setText(
            f"{str(self.mean_size*factor)} {unit}"
        )
        self.label_std_included.setText(f"{str(self.std_size*factor)} {unit}")
        # self.label_metric.included.setText(new value)

    def start_analysis_on_click(self):
        self.btn_start_analysis.setEnabled(False)
        self.btn_exclude.setEnabled(True)
        self.btn_include.setEnabled(True)
        self.btn_import.setEnabled(False)
        self.btn_export.setEnabled(True)
        self.btn_show_included.setEnabled(True)
        self.btn_show_excluded.setEnabled(True)
        self.btn_show_remaining.setEnabled(True)
        self.btn_segment.setEnabled(True)
        self.label_start_id.setText("Next cell:")
        self.label_layer.opacity = 0.3
        self.current_cell_layer = self.viewer.add_labels(
            np.zeros_like(self.label_layer.data), name="Current Cell"
        )
        start_id = int(self.lineedit_start_id.text())
        if not start_id in self.remaining_ids:
            lower_ids = [
                value for value in self.remaining_ids if value < start_id
            ]
            if len(lower_ids) > 0:
                start_id = lower_ids[-1]
            else:
                start_id = self.remaining_ids[0]
        self.lineedit_start_id.setText(str(start_id))
        # start iterating through ids to create label layer for and zoom into centroid of label
        self.display_cell()

    def display_cell(self):
        if len(self.remaining_ids) > 0:
            next_id = int(self.lineedit_start_id.text())
            original_length = len(self.remaining_ids)
            if next_id > self.remaining_ids[0]:
                if next_id not in self.remaining_ids:
                    candidate_ids = [
                        i for i in self.remaining_ids if i < next_id
                    ]
                    if len(candidate_ids) > 0:
                        next_id = candidate_ids[-1]
                    else:
                        next_id = self.remaining_ids[0]
                self.remaining_ids = [
                    i for i in self.remaining_ids if i >= next_id
                ]
                self.amount_remaining - len(self.remaining_ids)
                amount_removed = original_length - len(self.remaining_ids)
                self.amount_excluded += amount_removed
            elif next_id < self.remaining_ids[0]:
                msg = QMessageBox()
                msg.setWindowTitle("napari")
                msg.setText(
                    "Lowering the next cell id might not be a good idea."
                )
                msg.exec()
                all_ids = np.unique(self.label_layer.data)
                lower_ids = [i for i in all_ids if i < next_id]
                accepted_ids = np.unique(self.accepted_cells)
                not_accepted_ids = [
                    i for i in all_ids if i not in accepted_ids
                ]
                next_id = max([lower_ids[-1]], min(not_accepted_ids))
                original_length = len(self.remaining_ids)
                self.remaining_ids = [
                    i
                    for i in all_ids
                    if i not in accepted_ids and i >= next_id
                ]
                self.amount_remaining = len(self.remaining_ids)
                amount_requeued = self.amount_remaining - original_length
                print(f"amount requeued: {amount_requeued}")
                self.amount_excluded -= amount_requeued
                mask = np.isin(self.accepted_cells, self.remaining_ids)
                self.accepted_cells[mask] = 0
                self.metric_data = [
                    (i, j, k)
                    for i, j, k in self.metric_data
                    if i not in self.remaining_ids
                ]
                print(f"next id: {next_id}")
            print(self.remaining_ids)
            self.lineedit_start_id.setText(str(next_id))
            self.update_labels()

            # display cell with id self.remaining_ids[0]
            next_label = int(self.lineedit_start_id.text())
            self.current_cell_layer.data[:] = 0
            indices = np.where(self.label_layer.data == next_label)
            self.current_cell_layer.data[indices] = next_label
            self.current_cell_layer.refresh()
            centroid = ndimage.center_of_mass(
                self.current_cell_layer.data,
                labels=self.current_cell_layer.data,
                index=next_label,
            )
            # centroid = tuple(int(value) for value in centroid)
            self.viewer.camera.center = centroid
            self.viewer.camera.zoom = 2.0  # !!
            self.current_cell_layer.selected_label = next_label
            if len(self.remaining_ids) > 1:
                self.lineedit_start_id.setText(str(self.remaining_ids[1]))
            else:
                self.lineedit_start_id.setText("")

    def export_on_click(self):
        csv_filepath = Path(save_dialog(self))
        if csv_filepath == ".csv":
            return
        tiff_filepath = csv_filepath.with_suffix(".tiff")
        factor = (
            float(self.lineedit_conversion_rate.text())
            if self.lineedit_conversion_rate.text() != ""
            else 1
        )
        self.metric_data = sorted(self.metric_data, key=lambda x: x[0])
        write(
            csv_filepath,
            self.metric_data,
            (self.mean_size, self.std_size),
            (
                factor,
                self.combobox_conversion_unit.currentText(),
            ),
        )
        write(tiff_filepath, self.accepted_cells)

    def import_on_click(self):
        csv_filepath = Path(open_dialog(self))
        if str(csv_filepath) == ".":
            return
        tiff_filepath = csv_filepath.with_suffix(".tiff")
        self.metric_data, metrics, pixelsize = read(csv_filepath)
        self.mean_size, self.std_size = metrics  # , self.metric_value = ...
        self.lineedit_conversion_rate.setText(str(pixelsize[0]))
        self.combobox_conversion_unit.setCurrentText(pixelsize[1])
        self.accepted_cells = read(tiff_filepath)
        self.btn_export.setEnabled(True)
        max_id = np.max(self.accepted_cells)
        self.amount_included = len(np.unique(self.accepted_cells)) - 1

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
            # print("hotkey for include:", end=" ")
            self.include_on_click()

    def include_on_click(self, self_drawn=False):
        nonzero_current = np.transpose(
            np.nonzero(self.current_cell_layer.data)
        )
        print(np.max(self.current_cell_layer.data))
        accepted_cells = np.copy(self.accepted_cells)
        combined_layer = accepted_cells + self.label_layer.data
        if not self_drawn:
            combined_layer[combined_layer == np.max(self.current_cell_layer.data)] = 0
        nonzero_other = np.transpose(np.nonzero(combined_layer))
        overlap = set(map(tuple, nonzero_current)).intersection(
            map(tuple, nonzero_other)
        )
        # nonzero_accepted = np.transpose(np.nonzero(self.accepted_cells))
        # filtered_original = np.copy(self.label_layer.data)
        # filtered_original[filtered_original == np.max(nonzero_current)] = 0
        # nonzero_original = np.transpose(np.nonzero(filtered_original))
        # overlap = set(map(tuple, nonzero_current)).intersection(
        #     set(map(tuple, nonzero_original)).union(
        #         map(tuple, nonzero_accepted)
        #     )
        # )
        # overlap = set(map(tuple, nonzero_accepted)).intersection(
        #     map(tuple, nonzero_current)
        # )
        if overlap:
            overlap_indices = tuple(np.array(list(overlap)).T)
            # self.current_cell_layer.data[overlap_indices] += 1
            self.label_layer.opacity = 0.2
            self.current_cell_layer.opacity = 0.3
            overlap_layer = self.viewer.add_labels(
                np.zeros_like(self.label_layer.data), name="Overlap", opacity=1
            )
            overlap_layer.data[overlap_indices] = self.remaining_ids[0] + 1
            overlap_layer.refresh()
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            msg.setText(
                "Overlap detected and highlighted. Please remove the overlap!"
            )
            msg.exec()
            self.current_cell_layer.opacity = 0.7
            self.label_layer.opacity = 0.3
            self.viewer.layers.remove(overlap_layer)
            self.viewer.layers.select_all()
            self.viewer.layers.selection.select_only(self.current_cell_layer)
            return 0

        # print(self.accepted_cells.shape)
        self.accepted_cells += self.current_cell_layer.data
        self.amount_included += 1
        if not self_drawn:
            current_id = self.remaining_ids.pop(0)
        else:
            current_id = np.max(self.current_cell_layer.data)
            self.lineedit_start_id.setText(str(self.remaining_ids[0]))
        self.evaluated_ids.append(current_id)
        print(f"including {current_id}")
        self.amount_remaining = len(self.remaining_ids)
        centroid = ndimage.center_of_mass(self.current_cell_layer.data)
        centroid = tuple(int(value) for value in centroid)
        self.metric_data.append(  # TODO
            (
                current_id,
                np.count_nonzero(self.current_cell_layer.data),
                centroid,
            )
        )

        self.calculate_metrics()
        self.update_labels()

        if len(self.remaining_ids) > 0:
            self.display_cell()

    def on_hotkey_exclude(self, _):
        if self.btn_exclude.isEnabled():
            self.exclude_on_click()

    def exclude_on_click(self):
        self.amount_excluded += 1
        excluded_id = self.remaining_ids.pop(0)
        self.evaluated_ids.append(excluded_id)
        print(f"excluding {excluded_id}")
        self.amount_remaining = len(self.remaining_ids)

        self.update_labels()

        if len(self.remaining_ids) > 0:
            self.display_cell()

    def calculate_metrics(self):
        sizes = [t[1] for t in self.metric_data]
        if len(sizes):
            self.mean_size = np.round(np.mean(sizes), 3)
            self.std_size = np.round(np.std(sizes), 3)
        else:
            self.mean_size = 0
            self.std_size = 0
        # self.metric = metric_calculation

    def on_hotkey_undo(self, _):
        self.undo_on_click()

    def undo_on_click(self):
        if len(self.evaluated_ids) == 0:
            return
        last_evaluated = self.evaluated_ids.pop(-1)
        print(f"undoing {last_evaluated}")
        if last_evaluated in np.unique(self.label_layer.data):
            self.remaining_ids.insert(0, last_evaluated)
            print(f"requeueing {last_evaluated}")
        if last_evaluated in self.accepted_cells:
            self.amount_included -= 1
            self.metric_data.pop(-1)
            indices = np.where(self.accepted_cells == last_evaluated)
            self.accepted_cells[indices] = 0
        else:
            self.amount_excluded -= 1
        self.lineedit_start_id.setText(str(self.remaining_ids[0]))

        self.amount_remaining = len(self.remaining_ids)
        self.calculate_metrics()
        self.update_labels()
        self.display_cell()

    def toggle_visibility_label_layers_hotkey(self, _):
        self.toggle_visibility_label_layers()

    def toggle_visibility_label_layers(self):
        for layer in self.viewer.layers:
            if isinstance(layer, Labels):
                layer.visible = not layer.visible

    def show_included_on_click(self):
        if self.btn_show_included.text() == "Show Included":
            self.btn_show_included.setText("Back")
            self.toggle_visibility_label_layers()
            self.included_layer = self.viewer.add_labels(
                self.accepted_cells, name="Included Cells"
            )
        else:
            self.viewer.layers.remove(self.included_layer)
            self.toggle_visibility_label_layers()
            self.btn_show_included.setText("Show Included")

    def show_excluded_on_click(self):
        if self.btn_show_excluded.text() == "Show Excluded":
            self.btn_show_excluded.setText("Back")
            self.toggle_visibility_label_layers()
            data = self.label_layer.data
            accepted = np.unique(self.accepted_cells[self.accepted_cells != 0])
            mask = np.isin(data, np.append(accepted, self.remaining_ids))
            data[mask] = 0
            self.excluded_layer = self.viewer.add_labels(
                data, name="Excluded Cells"
            )
        else:
            self.viewer.layers.remove(self.excluded_layer)
            self.toggle_visibility_label_layers()
            self.btn_show_excluded.setText("Show Excluded")

    def show_remaining_on_click(self):
        if self.btn_show_remaining.text() == "Show Remaining":
            self.btn_show_remaining.setText("Back")
            self.toggle_visibility_label_layers()
            data = self.label_layer.data
            mask = ~np.isin(data, self.remaining_ids)
            data[mask] = 0
            self.remaining_layer = self.viewer.add_labels(
                data, name="Remaining Cells"
            )
        else:
            self.viewer.layers.remove(self.remaining_layer)
            self.toggle_visibility_label_layers()
            self.btn_show_remaining.setText("Show Remaining")
