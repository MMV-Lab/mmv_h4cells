import copy
import logging
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
import pandas as pd
from typing import List, Tuple, Set
from pathlib import Path
from mmv_h4cells._reader import open_dialog, read
from mmv_h4cells._writer import save_dialog, write
from napari.layers.labels.labels import Labels
from scipy import ndimage


class CellAnalyzer(QWidget):
    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__()
        self.viewer = viewer
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self.logger.addHandler(handler)
        self.logger.debug("Initializing CellAnalyzer...")

        self.layer_to_evaluate: Labels = None  # label layer to evaluate
        self.accepted_cells: (
            np.ndarray
        )  # label layer esque for all accepted cells
        self.current_cell_layer: Labels = (
            None  # label layer consisting of the current cell to evaluate
        )
        self.included_layer: Labels = None  # label layer of all included cells
        self.excluded_layer: Labels = None  # label layer of all excluded cells
        self.remaining_layer: Labels = (
            None  # label layer of all remaining cells
        )
        self.metric_data: List[Tuple[int, int, Tuple[int, int]]] = (
            []
        )  # list of tuples holding cell-id and metric data (adjust if more metrics need to be saved)
        self.mean_size: float = 0  # mean size of all selected cells
        self.std_size: float = (
            0  # standard deviation of size of all selected cells
        )
        # self.metric_value: datatype = 0
        self.remaining: Set[int] = set()  # set of all remaining cell ids
        self.included: Set[int] = set()  # set of all included cell ids
        self.excluded: Set[int] = set()  # set of all excluded cell ids
        self.undo_stack:List[int] = [] # stack of cell ids to undo
        # self.amount_included: int = 0
        # self.amount_excluded: int = 0
        # self.amount_remaining: int = 0
        # self.remaining_ids: List[int]
        # self.evaluated_ids: List[int] = []

        self.initialize_ui()

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

        self.viewer.layers.events.inserted.connect(self.get_label_layer)
        for layer in self.viewer.layers:
            if isinstance(layer, Labels):
                self.set_label_layer(layer)
                break

        self.logger.debug("CellAnalyzer initialized")
        self.logger.info("Ready to use")

    def on_hotkey_include(self, _):
        if self.btn_include.isEnabled():
            self.include_on_click()

    def on_hotkey_exclude(self, _):
        if self.btn_exclude.isEnabled():
            self.exclude_on_click()

    def on_hotkey_undo(self, _):
        if self.btn_undo.isEnabled():
            self.undo_on_click()

    def toggle_visibility_label_layers_hotkey(self, _):
        self.toggle_visibility_label_layers()

    def initialize_ui(self):
        self.logger.debug("Initializing UI...")

        ### QObjects
        # objects that can be updated are attributes of the class
        # for ease of access

        # Labels
        title = QLabel("<h1>MMV-Cell_Analyzer</h1>")
        self.label_next_id = QLabel("Start analysis at:")
        label_include = QLabel("Include:")
        label_included = QLabel("Included:")
        label_excluded = QLabel("Excluded:")
        label_remaining = QLabel("Remaining:")
        label_mean = QLabel("Mean size:")
        label_std = QLabel("Std size:")
        # label_metric = QLabel("Metric name:")
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
        self.btn_undo = QPushButton("Undo")
        self.btn_show_included = QPushButton("Show Included")
        self.btn_show_excluded = QPushButton("Show Excluded")
        self.btn_show_remaining = QPushButton("Show Remaining")
        self.btn_segment = QPushButton("Draw own cell")
        self.btn_include_multiple = QPushButton("Include multiple")

        self.btn_start_analysis.clicked.connect(self.start_analysis_on_click)
        self.btn_export.clicked.connect(self.export_on_click)
        self.btn_import.clicked.connect(self.import_on_click)
        self.btn_include.clicked.connect(self.include_on_click)
        self.btn_exclude.clicked.connect(self.exclude_on_click)
        self.btn_undo.clicked.connect(self.undo_on_click)
        self.btn_show_included.clicked.connect(self.show_included_on_click)
        self.btn_show_excluded.clicked.connect(self.show_excluded_on_click)
        self.btn_show_remaining.clicked.connect(self.show_remaining_on_click)
        self.btn_segment.clicked.connect(self.draw_own_cell)
        self.btn_include_multiple.clicked.connect(
            self.include_multiple_on_click
        )

        self.btn_export.setToolTip("Export tooltip")
        self.btn_import.setToolTip("Import tooltip")
        self.btn_include.setToolTip("J")
        self.btn_exclude.setToolTip("F")
        self.btn_undo.setToolTip("B")

        self.btn_start_analysis.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_include.setEnabled(False)
        self.btn_exclude.setEnabled(False)
        self.btn_undo.setEnabled(False)
        self.btn_show_included.setEnabled(False)
        self.btn_show_excluded.setEnabled(False)
        self.btn_show_remaining.setEnabled(False)
        self.btn_segment.setEnabled(False)
        self.btn_include_multiple.setEnabled(False)

        # LineEdits
        self.lineedit_next_id = QLineEdit()
        self.lineedit_conversion_rate = QLineEdit()
        self.lineedit_include = QLineEdit()

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
        content.layout().addWidget(self.label_next_id, 3, 1, 1, 1)
        content.layout().addWidget(self.lineedit_next_id, 3, 2, 1, 1)

        content.layout().addWidget(line1, 4, 0, 1, -1)

        content.layout().addWidget(self.btn_exclude, 5, 0, 1, 1)
        content.layout().addWidget(self.btn_undo, 5, 1, 1, 1)
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

        content.layout().addWidget(label_include, 14, 0, 1, 1)
        content.layout().addWidget(self.lineedit_include, 14, 1, 1, 1)
        content.layout().addWidget(self.btn_include_multiple, 14, 2, 1, 1)

        scroll_area = QScrollArea()
        scroll_area.setWidget(content)
        scroll_area.setWidgetResizable(True)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)

    def get_label_layer(self, event):
        self.logger.debug("New potential label layer detected...")
        if not (self.layer_to_evaluate is None and isinstance(event.value, Labels)):
            self.logger.debug("New layer invalid or already set")
            return
        self.logger.debug("Label layer is valid")
        self.set_label_layer(event.value)

    def set_label_layer(self, layer):
        self.logger.debug("Setting label layer...")
        self.layer_to_evaluate = layer
        self.btn_start_analysis.setEnabled(True)
        unique_ids = np.unique(self.layer_to_evaluate.data)
        self.logger.debug(f"{len(unique_ids)} unique ids found")
        if len(self.metric_data) > 0:
            max_id = np.max(self.accepted_cells)
            self.logger.debug(f"Highest evaluated id: {max_id}")
            self.remaining = set(unique_ids) - (self.included | self.excluded)
        else:
            max_id = 0
            self.accepted_cells = np.zeros_like(self.layer_to_evaluate.data)
            self.remaining = set(unique_ids) - {0}
        # self.remaining_ids = [value for value in unique_ids if value > max_id]
        # self.amount_remaining = len(self.remaining_ids)
        # self.amount_excluded = (
        #     len(unique_ids) - self.amount_included - self.amount_remaining - 1
        # )

        # self.lineedit_next_id.setText(str(min(self.remaining_ids)))
        self.lineedit_next_id.setText(str(min(self.remaining)))
        self.logger.debug("Sets updated")
        self.update_labels()

    def update_labels(self):
        self.logger.debug("Updating labels...")
        # self.label_amount_excluded.setText(str(self.amount_excluded))
        # self.label_amount_included.setText(str(self.amount_included))
        # self.label_amount_remaining.setText(str(self.amount_remaining))
        self.label_amount_excluded.setText(str(len(self.excluded)))
        self.label_amount_included.setText(str(len(self.included)))
        self.label_amount_remaining.setText(str(len(self.remaining)))
        if self.lineedit_conversion_rate.text().strip() == "":
            unit = "pixel"
            factor = 1
        else:
            unit = self.combobox_conversion_unit.currentText() + "²"
            factor = float(self.lineedit_conversion_rate.text())
        self.logger.debug(f"Conversion rate: {factor} {unit}")
        self.label_mean_included.setText(
            f"{str(self.mean_size*factor)} {unit}"
        )
        self.label_std_included.setText(f"{str(self.std_size*factor)} {unit}")
        # self.label_metric.included.setText(new value)

    def start_analysis_on_click(self):
        self.logger.debug("Analysis started...")
        self.btn_start_analysis.setEnabled(False)
        self.btn_exclude.setEnabled(True)
        self.btn_include.setEnabled(True)
        self.btn_import.setEnabled(False)
        self.btn_export.setEnabled(True)
        self.btn_show_included.setEnabled(True)
        self.btn_undo.setEnabled(True)
        self.btn_show_excluded.setEnabled(True)
        self.btn_show_remaining.setEnabled(True)
        self.btn_segment.setEnabled(True)
        self.btn_include_multiple.setEnabled(True)
        self.label_next_id.setText("Next cell:")
        self.layer_to_evaluate.opacity = 0.3
        self.current_cell_layer = self.viewer.add_labels(
            np.zeros_like(self.layer_to_evaluate.data), name="Current Cell"
        )
        start_id = int(self.lineedit_next_id.text()) # TODO: catch ValueError if not int
        if not start_id in self.remaining:
        # if not start_id in self.remaining_ids:
            self.logger.warning("Start id not in remaining ids")
            lower_ids = {value for value in self.remaining if value < start_id}
            # lower_ids = [
            #     value for value in self.remaining_ids if value < start_id
            # ]
            if len(lower_ids) > 0:
                self.logger.info("Using lower id")
                start_id = max(lower_ids)
                # start_id = lower_ids[-1]
            else:
                self.logger.info("Using lowest remaining id")
                start_id = min(self.remaining)
                # start_id = self.remaining_ids[0]
        self.lineedit_next_id.setText(str(min(x for x in self.remaining if x > start_id)))
        # start iterating through ids to create label layer for and zoom into centroid of label
        self.display_cell(start_id)

    def display_cell(self, cell_id: int):
        self.logger.debug(f"Displaying cell {cell_id}")
        self.current_cell_layer.data[:] = 0
        indices = np.where(self.layer_to_evaluate.data == cell_id)
        self.current_cell_layer.data[indices] = cell_id
        self.current_cell_layer.opacity = 0.7
        self.current_cell_layer.refresh()
        centroid = ndimage.center_of_mass(
            self.current_cell_layer.data,
            labels=self.current_cell_layer.data,
            index=cell_id,
        )
        self.viewer.camera.center = centroid
        self.viewer.camera.zoom = 7.5  # !!
        self.current_cell_layer.selected_label = cell_id

    def import_on_click(self):
        self.logger.debug("Importing data...")
        csv_filepath = Path(open_dialog(self))
        if str(csv_filepath) == ".":
            self.logger.debug("No file selected. Aborting.")
            return
        tiff_filepath = csv_filepath.with_suffix(".tiff")
        self.metric_data, metrics, pixelsize, self.excluded, self.undo_stack = read(
            csv_filepath
        )
        self.mean_size, self.std_size = metrics  # , self.metric_value = ...
        self.lineedit_conversion_rate.setText(str(pixelsize[0]))
        self.combobox_conversion_unit.setCurrentText(pixelsize[1])
        self.accepted_cells = read(tiff_filepath)
        self.included = set(pd.unique(self.accepted_cells.flatten())) - {0}
        # accepted_ids = [
        #     accepted
        #     for accepted in np.unique(self.accepted_cells)
        #     if accepted != 0
        # ]
        # self.evaluated_ids = excluded_cells + accepted_ids
        # self.evaluated_ids = sorted(self.evaluated_ids)
        self.btn_export.setEnabled(True)
        # self.amount_included = len(np.unique(self.accepted_cells)) - 1

        if not self.layer_to_evaluate is None:
            self.logger.debug("Filling in values for existing label layer")
            unique_ids = pd.unique(self.layer_to_evaluate.data.flatten())
            # accepted_cell_ids = pd.unique(self.accepted_cells)
            self.remaining = set(unique_ids) - (self.included | self.excluded | {0})
            # self.remaining_ids = [
            #     value
            #     for value in unique_ids
            #     if not value in self.evaluated_ids
            #     and not value in accepted_cell_ids
            # ]
            # self.amount_remaining = len(self.remaining_ids)
            # self.amount_excluded = (
            #     len(self.evaluated_ids) - self.amount_included
            # )
            self.lineedit_next_id.setText(str(min(self.remaining)))
            # self.lineedit_next_id.setText(str(min(self.remaining_ids)))
            self.btn_start_analysis.setEnabled(True)

        self.update_labels()

    def export_on_click(self):
        self.logger.debug("Exporting data...")
        csv_filepath = Path(save_dialog(self))
        if csv_filepath.name == ".csv":
            self.logger.debug("No file selected. Aborting.")
            return
        tiff_filepath = csv_filepath.with_suffix(".tiff")
        if self.lineedit_conversion_rate.text() == "":
            factor = 1
            unit = "pixel"
        else:
            factor = float(self.lineedit_conversion_rate.text()) # TODO: catch ValueError if not float
            unit = self.combobox_conversion_unit.currentText()
        # factor = (
        #     float(self.lineedit_conversion_rate.text())
        #     if self.lineedit_conversion_rate.text() != ""
        #     else None
        # )
        self.metric_data = sorted(self.metric_data, key=lambda x: x[0])
        # excluded = [
        #     int(value)
        #     for value in np.unique(self.layer_to_evaluate.data)
        #     if value in self.evaluated_ids
        #     and not value in np.unique(self.accepted_cells)
        # ]
        write(
            csv_filepath,
            self.metric_data,
            (self.mean_size, self.std_size),
            (
                factor,
                unit,
                # self.combobox_conversion_unit.currentText(),
            ),
            self.excluded,
            self.undo_stack,
        )
        self.logger.debug("Metrics written to csv")
        write(tiff_filepath, self.accepted_cells)
        self.logger.debug("Accepted cells written to tiff")

    def include_on_click(self, self_drawn=False):
        self.logger.debug("Including cell...")
        if len(self.remaining) < 1:
        # if len(self.remaining_ids) < 1:
            self.logger.info("No cell to include")
            return
        
        if self.check_for_overlap():
            return

        id_ = int(np.max(self.current_cell_layer.data))
        self.include(id_,self.current_cell_layer.data,not self_drawn)

        self.undo_stack.append(id_)

        if len(self.remaining) > 0:
        # if len(self.remaining_ids) > 0:
            self.display_next_cell()

    def exclude_on_click(self):
        self.logger.debug("Excluding cell...")
        if len(self.remaining) < 1:
        # if len(self.remaining_ids) < 1:
            self.logger.info("No cell to exclude")
            return
        # self.amount_excluded += 1
        current_id = int(max(pd.unique(self.current_cell_layer.data.flatten())))
        # current_id = np.max(self.current_cell_layer.data)
        self.excluded.add(current_id)
        self.remaining.remove(current_id)
        self.undo_stack.append(current_id)
        # self.remaining_ids.remove(current_id)
        # self.evaluated_ids.append(current_id)
        # self.amount_remaining = len(self.remaining_ids)

        self.update_labels()

        # if len(self.remaining_ids) > 0:
        if len(self.remaining) > 0:
            self.display_next_cell()

    def undo_on_click(self):
        self.logger.debug("Undoing last action...")
        if len(self.undo_stack) == 0:
        # if len(self.evaluated_ids) == 0:
            self.logger.info("No actions to undo")
            return
        self.logger.debug("Before undo:")
        self.logger.debug(f"Last evaluated: {self.undo_stack[-1]}")
        # self.logger.debug(f"Last evaluated: {self.evaluated_ids[-1]}")
        last_evaluated = self.undo_stack.pop(-1)
        # last_evaluated = self.evaluated_ids.pop(-1)
        if last_evaluated in self.layer_to_evaluate.data:
            self.logger.debug("Adding cell back to remaining")
            self.remaining.add(last_evaluated)
            # self.remaining_ids.insert(0, last_evaluated)
        if last_evaluated in self.accepted_cells:
            self.logger.debug("Removing cell from accepted")
            # self.amount_included -= 1
            self.metric_data.pop(-1)
            indices = np.where(self.accepted_cells == last_evaluated)
            self.accepted_cells[indices] = 0
            self.included.remove(last_evaluated)
        else:
            self.excluded.remove(last_evaluated)
            # self.amount_excluded -= 1
        self.lineedit_next_id.setText(str(last_evaluated))
        # self.lineedit_next_id.setText(str(self.remaining_ids[0]))

        # self.amount_remaining = len(self.remaining_ids)
        self.calculate_metrics()
        self.update_labels()
        self.display_next_cell()

    def show_included_on_click(self):
        if self.btn_show_included.text() == "Show Included":
            self.logger.debug("Showing included cells...")
            self.btn_show_included.setText("Back")
            self.toggle_visibility_label_layers()
            self.included_layer = self.viewer.add_labels(
                self.accepted_cells, name="Included Cells"
            )
        else:
            self.logger.debug("Hiding included cells...")
            self.viewer.layers.remove(self.included_layer)
            self.included_layer = None
            self.toggle_visibility_label_layers()
            self.btn_show_included.setText("Show Included")

    def show_excluded_on_click(self):
        if self.btn_show_excluded.text() == "Show Excluded":
            self.logger.debug("Showing excluded cells...")
            self.btn_show_excluded.setText("Back")
            self.toggle_visibility_label_layers()
            data = copy.deepcopy(self.layer_to_evaluate.data)
            mask = ~np.isin(data, self.excluded)
            # accepted = np.unique(self.accepted_cells[self.accepted_cells != 0])
            # mask = np.isin(data, np.append(accepted, self.remaining_ids))
            data[mask] = 0
            self.excluded_layer = self.viewer.add_labels(
                data, name="Excluded Cells"
            )
        else:
            self.logger.debug("Hiding excluded cells...")
            self.viewer.layers.remove(self.excluded_layer)
            self.excluded_layer = None
            self.toggle_visibility_label_layers()
            self.btn_show_excluded.setText("Show Excluded")

    def show_remaining_on_click(self):
        if self.btn_show_remaining.text() == "Show Remaining":
            self.logger.debug("Showing remaining cells...")
            self.btn_show_remaining.setText("Back")
            self.toggle_visibility_label_layers()
            data = copy.deepcopy(self.layer_to_evaluate.data)
            mask = ~np.isin(data, self.remaining)
            # mask = ~np.isin(data, self.remaining_ids)
            data[mask] = 0
            self.remaining_layer = self.viewer.add_labels(
                data, name="Remaining Cells"
            )
        else:
            self.logger.debug("Hiding remaining cells...")
            self.viewer.layers.remove(self.remaining_layer)
            self.remaining_layer = None
            self.toggle_visibility_label_layers()
            self.btn_show_remaining.setText("Show Remaining")

    def include_multiple_on_click(self):
        self.logger.debug(f"Including multiple cells for input {self.lineedit_include.text()}")
        given_ids = self.get_ids_to_include()
        if given_ids is None:
            self.logger.debug("No valid ids in input")
            return
        self.logger.debug(f"Given ids: {given_ids}")
        included, ignored, overlapped = self.include_multiple(given_ids)
        self.lineedit_include.setText("")
        self.lineedit_next_id.setText(str(min(self.remaining)))
        self.display_next_cell(False)
        msg = QMessageBox()
        msg.setWindowTitle("napari")
        msgtext = ""
        if len(included) > 0:
            msgtext += f"Cells included: {included}\n"
        if len(ignored) > 0:
            msgtext += f"Cells ignored as they are already evaluated: {ignored}\n"
            msgtext += "Only unprocessed cells can be included.\n"
        if len(overlapped) > 0:
            msgtext += f"Cells not included due to overlap: {overlapped}\n"
            msgtext += "Please remove the overlap(s)."
        msg.setText(msgtext)
        msg.exec_()
        # if self.lineedit_include.text() == "":
        #     return
        # included_ids = self.lineedit_include.text().split(",")
        # try:
        #     included_ids = [int(i) for i in included_ids]
        # except ValueError:
        #     self.logger.debug("Invalid input")
        #     msg = QMessageBox()
        #     msg.setWindowTitle("napari")
        #     msg.setText("Please enter a comma separated list of integers.")
        #     msg.exec_()
        #     return
        # processed_ids = [val for val in given_ids if val in self.remaining]
        # # processed_ids = [id for id in included_ids if id in self.remaining_ids]
        # self.logger.debug(f"ids given by user: {given_ids}")
        # self.logger.debug(f"ids being looked at: {processed_ids}")
        # for val in given_ids:
        #     if val in self.remaining:
        #     # if id in self.remaining_ids:
        #         self.display_cell(val)
        #         if self.check_for_overlap():
        #             break
        #         self.include()
        # self.lineedit_include.setText("")
        # self.lineedit_next_id.setText(str(self.remaining_ids[0]))
        # self.display_next_cell(False)
        # msg = QMessageBox()
        # msg.setWindowTitle("napari")
        # if len(processed_ids) == len(given_ids):
        #     msg.setText(f"Cells included: {processed_ids}")
        # elif len(processed_ids) == 0:
        #     msg.setText(
        #         f"No new cells included.\nAll specified cells already included or excluded.\nOnly unprocessed cells can be included."
        #     )
        # else:
        #     msg.setText(
        #         f"Cells included: {processed_ids}\nCells not included: {list(set(given_ids) - set(processed_ids))}\nAll other specified cells already included or excluded.\nOnly unprocessed cells can be included."
        #     )
        # msg.exec_()

    def get_ids_to_include(self):
        """
        Returns the ids of the cells to include or None.
        
        Returns:
        --------
        ids: list of int or None
            A list of the ids of the cells to include or None if the input is invalid.
        """
        ids = self.lineedit_include.text()
        if len(ids) == 0:
            return None
        ids = ids.split(",")
        try:
            ids = [int(i) for i in ids]
        except ValueError:
            self.logger.debug("Invalid input")
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            msg.setText("Please enter a comma separated list of integers.")
            msg.exec_()
            return None
        return ids
    
    def include_multiple(self, ids: List[int]) -> Tuple[Set[int], Set[int], Set[int]]:
        self.logger.debug("Including multiple cells...")
        included = set()
        ignored = set()
        overlapped = set()
        for val in ids:
            if val not in self.remaining:
                ignored.add(val)
                continue
            indices = np.where(self.layer_to_evaluate.data == val)
            print(np.sum(self.accepted_cells[indices]))
            if np.sum(self.accepted_cells[indices]):
                overlapped.add(val)
                continue
            data_array = np.zeros_like(self.layer_to_evaluate.data)
            data_array[indices] = val
            self.include(val, data_array)
            included.add(val)
        self.logger.debug("Multiple cells evaluated")
        return included, ignored, overlapped

    def draw_own_cell(self):
        if self.btn_segment.text() == "Draw own cell":
            self.logger.debug("Draw own cell initialized")
            self.btn_segment.setText("Confirm/Back")
            # Set next id label to current cell layer id
            current_id = str(np.max(self.current_cell_layer.data))
            self.lineedit_next_id.setText(current_id)
            # Display empty current cell layer
            self.current_cell_layer.data[:] = 0
            self.current_cell_layer.refresh()
            # Select current cell layer, set mode to paint
            self.viewer.layers.select_all()
            self.viewer.layers.selection.select_only(self.current_cell_layer)
            self.current_cell_layer.mode = "paint"
            # Select unique id
            self.current_cell_layer.selected_label = (
                max(np.max(self.accepted_cells), np.max(self.layer_to_evaluate.data)) + 1
            )
        else:
            self.logger.debug("Draw own cell confirmed")
            self.include_on_click(True)
            self.btn_segment.setText("Draw own cell")

    def display_next_cell(self, check_lowered=True):
        self.logger.debug("Displaying next cell...")
        if len(self.remaining) < 1:
        # if len(self.remaining_ids) < 0: # TODO: same in include/exclude
            self.logger.debug("No cells left to evaluate")
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            msg.setText("No more cells to evaluate.")
            msg.exec_()
            return

        given_id = int(self.lineedit_next_id.text())
        self.logger.debug(f"Id given by textfield: {given_id}")
        last_evaluated_id = self.undo_stack[-1] if len(self.undo_stack) > 0 else 0
        # last_evaluated_id = self.evaluated_ids[-1] if len(self.evaluated_ids) > 0 else 0
        self.logger.debug(f"Last evaluated id: {last_evaluated_id}")
        candidate_ids = [i for i in self.remaining if i > last_evaluated_id]
        next_id_computed = min(candidate_ids) if len(candidate_ids) > 0 else min(self.remaining)
        # candidate_ids = [i for i in self.remaining_ids if i > last_evaluated_id]
        # next_id_computed = min(candidate_ids) if len(candidate_ids) > 0 else self.remaining_ids[0]
        self.logger.debug(f"Computed next id: {next_id_computed}")

        if given_id != next_id_computed:
            if given_id not in self.remaining:
                msg = QMessageBox()
                msg.setWindowTitle("napari")
                msg.setText("Given id is not in remaining cells.")
                msg.exec_()
                candidate_ids = [i for i in self.remaining if i < given_id]
            # if given_id not in self.remaining_ids:
            #     candidate_ids = [i for i in self.remaining_ids if i < given_id]
                if len(candidate_ids) > 0:
                    self.logger.debug("Using highest lower id")
                    next_id = candidate_ids[-1]
                else:
                    self.logger.debug("Using lowest remaining id")
                    next_id = min(self.remaining)
                    # next_id = self.remaining_ids[0]
            else:
                self.logger.debug("Using given id (not computed)")
                next_id = given_id
        else:
            self.logger.debug("Using given id (computed)")
            next_id = given_id

        if not check_lowered:
            self.logger.debug("Skipping check for lowered id")

        if check_lowered and next_id < last_evaluated_id:
            self.logger.debug("Next id lower than last evaluated id")
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            if next_id_computed < last_evaluated_id:
                self.logger.debug("No higher id remaining")
                msg.setText(
                    "Dataset is finished. Jumping to earlier cells."
                )
            else:
                self.logger.debug("Higher id is remaining")
                msg.setText(
                    "Lowering the next cell id is a bad idea."
                )
            msg.exec_()
        self.display_cell(next_id)

        if len(self.remaining) > 1:
        # if len(self.remaining_ids) > 1:
            candidate_ids = [i for i in self.remaining if i > next_id]
            # candidate_ids = [i for i in self.remaining_ids if i > next_id]
            if len(candidate_ids) > 0:
                next_label = candidate_ids[0]
            else:
                next_label = min(self.remaining)
                # next_label = self.remaining_ids[0]
            self.lineedit_next_id.setText(str(next_label))
        else:
            self.lineedit_next_id.setText("")
        self.logger.debug("Value for next cell set")

    def include(self, id_: int, data_array: np.ndarray, remove_from_remaining: bool = True):
        self.logger.debug("Including cell...")
        self.accepted_cells += data_array
        # self.accepted_cells += self.current_cell_layer.data
        # current_id = np.max(self.current_cell_layer.data)
        if remove_from_remaining:
            self.remaining.remove(id_)
            # self.remaining.remove(current_id)
            # self.remaining_ids.remove(current_id)
        self.included.add(id_)

        self.add_cell_to_accepted(id_, data_array)
        # self.add_cell_to_accepted(current_id, self.current_cell_layer.data)

    def check_for_overlap(self, self_drawn=False):
        self.logger.debug("Checking for overlap...")
        overlap = self.get_overlap()

        if not overlap:
            return False
        
        self.handle_overlap(overlap, self_drawn)
        return True

    def get_overlap(self):
        """
        Returns the indices of the overlapping pixels between the current cell and the accepted cells.
        
        Returns:
        --------
        overlap: set
            A set of tuples containing the indices of the overlapping pixels.
        """
        self.logger.debug("Calculating overlap...")
        nonzero_current = np.nonzero(self.current_cell_layer.data)
        accepted_cells = np.copy(self.accepted_cells)
        nonzero_accepted = np.nonzero(accepted_cells)
        combined_layer = np.zeros_like(self.layer_to_evaluate.data)
        combined_layer[nonzero_current] += 1
        combined_layer[nonzero_accepted] += 1
        overlap = set(map(tuple, np.transpose(np.where(combined_layer == 2))))
        return overlap

    def add_cell_to_accepted(self, cell_id: int, data: np.ndarray):
        self.logger.debug("Adding cell to list of accepted...")
        self.included.add(cell_id)
        # self.amount_included += 1
        # self.evaluated_ids.append(id)
        centroid = ndimage.center_of_mass(data)
        centroid = tuple(int(value) for value in centroid)
        self.metric_data.append(  # TODO
            (
                cell_id,
                np.count_nonzero(data),
                centroid,
            )
        )
        # self.amount_remaining = len(self.remaining_ids)

        self.calculate_metrics()
        self.update_labels()

    def handle_overlap(self, overlap: set, user_drawn: bool = False):
        """
        Handles the overlap between the current cell and the accepted cells.
        
        Parameters:
        -----------
        overlap: set
            A set of tuples containing the indices of the overlapping pixels.
        user_drawn: bool
            A boolean indicating whether the current cell was drawn by the user.
        """
        self.logger.debug("Handling overlap...")
        overlap_indices = tuple(np.array(list(overlap)).T)
        self.layer_to_evaluate.opacity = 0.2
        self.current_cell_layer.opacity = 0.3
        self.logger.debug("Displaying overlap...")
        overlap_layer = self.viewer.add_labels(
            np.zeros_like(self.layer_to_evaluate.data), name="Overlap", opacity=1
        )
        overlap_layer.data[overlap_indices] = np.amax(self.current_cell_layer.data) + 1
        # overlap_layer.data[overlap_indices] = self.remaining_ids[0] + 1
        overlap_layer.refresh()
        msg = QMessageBox()
        msg.setWindowTitle("napari")
        msg.setText(
            "Overlap detected and highlighted. Please remove the overlap!"
        )
        if user_drawn:
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg.setDefaultButton(QMessageBox.Ok)
        return_value = msg.exec_()
        self.current_cell_layer.opacity = 0.7
        self.layer_to_evaluate.opacity = 0.3
        self.viewer.layers.remove(overlap_layer)
        self.logger.debug("Overlap display removed")
        self.viewer.layers.select_all()
        self.viewer.layers.selection.select_only(self.current_cell_layer)
        # if return_value == QMessageBox.Cancel and len(self.remaining_ids) > 0:
        if return_value == QMessageBox.Cancel and len(self.remaining) > 0:
            self.btn_segment.setText("Draw own cell")
            self.current_cell_layer.mode = "pan_zoom"

    def calculate_metrics(self):
        self.logger.debug("Calculating metrics...")
        sizes = [t[1] for t in self.metric_data]
        if len(sizes):
            self.mean_size = np.round(np.mean(sizes), 3)
            self.std_size = np.round(np.std(sizes), 3)
        else:
            self.mean_size = 0
            self.std_size = 0
        # self.metric = metric_calculation

    def toggle_visibility_label_layers(self):
        self.logger.debug("Toggling visibility of label layers...")
        for layer in self.viewer.layers:
            if isinstance(layer, Labels):
                layer.visible = not layer.visible
