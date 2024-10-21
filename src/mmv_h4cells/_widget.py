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
    QGroupBox,
    QDialog,
)
from qtpy.QtCore import QEvent

import napari
import numpy as np
import pandas as pd
from typing import List, Tuple, Set
from pathlib import Path
from mmv_h4cells import __version__ as version
from mmv_h4cells._reader import open_dialog, read
from mmv_h4cells._roi import analyse_roi
from mmv_h4cells._writer import save_dialog, write
from napari.layers.labels.labels import Labels
from scipy import ndimage

import time


class CellAnalyzer(QWidget):
    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__()
        self.viewer = viewer
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(
            logging.DEBUG
        )  # TODO: change to ERROR for release
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

        self.layer_to_evaluate: Labels = (
            None  # label layer with remaining and included cells
        )
        self.accepted_cells: (
            np.ndarray
        )  # label layer esque for all accepted cells
        self.rejected_cells: np.ndarray = (
            None  # label layer esque for all rejected cells
        )
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
        self.undo_stack: List[int] = []  # stack of cell ids to undo

        self.next_id: int = None  # computed id of the next cell to evaluate

        self.selfdrawn_lower_bound: int = (
            None  # lower bound of self drawn cell id
        )

        self.initialize_ui()

        # Hotkeys

        hotkeys = self.viewer.keymap.keys()
        self.logger.debug(f"Current hotkeys: {hotkeys}")
        custom_binds = [
            ("K", self.on_hotkey_include),
            ("G", self.on_hotkey_exclude),
            ("H", self.on_hotkey_undo),
            ("J", self.toggle_visibility_label_layers_hotkey),
        ]
        for custom_bind in custom_binds:
            napari_ver = napari.__version__.split(".")
            if int(napari_ver[0]) == 0 and int(napari_ver[1]) < 5:
                condition = custom_bind[0] in hotkeys
            else:
                custom_bind_keys = [bind.to_text() for bind in self.viewer.keymap]
                condition = custom_bind[0] in custom_bind_keys
            if not condition:
                viewer.bind_key(*custom_bind)

        self.viewer.layers.events.inserted.connect(self.get_label_layer)
        for layer in self.viewer.layers:
            if isinstance(layer, Labels):
                self.set_label_layer(layer)
                break

        self.viewer.layers.events.removed.connect(self.slot_layer_deleted)
        self.installEventFilter(self)

        self.logger.debug(f"CellAnalyzer v{version} initialized")
        self.logger.info("Ready to use")

    def slot_layer_deleted(self, event):
        def readd_layer(data, name):
            self.logger.debug("Important layer removed")
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            msg.setText("Please don't remove this layer, we need it.")
            msg.exec_()
            return self.viewer.add_labels(data, name=name)

        self.logger.debug("Layer deleted")
        if event.value in [self.current_cell_layer, self.layer_to_evaluate]:
            layer = readd_layer(event.value.data, event.value.name)
            if event.value.name == self.layer_to_evaluate.name:
                self.layer_to_evaluate = layer
            else:
                self.current_cell_layer = layer

    def eventFilter(self, source, event):
        if event.type() == QEvent.Hide:
            self.logger.debug("Disconnecting slots...")
            self.viewer.layers.events.inserted.disconnect(self.get_label_layer)
            self.viewer.layers.events.removed.disconnect(
                self.slot_layer_deleted
            )
        return super().eventFilter(source, event)

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
        starttime = time.time()

        ### QObjects
        # objects that can be updated are attributes of the class
        # for ease of access

        # Labels
        title = QLabel("<h1>MMV_H4Cells</h1>")
        self.label_next_id = QLabel("Start analysis at:")
        label_include = QLabel("Include:")
        label_included = QLabel("Included:")
        label_excluded = QLabel("Excluded:")
        label_remaining = QLabel("Remaining:")
        label_mean = QLabel("Mean size [px]:")
        label_std = QLabel("Std size [px]:")
        # label_metric = QLabel("Metric name:")
        # label_conversion = QLabel("1 pixel equals:")
        self.label_amount_included = QLabel("0")
        self.label_amount_excluded = QLabel("0")
        self.label_amount_remaining = QLabel("0")
        self.label_mean_included = QLabel("0")
        self.label_std_included = QLabel("0")
        # self.label_metric_included = QLabel("0")
        label_range_x = QLabel("Range x:")
        label_range_x.setToolTip(
            "The range of x values (left to right) to be included in the analysis.\n"
            + "First value must be lower than second. First value must be at least 0.\n"
            + "First value can be -1 to evaluate everything right of the first value."
        )
        label_range_y = QLabel("Range y:")
        label_range_y.setToolTip(
            "The range of y values (top to bottom) to be included in the analysis.\n"
            + "First value must be lower than second. First value must be at least 0.\n"
            + "First value can be -1 to evaluate everything below the first value."
        )
        label_threshold_size = QLabel("Threshold size:")

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
        self.btn_cancel = QPushButton("Cancel")
        self.btn_show_included = QPushButton("Show Included")
        self.btn_show_excluded = QPushButton("Show Excluded")
        self.btn_show_remaining = QPushButton("Show Remaining")
        self.btn_segment = QPushButton("Draw own cell")
        self.btn_include_multiple = QPushButton("Include multiple")
        self.btn_export_roi = QPushButton("Export ROI")

        self.btn_start_analysis.clicked.connect(self.start_analysis_on_click)
        self.btn_export.clicked.connect(self.export_on_click)
        self.btn_import.clicked.connect(self.import_on_click)
        self.btn_include.clicked.connect(self.include_on_click)
        self.btn_exclude.clicked.connect(self.exclude_on_click)
        self.btn_undo.clicked.connect(self.undo_on_click)
        self.btn_cancel.clicked.connect(self.cancel_on_click)
        self.btn_show_included.clicked.connect(self.show_included_on_click)
        self.btn_show_excluded.clicked.connect(self.show_excluded_on_click)
        self.btn_show_remaining.clicked.connect(self.show_remaining_on_click)
        self.btn_segment.clicked.connect(self.draw_own_cell)
        self.btn_include_multiple.clicked.connect(
            self.include_multiple_on_click
        )
        self.btn_export_roi.clicked.connect(self.export_roi_on_click)

        self.btn_export.setToolTip(
            "Export mask of included cells and analysis csv"
        )
        self.btn_import.setToolTip(
            "Import previously exported mask and analysis csv to continue analysis"
        )
        self.btn_include.setToolTip(
            'Include checked cell. Instead of clicking this button, you can also press the "K" key.'
        )
        self.btn_exclude.setToolTip(
            'Exclude checked cell. Instead of clicking this button, you can also press the "G" key.'
        )
        self.btn_undo.setToolTip(
            'Undo last selection. Instead of clicking this button, you can also press the "H" key.'
        )

        self.btn_start_analysis.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_include.setEnabled(False)
        self.btn_exclude.setEnabled(False)
        self.btn_undo.setEnabled(False)
        self.btn_cancel.setVisible(False)
        self.btn_show_included.setEnabled(False)
        self.btn_show_excluded.setEnabled(False)
        self.btn_show_remaining.setEnabled(False)
        self.btn_segment.setEnabled(False)
        self.btn_include_multiple.setEnabled(False)

        # LineEdits
        self.lineedit_next_id = QLineEdit()
        # self.lineedit_conversion_rate = QLineEdit()
        # self.lineedit_conversion_rate.returnPressed.connect(self.update_labels)
        self.lineedit_include = QLineEdit()
        self.lineedit_x_low = QLineEdit()
        self.lineedit_x_low.setObjectName("x_low")
        self.lineedit_x_high = QLineEdit()
        self.lineedit_x_high.setObjectName("x_high")
        self.lineedit_y_low = QLineEdit()
        self.lineedit_y_low.setObjectName("y_low")
        self.lineedit_y_high = QLineEdit()
        self.lineedit_y_high.setObjectName("y_high")
        self.lineedit_threshold_size = QLineEdit()
        self.lineedit_threshold_size.setPlaceholderText("0")
        self.lineedit_threshold_size.setToolTip(
            "The ROI may split cells at the edge, this threshold allows cells with fewer pixels to be excluded"
        )

        # Comboboxes
        # self.combobox_conversion_unit = QComboBox()

        # self.combobox_conversion_unit.addItems(["mm", "µm", "nm"])

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

        line3 = QWidget()
        line3.setFixedHeight(4)
        line3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line3.setStyleSheet("background-color: #c0c0c0")

        # QGroupBoxes
        groupbox_roi = QGroupBox("ROI Analysis")
        groupbox_roi.setStyleSheet(
            """
            QGroupBox {
                border: 1px solid silver;
                margin-top: 2ex; /* leave space at the top for the title */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* position at the top center */
                padding: 0 3px;
            }
            """
        )
        groupbox_roi.setLayout(QGridLayout())
        groupbox_roi.layout().addWidget(label_range_y, 0, 0, 1, 1)
        groupbox_roi.layout().addWidget(self.lineedit_y_low, 0, 1, 1, 1)
        groupbox_roi.layout().addWidget(QLabel("-"), 0, 2, 1, 1)
        groupbox_roi.layout().addWidget(self.lineedit_y_high, 0, 3, 1, 1)

        groupbox_roi.layout().addWidget(label_range_x, 1, 0, 1, 1)
        groupbox_roi.layout().addWidget(self.lineedit_x_low, 1, 1, 1, 1)
        groupbox_roi.layout().addWidget(QLabel("-"), 1, 2, 1, 1)
        groupbox_roi.layout().addWidget(self.lineedit_x_high, 1, 3, 1, 1)

        groupbox_roi.layout().addWidget(label_threshold_size, 2, 0, 1, 1)
        groupbox_roi.layout().addWidget(
            self.lineedit_threshold_size, 2, 1, 1, -1
        )

        groupbox_roi.layout().addWidget(self.btn_export_roi, 3, 0, 1, -1)

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
        content.layout().addWidget(self.btn_cancel, 5, 1, 1, 1)
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

        content.layout().addWidget(label_include, 12, 0, 1, 1)
        content.layout().addWidget(self.lineedit_include, 12, 1, 1, 1)
        content.layout().addWidget(self.btn_include_multiple, 12, 2, 1, 1)

        # content.layout().addWidget(label_conversion, 13, 0, 1, 1)
        # content.layout().addWidget(self.lineedit_conversion_rate, 13, 1, 1, 1)
        # content.layout().addWidget(self.combobox_conversion_unit, 13, 2, 1, 1)

        content.layout().addWidget(self.btn_show_included, 14, 0, 1, 1)
        content.layout().addWidget(self.btn_show_excluded, 14, 1, 1, 1)
        content.layout().addWidget(self.btn_show_remaining, 14, 2, 1, 1)

        content.layout().addWidget(line3, 15, 0, 1, -1)

        content.layout().addWidget(groupbox_roi, 16, 0, 1, -1)

        scroll_area = QScrollArea()
        scroll_area.setWidget(content)
        scroll_area.setWidgetResizable(True)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)

        endtime = time.time()
        self.logger.debug(f"Runtime UI initialization: {endtime - starttime}")

    def get_label_layer(self, event):
        self.logger.debug("New potential label layer detected...")
        if not self.layer_to_evaluate is None:
            self.logger.debug("Label layer already set")
            return
        if not isinstance(event.value, Labels):
            self.logger.debug("New layer invalid")
            return
        self.logger.debug("Label layer is valid")
        self.set_label_layer(event.value)

    def set_label_layer(self, layer):
        self.logger.debug("Setting label layer...")
        starttime = time.time()
        self.layer_to_evaluate = layer
        self.btn_start_analysis.setEnabled(True)
        unique_ids = np.unique(self.layer_to_evaluate.data)
        self.logger.debug(f"{len(unique_ids)} unique ids found")
        if self.selfdrawn_lower_bound is None:
            self.selfdrawn_lower_bound = max(unique_ids) + 1

        if len(self.metric_data) == 0:
            self.accepted_cells = np.zeros_like(self.layer_to_evaluate.data)
            self.rejected_cells = np.zeros_like(self.layer_to_evaluate.data)
            self.remaining = set(unique_ids) - {0}
        next_id = str(min(self.remaining)) if len(self.remaining) > 0 else ""
        self.lineedit_next_id.setText(next_id)
        self.next_id = next_id if next_id != "" else None
        self.logger.debug(
            f"Selfdrawn lower bound: {self.selfdrawn_lower_bound}"
        )
        self.logger.debug("Sets updated")
        self.update_labels()
        endtime = time.time()
        self.logger.debug(f"Runtime set label layer: {endtime - starttime}")

    def update_labels(self):
        self.logger.debug("Updating labels...")
        self.label_amount_excluded.setText(str(len(self.excluded)))
        self.label_amount_included.setText(str(len(self.included)))
        self.label_amount_remaining.setText(str(len(self.remaining)))
        self.label_mean_included.setText(str(self.mean_size))
        self.label_std_included.setText(str(self.std_size))

    def start_analysis_on_click(self):
        self.logger.debug("Analysis started...")
        label_layers = [
            layer.name for layer in self.viewer.layers if isinstance(layer, Labels)
        ]
        if len(label_layers) > 1:
            dialog = ChoiceDialog(label_layers, self.layer_to_evaluate.name)
            choice = dialog.exec_()
            if choice >= 0:
                layer = self.viewer.layers[label_layers[choice]]
                self.set_label_layer(layer)
        self.logger.debug(f"Using label layer: {self.layer_to_evaluate.name}")
        starttime = time.time()
        try:
            start_id = int(self.lineedit_next_id.text())
        except ValueError:
            self.logger.warning("Invalid start id")
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            msg.setText("Invalid start id")
            msg.exec_()
            return
        if len(self.remaining) < 1:
            self.logger.info("No cell to evaluate")
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            msg.setText("No cells to evaluate")
            msg.exec_()
            return
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
        if not start_id in self.remaining:
            self.logger.warning("Start id not in remaining ids")
            lower_ids = {value for value in self.remaining if value < start_id}
            if len(lower_ids) > 0:
                self.logger.info("Using lower id")
                start_id = max(lower_ids)
            else:
                self.logger.info("Using lowest remaining id")
                start_id = min(self.remaining)
        self.next_id = min(x for x in self.remaining if x > start_id)
        self.lineedit_next_id.setText(
            str(self.next_id) if self.next_id is not None else ""
        )
        # start iterating through ids to create label layer for and zoom into centroid of label
        self.display_cell(start_id)
        endtime = time.time()
        self.logger.debug(f"Runtime start analysis: {endtime - starttime}")

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
        self.logger.debug(f"Centroid: {centroid}")
        self.viewer.camera.zoom = 7.5  # !!
        self.current_cell_layer.selected_label = cell_id

    def import_on_click(self):
        self.logger.debug("Importing data...")
        csv_filepath = Path(open_dialog(self))
        if str(csv_filepath) == ".":
            self.logger.debug("No csv file selected. Aborting.")
            return
        zarr_filepath = csv_filepath.with_suffix(".zarr")
        try:
            data_to_evaluate, accepted_cells, rejected_cells, data, metrics, undo_stack, self.selfdrawn_lower_bound = read(
                zarr_filepath
            )
        except FileNotFoundError:
            zarr_filepath = Path(open_dialog(self), dir=True)
            if str(zarr_filepath) == ".":
                self.logger.debug("No zarr file selected. Aborting.")
                return
            data_to_evaluate, accepted_cells, rejected_cells, data, metrics, undo_stack, self.selfdrawn_lower_bound = read(
                zarr_filepath
            )
        self.logger.debug(f"Minimum id: {min(pd.unique(data_to_evaluate.flatten()))}, Maximum id: {max(pd.unique(data_to_evaluate.flatten()))}")
        self.accepted_cells = accepted_cells
        self.rejected_cells = rejected_cells
        self.mean_size, self.std_size = metrics  # , self.metric_value = ...
        self.undo_stack = undo_stack.tolist()
        self.included = set(pd.unique(self.accepted_cells.flatten())) - {0}
        self.excluded = set(pd.unique(self.rejected_cells.flatten())) - {0}
        self.btn_export.setEnabled(True)
        self.metric_data = data

        layer = self.viewer.add_labels(data_to_evaluate, name="Imported Data")
        self.set_label_layer(layer)

        self.logger.debug("Filling in values for imported data")
        unique_ids = pd.unique(self.layer_to_evaluate.data.flatten())
        self.remaining = set(unique_ids) - (
            self.included | self.excluded | {0}
        )
        next_id = (
            str(min(self.remaining)) if len(self.remaining) > 0 else ""
        )
        self.lineedit_next_id.setText(next_id)
        self.btn_start_analysis.setEnabled(True)

        self.update_labels()

    def export_on_click(self):
        self.logger.debug("Exporting data...")
        csv_filepath = Path(save_dialog(self))
        if csv_filepath.name == ".csv":
            self.logger.debug("No file selected. Aborting.")
            return
        zarr_filepath = csv_filepath.with_suffix(".zarr")
        tiff_filepath = csv_filepath.with_suffix(".tiff")
        self.metric_data = sorted(self.metric_data, key=lambda x: x[0])
        write(
            csv_filepath,
            self.metric_data,
            (self.mean_size, self.std_size, 0),
        )
        self.logger.debug("Metrics written to csv")
        write(tiff_filepath, self.accepted_cells)
        self.logger.debug("Accepted cells written to tiff")
        write(
            zarr_filepath,
            self.layer_to_evaluate.data,
            self.accepted_cells,
            self.rejected_cells,
            self.metric_data,
            (self.mean_size, self.std_size),
            self.undo_stack,
            self.selfdrawn_lower_bound,
        )
        self.logger.debug("Data written to zarr")

    def include_on_click(self, self_drawn=False):
        """
        Includes the current cell in the analysis.

        Parameters
        ----------
        self_drawn : bool, optional
            Whether the cell was drawn by the user, by default False

        Returns
        -------
        bool
            Whether the cell was included successfully"""
        self.logger.debug("Including cell...")
        starttime_abs = time.time()
        if len(self.remaining) < 1:
            self.logger.info("No cell to include")
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            msg.setText("No remaining cells")
            msg.exec_()
            return False

        if self.check_for_overlap():
            return False

        endtime = time.time()
        self.logger.debug(f"Runtime overlap check: {endtime - starttime_abs}")
        starttime = time.time()
        if len(pd.unique(self.current_cell_layer.data.flatten())) > 2:
            self.logger.debug("Multiple ids in current cell layer")
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            msg.setText(
                "More than one label detected in current cell layer. Please remove the added label."
            )
            msg.exec_()
            return False
        endtime = time.time()
        self.logger.debug(f"Runtime multiple ids check: {endtime - starttime}")
        starttime = time.time()
        id_ = int(np.max(self.current_cell_layer.data))
        self.include(id_, self.current_cell_layer.data, not self_drawn)
        if self_drawn:
            self.layer_to_evaluate.data += self.current_cell_layer.data
        endtime = time.time()
        self.logger.debug(f"Runtime include: {endtime - starttime}")

        self.undo_stack.append(id_)

        if len(self.remaining) > 0:
            starttime = time.time()
            self.display_next_cell()
            endtime = time.time()
            self.logger.debug(
                f"Runtime display next cell: {endtime - starttime}"
            )
        endtime_abs = time.time()
        self.logger.debug(f"Runtime complete: {endtime_abs - starttime_abs}")
        return True

    def exclude_on_click(self):
        self.logger.debug("Excluding cell...")
        if len(self.remaining) < 1:
            self.logger.info("No cell to exclude")
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            msg.setText("No remaining cells")
            msg.exec_()
            return

        unique_ids = pd.unique(self.current_cell_layer.data.flatten())
        if len(unique_ids) > 2:
            self.logger.debug("Multiple ids in current cell layer")
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            msg.setText(
                "More than one label detected in current cell layer. Please remove the added label."
            )
            msg.exec_()
            return

        current_id = int(max(unique_ids))
        self.excluded.add(current_id)
        self.remaining.remove(current_id)
        self.undo_stack.append(current_id)

        mask = np.where(self.current_cell_layer.data == current_id)
        self.rejected_cells[mask] = current_id
        self.layer_to_evaluate.data[mask] = 0
        self.layer_to_evaluate.refresh()

        self.update_labels()

        if len(self.remaining) > 0:
            self.display_next_cell()

    def undo_on_click(self):
        self.logger.debug("Undoing last action...")
        if len(self.undo_stack) == 0:
            self.logger.info("No actions to undo")
            return
        self.logger.debug("Before undo:")
        self.logger.debug(f"Last evaluated: {self.undo_stack[-1]}")
        last_evaluated = self.undo_stack.pop(-1)
        if last_evaluated < self.selfdrawn_lower_bound:
            self.logger.debug("Adding cell back to remaining")
            self.remaining.add(last_evaluated)
        if last_evaluated in self.accepted_cells:
            self.logger.debug("Removing cell from accepted")
            self.metric_data.pop(-1)
            indices = np.where(self.accepted_cells == last_evaluated)
            self.accepted_cells[indices] = 0
            self.included.remove(last_evaluated)
            if last_evaluated >= self.selfdrawn_lower_bound:
                mask = np.where(self.layer_to_evaluate.data == last_evaluated)
                self.layer_to_evaluate.data[mask] = 0
                self.layer_to_evaluate.refresh()
        else:
            self.excluded.remove(last_evaluated)
            mask = np.where(self.rejected_cells == last_evaluated)
            self.layer_to_evaluate.data[mask] = last_evaluated
            self.rejected_cells[mask] = 0
        self.lineedit_next_id.setText(str(last_evaluated))

        self.calculate_metrics()
        self.update_labels()
        if last_evaluated < self.selfdrawn_lower_bound:
            self.display_next_cell(True)
        else:
            self.redisplay_current_cell()

    def cancel_on_click(self):
        self.logger.debug("Cancelling draw own cell...")
        self.btn_include.setEnabled(True)
        self.btn_exclude.setEnabled(True)
        self.btn_undo.setVisible(True)
        self.btn_cancel.setVisible(False)
        self.current_cell_layer.mode = "pan_zoom"
        self.btn_segment.setText("Draw own cell")
        self.display_next_cell()

    def show_included_on_click(self):
        if self.btn_show_included.text() == "Show Included":
            self.logger.debug("Showing included cells...")
            self.btn_include.setEnabled(False)
            self.btn_exclude.setEnabled(False)
            self.btn_undo.setEnabled(False)
            self.btn_show_included.setText("Back")
            self.set_visitibility_label_layers(False)
            if self.excluded_layer is not None:
                self.viewer.layers.remove(self.excluded_layer)
                self.excluded_layer = None
                self.btn_show_excluded.setText("Show Excluded")
            if self.remaining_layer is not None:
                self.viewer.layers.remove(self.remaining_layer)
                self.remaining_layer = None
                self.btn_show_remaining.setText("Show Remaining")
            self.included_layer = self.viewer.add_labels(
                self.accepted_cells, name="Included Cells"
            )
            self.viewer.camera.zoom = 1
        else:
            self.logger.debug("Hiding included cells...")
            self.viewer.layers.remove(self.included_layer)
            self.included_layer = None
            self.set_visitibility_label_layers(True)
            self.btn_show_included.setText("Show Included")
            self.viewer.layers.selection.active = self.current_cell_layer
            centroid = ndimage.center_of_mass(
                self.current_cell_layer.data,
                labels=self.current_cell_layer.data,
                index=self.current_cell_layer.selected_label,
            )
            self.viewer.camera.center = centroid
            self.viewer.camera.zoom = 7.5
            self.btn_include.setEnabled(True)
            self.btn_exclude.setEnabled(True)
            self.btn_undo.setEnabled(True)

    def show_excluded_on_click(self):
        if self.btn_show_excluded.text() == "Show Excluded":
            self.logger.debug("Showing excluded cells...")
            self.btn_include.setEnabled(False)
            self.btn_exclude.setEnabled(False)
            self.btn_undo.setEnabled(False)
            self.btn_show_excluded.setText("Back")
            self.set_visitibility_label_layers(False)
            if self.included_layer is not None:
                self.viewer.layers.remove(self.included_layer)
                self.included_layer = None
                self.btn_show_included.setText("Show Included")
            if self.remaining_layer is not None:
                self.viewer.layers.remove(self.remaining_layer)
                self.remaining_layer = None
                self.btn_show_remaining.setText("Show Remaining")
            self.excluded_layer = self.viewer.add_labels(
                self.rejected_cells, name="Excluded Cells"
            )
            self.viewer.camera.zoom = 1
        else:
            self.logger.debug("Hiding excluded cells...")
            self.viewer.layers.remove(self.excluded_layer)
            self.excluded_layer = None
            self.set_visitibility_label_layers(True)
            self.btn_show_excluded.setText("Show Excluded")
            self.viewer.layers.selection.active = self.current_cell_layer
            centroid = ndimage.center_of_mass(
                self.current_cell_layer.data,
                labels=self.current_cell_layer.data,
                index=self.current_cell_layer.selected_label,
            )
            self.viewer.camera.center = centroid
            self.viewer.camera.zoom = 7.5
            self.btn_include.setEnabled(True)
            self.btn_exclude.setEnabled(True)
            self.btn_undo.setEnabled(True)

    def show_remaining_on_click(self):
        if self.btn_show_remaining.text() == "Show Remaining":
            self.logger.debug("Showing remaining cells...")
            self.btn_include.setEnabled(False)
            self.btn_exclude.setEnabled(False)
            self.btn_undo.setEnabled(False)
            self.btn_show_remaining.setText("Back")
            self.set_visitibility_label_layers(False)
            if self.included_layer is not None:
                self.viewer.layers.remove(self.included_layer)
                self.included_layer = None
                self.btn_show_included.setText("Show Included")
            if self.excluded_layer is not None:
                self.viewer.layers.remove(self.excluded_layer)
                self.excluded_layer = None
                self.btn_show_excluded.setText("Show Excluded")
            data = copy.deepcopy(self.layer_to_evaluate.data)
            mask = np.isin(data, list(self.remaining | {0}), invert=True)
            data[mask] = 0
            self.remaining_layer = self.viewer.add_labels(
                data, name="Remaining Cells"
            )
            self.viewer.camera.zoom = 1
        else:
            self.logger.debug("Hiding remaining cells...")
            self.viewer.layers.remove(self.remaining_layer)
            self.remaining_layer = None
            self.set_visitibility_label_layers(True)
            self.btn_show_remaining.setText("Show Remaining")
            self.viewer.layers.selection.active = self.current_cell_layer
            centroid = ndimage.center_of_mass(
                self.current_cell_layer.data,
                labels=self.current_cell_layer.data,
                index=self.current_cell_layer.selected_label,
            )
            self.viewer.camera.center = centroid
            self.viewer.camera.zoom = 7.5
            self.btn_include.setEnabled(True)
            self.btn_exclude.setEnabled(True)
            self.btn_undo.setEnabled(True)

    def include_multiple_on_click(self):
        self.logger.debug(
            f"Including multiple cells for input {self.lineedit_include.text()}"
        )
        given_ids = self.get_ids_to_include()
        if given_ids is None:
            self.logger.debug("No valid ids in input")
            return
        self.logger.debug(f"Given ids: {given_ids}")
        included, ignored, overlapped, faulty = self.include_multiple(
            given_ids
        )
        self.lineedit_include.setText("")
        next_id = str(min(self.remaining)) if len(self.remaining) > 0 else ""
        self.lineedit_next_id.setText(next_id)
        self.display_next_cell(True)
        msg = QMessageBox()
        msg.setWindowTitle("napari")
        msgtext = ""
        if len(included) > 0:
            msgtext += f"Cells included: {included}\n"
        if len(ignored) > 0:
            msgtext += (
                f"Cells ignored as they are already evaluated: {ignored}\n"
            )
            msgtext += "Only unprocessed cells can be included.\n"
        if len(faulty) > 0:
            msgtext += f"Cells ignored due to nonexistance: {faulty}\n"
            msgtext += "Please only enter existing cell ids.\n"
        if len(overlapped) > 0:
            msgtext += f"Cells not included due to overlap: {overlapped}\n"
            msgtext += "Please remove the overlap(s)."
        if msgtext == "":
            msgtext = "0 is not a valid cell id."
        msg.setText(msgtext)
        msg.exec_()

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
            msg.setText(
                "Please enter a comma separated list of integers, for example: 1, 5, 2, 8, 199, 5000."
            )
            msg.exec_()
            return None
        return ids

    def include_multiple(
        self, ids: List[int]
    ) -> Tuple[Set[int], Set[int], Set[int]]:
        self.logger.debug("Including multiple cells...")
        included = set()
        ignored = set()
        overlapped = set()
        faulty = set()
        existing_ids = set(pd.unique(self.layer_to_evaluate.data.flatten()))
        for val in ids:
            if val == 0:
                continue
            if val not in existing_ids:
                faulty.add(val)
                continue
            if val not in self.remaining:
                ignored.add(val)
                continue
            indices = np.where(self.layer_to_evaluate.data == val)
            if np.sum(self.accepted_cells[indices]):
                overlapped.add(val)
                continue
            data_array = np.zeros_like(self.layer_to_evaluate.data)
            data_array[indices] = val
            self.include(val, data_array)
            included.add(val)
            self.undo_stack.append(val)
        self.logger.debug("Multiple cells evaluated")
        return included, ignored, overlapped, faulty

    def draw_own_cell(self):
        if self.btn_segment.text() == "Draw own cell":
            self.logger.debug("Draw own cell initialized")
            self.btn_segment.setText("Confirm")
            self.btn_exclude.setEnabled(False)
            self.btn_include.setEnabled(False)
            self.btn_undo.setVisible(False)
            self.btn_cancel.setVisible(True)
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
                max(
                    np.max(self.accepted_cells),
                    np.max(self.rejected_cells),
                    np.max(self.layer_to_evaluate.data),
                )
                + 1
            )
        else:
            self.logger.debug("Draw own cell confirmed")
            self.current_cell_layer.mode = "pan_zoom"
            if len(pd.unique(self.current_cell_layer.data.flatten())) < 2:
                self.logger.debug("No label drawn")
                msg = QMessageBox()
                msg.setWindowTitle("napari")
                msg.setText("No cell annotated.")
                msg.exec_()
                return
            if self.include_on_click(True):
                self.btn_segment.setText("Draw own cell")
                self.btn_exclude.setEnabled(True)
                self.btn_include.setEnabled(True)
                self.btn_undo.setVisible(True)
                self.btn_cancel.setVisible(False)

    def display_next_cell(self, ignore_jump_back: bool = False):
        self.logger.debug("Displaying next cell...")
        if len(self.remaining) < 1:
            self.logger.debug("No cells left to evaluate")
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            msg.setText("No more cells to evaluate.")
            msg.exec_()
            return

        try:
            given_id = int(self.lineedit_next_id.text())
        except ValueError:
            given_id = None
        self.logger.debug(f"Id given by textfield: {given_id}")
        last_evaluated_id = (
            self.undo_stack[-1] if len(self.undo_stack) > 0 else 0
        )
        self.logger.debug(f"Last evaluated id: {last_evaluated_id}")
        next_lower = max(
            [i for i in self.remaining if i < given_id], default=None
        )
        self.logger.debug(f"Next lower id: {next_lower}")
        next_higher = min(
            [i for i in self.remaining if i > given_id], default=None
        )
        self.logger.debug(f"Next higher id: {next_higher}")
        computed_id = self.next_id
        self.logger.debug(f"Computed next id: {computed_id}")

        if given_id is None:
            # no valid id given
            next_id = computed_id
        elif given_id == computed_id:
            # id was not changed
            if computed_id < last_evaluated_id and not ignore_jump_back:
                # jump back to lowest unprocessed id
                msg = QMessageBox()
                msg.setWindowTitle("napari")
                self.logger.debug("No higher id remaining")
                msg.setText("Dataset is finished. Jumping to earlier cells.")
            next_id = computed_id
        elif given_id != computed_id and given_id in self.remaining:
            # id was changed and is in remaining
            next_id = given_id
        elif given_id > computed_id:
            # id was increased and is not in remaining
            next_id = next_lower
        else:
            # given_id < computed_id -> id was decreased and is not in remaining
            if given_id > last_evaluated_id:
                # maybe others < last < given < computed
                next_id = next_lower if next_lower is not None else computed_id
            elif computed_id < last_evaluated_id and not ignore_jump_back:
                # given < computed (smallest existing) < maybe others < last
                next_id = computed_id
            else:
                # others? < given < others? < computed < last (self drawn)
                # others? < given < others? < last < computed
                next_id = next_lower if next_lower is not None else next_higher

        if "msg" in locals():
            msg.exec_()
        self.display_cell(next_id)

        if len(self.remaining) > 1:
            candidate_ids = sorted([i for i in self.remaining if i > next_id])
            if len(candidate_ids) > 0:
                self.next_id = candidate_ids[0]
            else:
                self.next_id = min(self.remaining)
            self.lineedit_next_id.setText(str(self.next_id))
        else:
            self.lineedit_next_id.setText("")
            self.next_id = None
        self.logger.debug("Value for next cell set")

    def redisplay_current_cell(self):
        self.logger.debug("Redisplaying current cell...")
        id_ = self.current_cell_layer.selected_label
        self.display_cell(id_)

        if len(self.remaining) > 1:
            candidate_ids = sorted([i for i in self.remaining if i > id_])
            if len(candidate_ids) > 0:
                self.next_id = candidate_ids[0]
            else:
                self.next_id = min(self.remaining)
            self.lineedit_next_id.setText(str(self.next_id))
        else:
            self.lineedit_next_id.setText("")
            self.next_id = None
        self.logger.debug("Value for next cell set")

    def include(
        self,
        id_: int,
        data_array: np.ndarray,
        remove_from_remaining: bool = True,
    ):
        self.logger.debug("Including cell...")
        self.accepted_cells += data_array
        if remove_from_remaining:
            self.remaining.remove(id_)
        self.included.add(id_)

        self.add_cell_to_accepted(id_, data_array)

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
        eval_cells = np.copy(self.layer_to_evaluate.data)
        id_ = self.current_cell_layer.selected_label
        eval_cells[eval_cells == id_] = 0
        nonzero_eval = np.nonzero(eval_cells)
        combined_layer = np.zeros_like(self.layer_to_evaluate.data)
        combined_layer[nonzero_current] += 1
        combined_layer[nonzero_eval] += 1
        overlap = set(map(tuple, np.transpose(np.where(combined_layer == 2))))
        return overlap

    def add_cell_to_accepted(self, cell_id: int, data: np.ndarray):
        self.logger.debug("Adding cell to list of accepted...")
        self.included.add(cell_id)
        centroid = ndimage.center_of_mass(data)
        centroid = tuple(int(value) for value in centroid)
        self.metric_data.append(  # TODO
            (
                cell_id,
                np.count_nonzero(data),
                centroid,
            )
        )

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
            np.zeros_like(self.layer_to_evaluate.data),
            name="Overlap",
            opacity=1,
        )
        overlap_layer.data[overlap_indices] = (
            np.amax(self.current_cell_layer.data) + 1
        )
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

    def set_visitibility_label_layers(self, visible: bool):
        self.logger.debug(
            f"Setting visibility of label layers to {visible}..."
        )
        for layer in self.viewer.layers:
            if isinstance(layer, Labels):
                layer.visible = visible

    def toggle_visibility_label_layers(self):
        self.logger.debug("Toggling visibility of label layers...")
        for layer in self.viewer.layers:
            if isinstance(layer, Labels):
                layer.visible = not layer.visible

    def export_roi_on_click(self):
        self.logger.debug("Exporting ROI data...")
        try:
            lower_y, upper_y, lower_x, upper_x, threshold = (
                self.validate_roi_params()
            )
        except ValueError:
            return
        self.logger.debug("Valid ROI parameters")
        self.logger.debug(
            f"ROI parameters: {lower_y}, {upper_y}, {lower_x}, {upper_x}, {threshold}"
        )
        csv_filepath = Path(save_dialog(self, "(*.csv);; (*.tiff *.tif)"))
        if csv_filepath.name == ".csv":
            self.logger.debug("No file selected. Aborting.")
            return
        csv_filepath = csv_filepath.with_name(
            csv_filepath.stem + "_roi" + csv_filepath.suffix
        )
        tiff_filepath = csv_filepath.with_suffix(".tiff")
        worker = analyse_roi(
            self.layer_to_evaluate.data,
            (lower_y, upper_y),
            (lower_x, upper_x),
            threshold,
            (csv_filepath, tiff_filepath),
        )
        worker.returned.connect(self.call_export)
        worker.start()

    def validate_roi_params(self):
        self.logger.debug("Validating ROI parameters...")

        params = []
        for lineedit in [
            self.lineedit_y_low,
            self.lineedit_y_high,
            self.lineedit_x_low,
            self.lineedit_x_high,
            self.lineedit_threshold_size,
        ]:
            value = self.get_roi_param(lineedit)
            if (
                value is None
                or value < 0
                and not ("high" in lineedit.objectName() and value == -1)
            ):
                lineedit.setText("")
                params.append(None)
                continue
            min_ = 1 if "high" in lineedit.objectName() else 0
            max_ = (
                self.layer_to_evaluate.data.shape[0]
                if "y" in lineedit.objectName()
                else self.layer_to_evaluate.data.shape[1]
            )
            max_ -= 1 if "low" in lineedit.objectName() else 0
            if (
                value < min_ or value > max_ and lineedit.objectName() != ""
            ) and value != -1:
                lineedit.setText("")
                params.append(None)
                continue
            params.append(value)

        self.logger.debug(f"ROI parameters: {params}")
        if None in params:
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            if params[-1] is None:
                msg_text = "Threshold must be a positive integer."
            else:
                msg_text = "All values for Range y and Range x must be set."
            msg.setText(msg_text)
            msg.exec_()
            raise ValueError("Invalid ROI parameters.")

        return params

    def get_roi_param(self, lineedit):
        try:
            value = int(lineedit.text())
        except ValueError:
            if lineedit.objectName() == "":
                value = 0
            else:
                value = None
        return value

    def call_export(self, params):
        self.logger.debug("Exporting ROI data...")
        image, df, paths, threshold = params
        csv_filepath, tiff_filepath = paths
        data = df.itertuples(index=False)
        metrics = (
            np.round(df["count [px]"].mean(), 3),
            np.round(df["count [px]"].std(), 3),
            threshold,
        )
        # if self.lineedit_conversion_rate.text() == "":
        #     factor = 1
        #     unit = "pixel"
        # else:
        #     factor = float(
        #         self.lineedit_conversion_rate.text()
        #     )  # TODO: catch ValueError if not float
        #     unit = self.combobox_conversion_unit.currentText()
        # pixelsize = (factor, unit)
        # undo_stack = df["id"].tolist()
        write(csv_filepath, data, metrics)
        # write(csv_filepath, data, metrics, pixelsize, set(), undo_stack)
        write(tiff_filepath, image)
        self.logger.debug("ROI data exported.")
        msg = QMessageBox()
        msg.setWindowTitle("napari")
        msg.setText("ROI data exported.")
        msg.exec_()


class ChoiceDialog(QDialog):
    def __init__(self, layernames: List[str], selected: str):
        super().__init__()
        self.setWindowTitle("Choose Label Layer")
        self.combobox = QComboBox()
        self.combobox.addItems(layernames)
        self.combobox.setCurrentText(selected)
        btn_select = QPushButton("Select")
        btn_select.clicked.connect(self.accept)
        layout = QVBoxLayout()
        layout.addWidget(self.combobox)
        layout.addWidget(btn_select)
        self.setLayout(layout)
        self.setMinimumSize(250, 100)

    def accept(self):
        self.done(self.combobox.currentIndex())

    def reject(self):
        self.done(-1)
