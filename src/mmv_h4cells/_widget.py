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
)

import napari
import numpy as np
import pandas as pd
from typing import List, Tuple, Set
from pathlib import Path
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
        self.undo_stack: List[int] = []  # stack of cell ids to undo

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
            'Include checked cell. Instead of clicking this button, you can also press the "J" key.'
        )
        self.btn_exclude.setToolTip(
            'Undo last selection. Instead of clicking this button, you can also press the "F" key.'
        )
        self.btn_undo.setToolTip(
            'Exclude checked cell. Instead of clicking this button, you can also press the "B" key.'
        )

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
        groupbox_roi.setStyleSheet("""
            QGroupBox {
                border: 1px solid silver;
                margin-top: 2ex; /* leave space at the top for the title */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* position at the top center */
                padding: 0 3px;
            }
            """)
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

    def get_label_layer(self, event):
        self.logger.debug("New potential label layer detected...")
        if not (
            self.layer_to_evaluate is None and isinstance(event.value, Labels)
        ):
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
            self.remaining = set(unique_ids) - (self.included | self.excluded | {0})
        else:
            max_id = 0
            self.accepted_cells = np.zeros_like(self.layer_to_evaluate.data)
            self.remaining = set(unique_ids) - {0}
        next_id = str(min(self.remaining)) if len(self.remaining) > 0 else ""
        self.lineedit_next_id.setText(next_id)
        self.logger.debug("Sets updated")
        self.update_labels()

    def update_labels(self):
        self.logger.debug("Updating labels...")
        self.label_amount_excluded.setText(str(len(self.excluded)))
        self.label_amount_included.setText(str(len(self.included)))
        self.label_amount_remaining.setText(str(len(self.remaining)))
        # if self.lineedit_conversion_rate.text().strip() == "":
        #     unit = "pixel"
        #     factor = 1
        # else:
        #     unit = self.combobox_conversion_unit.currentText() + "²"
        #     factor = float(self.lineedit_conversion_rate.text())
        # self.logger.debug(f"Conversion rate: {factor} {unit}")
        self.label_mean_included.setText(str(self.mean_size))
        self.label_std_included.setText(str(self.std_size))
        # self.label_mean_included.setText(
        #     f"{str(self.mean_size*factor)} {unit}"
        # )
        # self.label_std_included.setText(f"{str(self.std_size*factor)} {unit}")
        # self.label_metric.included.setText(new value)

    def start_analysis_on_click(self):
        self.logger.debug("Analysis started...")
        try:
            start_id = int(
                self.lineedit_next_id.text()
            )  
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
        self.lineedit_next_id.setText(
            str(min(x for x in self.remaining if x > start_id))
        )
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
            self.logger.debug("No csv file selected. Aborting.")
            return
        tiff_filepath = csv_filepath.with_suffix(".tiff")
        try:
            accepted_cells = read(tiff_filepath)
        except FileNotFoundError:
            tiff_filepath = tiff_filepath.with_suffix(".tif")
            try:
                accepted_cells = read(tiff_filepath)
            except FileNotFoundError:
                tiff_filepath = Path(
                    open_dialog(self, filetype="*.tiff *.tif")
                )
                if str(tiff_filepath) == ".":
                    self.logger.debug("No tiff file selected. Aborting.")
                    return
                accepted_cells = read(tiff_filepath)
        self.accepted_cells = accepted_cells
        (
            self.metric_data,
            metrics,
            # pixelsize,
            self.excluded,
            self.undo_stack,
        ) = read(csv_filepath)
        self.mean_size, self.std_size = metrics  # , self.metric_value = ...
        # (
        #     self.lineedit_conversion_rate.setText(str(pixelsize[0]))
        #     if pixelsize[1] != "pixel"
        #     else self.lineedit_conversion_rate.setText("")
        # )
        # self.combobox_conversion_unit.setCurrentText(pixelsize[1])
        self.included = set(pd.unique(self.accepted_cells.flatten())) - {0}
        self.btn_export.setEnabled(True)

        if not self.layer_to_evaluate is None:
            self.logger.debug("Filling in values for existing label layer")
            unique_ids = pd.unique(self.layer_to_evaluate.data.flatten())
            self.remaining = set(unique_ids) - (
                self.included | self.excluded | {0}
            )
            next_id = str(min(self.remaining)) if len(self.remaining) > 0 else ""
            self.lineedit_next_id.setText(next_id)
            self.btn_start_analysis.setEnabled(True)

        self.update_labels()

    def export_on_click(self):
        self.logger.debug("Exporting data...")
        csv_filepath = Path(save_dialog(self))
        if csv_filepath.name == ".csv":
            self.logger.debug("No file selected. Aborting.")
            return
        tiff_filepath = csv_filepath.with_suffix(".tiff")
        # if self.lineedit_conversion_rate.text() == "":
        #     factor = 1
        #     unit = "pixel"
        # else:
        #     factor = float(
        #         self.lineedit_conversion_rate.text()
        #     )  # TODO: catch ValueError if not float
        #     unit = self.combobox_conversion_unit.currentText()
        self.metric_data = sorted(self.metric_data, key=lambda x: x[0])
        write(
            csv_filepath,
            self.metric_data,
            (self.mean_size, self.std_size, 0),
            # (
            #     factor,
            #     unit,
            #     # self.combobox_conversion_unit.currentText(),
            # ),
            self.excluded,
            self.undo_stack,
        )
        self.logger.debug("Metrics written to csv")
        write(tiff_filepath, self.accepted_cells)
        self.logger.debug("Accepted cells written to tiff")

    def include_on_click(self, self_drawn=False):
        self.logger.debug("Including cell...")
        starttime_abs = time.time()
        if len(self.remaining) < 1:
            self.logger.info("No cell to include")
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            msg.setText("No remaining cells")
            msg.exec_()
            return

        if self.check_for_overlap():
            return

        endtime = time.time()
        self.logger.debug(f"Runtime overlap check: {endtime - starttime_abs}")
        starttime = time.time()
        id_ = int(np.max(self.current_cell_layer.data))
        self.include(id_, self.current_cell_layer.data, not self_drawn)
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

    def exclude_on_click(self):
        self.logger.debug("Excluding cell...")
        if len(self.remaining) < 1:
            self.logger.info("No cell to exclude")
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            msg.setText("No remaining cells")
            msg.exec_()
            return
        current_id = int(
            max(pd.unique(self.current_cell_layer.data.flatten()))
        )
        self.excluded.add(current_id)
        self.remaining.remove(current_id)
        self.undo_stack.append(current_id)

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
        if last_evaluated in self.layer_to_evaluate.data:
            self.logger.debug("Adding cell back to remaining")
            self.remaining.add(last_evaluated)
        if last_evaluated in self.accepted_cells:
            self.logger.debug("Removing cell from accepted")
            self.metric_data.pop(-1)
            indices = np.where(self.accepted_cells == last_evaluated)
            self.accepted_cells[indices] = 0
            self.included.remove(last_evaluated)
        else:
            self.excluded.remove(last_evaluated)
        self.lineedit_next_id.setText(str(last_evaluated))

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
            self.viewer.camera.zoom = 1
        else:
            self.logger.debug("Hiding included cells...")
            self.viewer.layers.remove(self.included_layer)
            self.included_layer = None
            self.toggle_visibility_label_layers()
            self.btn_show_included.setText("Show Included")
            self.viewer.layers.selection.active = self.current_cell_layer
            centroid = ndimage.center_of_mass(
                self.current_cell_layer.data,
                labels=self.current_cell_layer.data,
                index=self.current_cell_layer.selected_label,
            )
            self.viewer.camera.center = centroid
            self.viewer.camera.zoom = 7.5

    def show_excluded_on_click(self):
        if self.btn_show_excluded.text() == "Show Excluded":
            self.logger.debug("Showing excluded cells...")
            self.btn_show_excluded.setText("Back")
            self.toggle_visibility_label_layers()
            data = copy.deepcopy(self.layer_to_evaluate.data)
            mask = np.isin(data, list(self.excluded | {0}), invert=True)
            data[mask] = 0
            self.excluded_layer = self.viewer.add_labels(
                data, name="Excluded Cells"
            )
            self.viewer.camera.zoom = 1
        else:
            self.logger.debug("Hiding excluded cells...")
            self.viewer.layers.remove(self.excluded_layer)
            self.excluded_layer = None
            self.toggle_visibility_label_layers()
            self.btn_show_excluded.setText("Show Excluded")
            self.viewer.layers.selection.active = self.current_cell_layer
            centroid = ndimage.center_of_mass(
                self.current_cell_layer.data,
                labels=self.current_cell_layer.data,
                index=self.current_cell_layer.selected_label,
            )
            self.viewer.camera.center = centroid
            self.viewer.camera.zoom = 7.5

    def show_remaining_on_click(self):
        if self.btn_show_remaining.text() == "Show Remaining":
            self.logger.debug("Showing remaining cells...")
            self.btn_show_remaining.setText("Back")
            self.toggle_visibility_label_layers()
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
            self.toggle_visibility_label_layers()
            self.btn_show_remaining.setText("Show Remaining")
            self.viewer.layers.selection.active = self.current_cell_layer
            centroid = ndimage.center_of_mass(
                self.current_cell_layer.data,
                labels=self.current_cell_layer.data,
                index=self.current_cell_layer.selected_label,
            )
            self.viewer.camera.center = centroid
            self.viewer.camera.zoom = 7.5

    def include_multiple_on_click(self):
        self.logger.debug(
            f"Including multiple cells for input {self.lineedit_include.text()}"
        )
        given_ids = self.get_ids_to_include()
        if given_ids is None:
            self.logger.debug("No valid ids in input")
            return
        self.logger.debug(f"Given ids: {given_ids}")
        included, ignored, overlapped, faulty = self.include_multiple(given_ids)
        self.lineedit_include.setText("")
        next_id = str(min(self.remaining)) if len(self.remaining) > 0 else ""
        self.lineedit_next_id.setText(next_id)
        self.display_next_cell(False)
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
            msg.setText("Please enter a comma separated list of integers.")
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
                max(
                    np.max(self.accepted_cells),
                    np.max(self.layer_to_evaluate.data),
                )
                + 1
            )
        else:
            self.logger.debug("Draw own cell confirmed")
            self.include_on_click(True)
            self.btn_segment.setText("Draw own cell")

    def display_next_cell(self, check_lowered=True):
        self.logger.debug("Displaying next cell...")
        if len(self.remaining) < 1:
            self.logger.debug("No cells left to evaluate")
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            msg.setText("No more cells to evaluate.")
            msg.exec_()
            return

        given_id = int(self.lineedit_next_id.text())
        self.logger.debug(f"Id given by textfield: {given_id}")
        last_evaluated_id = (
            self.undo_stack[-1] if len(self.undo_stack) > 0 else 0
        )
        self.logger.debug(f"Last evaluated id: {last_evaluated_id}")
        candidate_ids = sorted(
            [i for i in self.remaining if i > last_evaluated_id]
        )
        next_id_computed = (
            candidate_ids[0] if len(candidate_ids) > 0 else min(self.remaining)
        )
        self.logger.debug(f"Computed next id: {next_id_computed}")

        if given_id != next_id_computed:
            if given_id not in self.remaining:
                msg = QMessageBox()
                msg.setWindowTitle("napari")
                msg.setText("Given id is not in remaining cells.")
                msg.exec_()
                candidate_ids = sorted(
                    [i for i in self.remaining if i < given_id]
                )
                if len(candidate_ids) > 0:
                    self.logger.debug("Using highest lower id")
                    next_id = candidate_ids[-1]
                else:
                    self.logger.debug("Using lowest remaining id")
                    next_id = min(self.remaining)
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
                msg.setText("Dataset is finished. Jumping to earlier cells.")
            else:
                self.logger.debug("Higher id is remaining")
                msg.setText("Lowering the next cell id is a bad idea.")
            msg.exec_()
        self.display_cell(next_id)

        if len(self.remaining) > 1:
            candidate_ids = sorted([i for i in self.remaining if i > next_id])
            if len(candidate_ids) > 0:
                next_label = candidate_ids[0]
            else:
                next_label = min(self.remaining)
            self.lineedit_next_id.setText(str(next_label))
        else:
            self.lineedit_next_id.setText("")
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
        csv_filepath = csv_filepath.with_name(csv_filepath.stem + "_roi" + csv_filepath.suffix)
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
            if (value < min_ or value > max_ and lineedit.objectName() != "") and value != -1:
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
        metrics = (np.round(df["count [px]"].mean(), 3), np.round(df["count [px]"].std(), 3), threshold)
        # if self.lineedit_conversion_rate.text() == "":
        #     factor = 1
        #     unit = "pixel"
        # else:
        #     factor = float(
        #         self.lineedit_conversion_rate.text()
        #     )  # TODO: catch ValueError if not float
        #     unit = self.combobox_conversion_unit.currentText()
        # pixelsize = (factor, unit)
        undo_stack = df["id"].tolist()
        write(csv_filepath, data, metrics, set(), undo_stack)
        # write(csv_filepath, data, metrics, pixelsize, set(), undo_stack)
        write(tiff_filepath, image)
        self.logger.debug("ROI data exported.")
        msg = QMessageBox()
        msg.setWindowTitle("napari")
        msg.setText("ROI data exported.")
        msg.exec_()
