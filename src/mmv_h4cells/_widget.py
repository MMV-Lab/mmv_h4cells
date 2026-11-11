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
from typing import List
from pathlib import Path
from mmv_h4cells import __version__ as version
from mmv_h4cells._reader import open_dialog, read
from mmv_h4cells._roi import analyse_roi
from mmv_h4cells._session import Session, Status
from mmv_h4cells._writer import save_dialog, write
from napari.layers.labels.labels import Labels

logger = logging.getLogger(__name__)

# The views of evaluated cells (see CellAnalyzer.toggle_cell_view)
CELL_VIEWS = ("included", "excluded", "remaining")


def _button(text: str, handler, tooltip: str = None) -> QPushButton:
    button = QPushButton(text)
    button.clicked.connect(handler)
    if tooltip is not None:
        button.setToolTip(tooltip)
    return button


def _separator(visible: bool = True) -> QWidget:
    """A thin horizontal line (or just a spacer) to structure the layout."""
    line = QWidget()
    line.setFixedHeight(4)
    line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    if visible:
        line.setStyleSheet("background-color: #c0c0c0")
    return line


def _from_session(name: str) -> property:
    """A widget attribute that lives in the Session."""
    return property(
        lambda self: getattr(self.session, name),
        lambda self, value: setattr(self.session, name, value),
    )


class CellAnalyzer(QWidget):
    """The napari widget to include and exclude the cells of a label layer.

    The cells and the decisions about them are managed by a Session. The
    widget shows them in napari and takes care of the buttons and messages.
    """

    # the state of the session, as attributes of the widget
    layer_to_evaluate = _from_session("layer_to_evaluate")
    current_cell_layer = _from_session("current_cell_layer")
    accepted_cells = _from_session("accepted_cells")
    rejected_cells = _from_session("rejected_cells")
    metric_data = _from_session("metric_data")
    mean_size = _from_session("mean_size")
    std_size = _from_session("std_size")
    remaining = _from_session("remaining")
    included = _from_session("included")
    excluded = _from_session("excluded")
    undo_stack = _from_session("undo_stack")
    next_id = _from_session("next_id")
    selfdrawn_lower_bound = _from_session("selfdrawn_lower_bound")

    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__()
        self.viewer = viewer
        logger.debug("Initializing CellAnalyzer...")

        self.session = Session()
        # label layers showing the included, excluded or remaining cells
        self.included_layer: Labels = None
        self.excluded_layer: Labels = None
        self.remaining_layer: Labels = None
        self._watched_layer = None

        self.initialize_ui()

        self.bind_hotkeys()

        self._layer_events_connected = False
        self.connect_layer_events()
        self.adopt_existing_label_layer()
        self.installEventFilter(self)

        logger.debug(f"CellAnalyzer v{version} initialized")
        logger.info("Ready to use")

    def connect_layer_events(self):
        if self._layer_events_connected:
            return
        logger.debug("Connecting slots...")
        self.viewer.layers.events.inserted.connect(self.get_label_layer)
        self.viewer.layers.events.removed.connect(self.slot_layer_deleted)
        self._layer_events_connected = True

    def disconnect_layer_events(self):
        if not self._layer_events_connected:
            return
        logger.debug("Disconnecting slots...")
        self.viewer.layers.events.inserted.disconnect(self.get_label_layer)
        self.viewer.layers.events.removed.disconnect(self.slot_layer_deleted)
        self._layer_events_connected = False

    def adopt_existing_label_layer(self):
        """Use the first label layer of the viewer, if none is set yet."""
        if self.layer_to_evaluate is not None:
            return
        for layer in self.viewer.layers:
            if isinstance(layer, Labels):
                self.set_label_layer(layer)
                break

    def slot_layer_deleted(self, event):
        layer = event.value
        logger.debug("Layer deleted")
        if layer is self.layer_to_evaluate and self.current_cell_layer is None:
            # analysis not started yet: nothing to protect, just forget it
            logger.debug("Label layer removed before analysis")
            self.reset_session()
            self.adopt_existing_label_layer()
        elif layer is self.layer_to_evaluate or layer is self.current_cell_layer:
            # Do not re-add the layer here: the viewer removes all layers when
            # it is closed or cleared, and re-adding would fight against that.
            # The layer is restored before the next action instead.
            logger.debug("Important layer removed")

    def restore_missing_layers(self):
        """Re-add the layers needed for the analysis if they were removed."""
        missing = [
            layer
            for layer in (self.layer_to_evaluate, self.current_cell_layer)
            if layer is not None and layer not in self.viewer.layers
        ]
        if not missing:
            return
        logger.debug("Restoring removed layers")
        for layer in missing:
            self.viewer.add_layer(layer)
        self.show_message(
            "Layers needed for the analysis were removed and have been "
            "restored. Please don't remove them."
        )

    def reset_session(self):
        """Forget the current label layer and all evaluation state."""
        self.watch_layer_edits(None)
        self.session.reset()
        self.show_next_id()
        self.btn_start_analysis.setEnabled(False)
        self.update_labels()
        self.update_undo_button()

    def eventFilter(self, source, event):
        if event.type() == QEvent.Hide:
            self.disconnect_layer_events()
        elif event.type() == QEvent.Show:
            self.bind_hotkeys()
            self.connect_layer_events()
            self.adopt_existing_label_layer()
        return super().eventFilter(source, event)

    def show_message(self, text: str, buttons=None, default_button=None):
        """Show a message box and return what the user chose."""
        msg = QMessageBox()
        msg.setWindowTitle("napari")
        msg.setText(text)
        if buttons is not None:
            msg.setStandardButtons(buttons)
        if default_button is not None:
            msg.setDefaultButton(default_button)
        return msg.exec_()

    def bind_hotkeys(self):
        """Bind the hotkeys to this widget.

        Existing bindings are overwritten: napari may create a new widget when
        the panel is reopened, and the old bindings would still point to the
        previous (hidden or deleted) widget.
        """
        for key, callback in (
            ("K", self.on_hotkey_include),
            ("G", self.on_hotkey_exclude),
            ("H", self.on_hotkey_undo),
            ("J", self.toggle_visibility_label_layers_hotkey),
        ):
            self.viewer.bind_key(key, callback, overwrite=True)

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
        logger.debug("Initializing UI...")
        # widgets that are updated or read later are attributes of the class
        # for ease of access
        self.create_labels()
        self.create_buttons()
        self.create_line_edits()
        content = self.create_content(self.create_roi_group())

        scroll_area = QScrollArea()
        scroll_area.setWidget(content)
        scroll_area.setWidgetResizable(True)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)

    def create_labels(self):
        self.label_next_id = QLabel("Start analysis at:")
        # label_metric = QLabel("Metric name:")
        self.label_amount_included = QLabel("0")
        self.label_amount_excluded = QLabel("0")
        self.label_amount_remaining = QLabel("0")
        self.label_mean_included = QLabel("0")
        self.label_std_included = QLabel("0")
        # self.label_metric_included = QLabel("0")

    def create_buttons(self):
        self.btn_start_analysis = _button(
            "Start analysis", self.start_analysis_on_click
        )
        self.btn_export = _button(
            "Export",
            self.export_on_click,
            "Export mask of included cells and analysis csv",
        )
        self.btn_import = _button(
            "Import",
            self.import_on_click,
            "Import previously exported mask and analysis csv to continue analysis",
        )
        self.btn_include = _button(
            "Include",
            self.include_on_click,
            'Include checked cell. Instead of clicking this button, you can also press the "K" key.',
        )
        self.btn_exclude = _button(
            "Exclude",
            self.exclude_on_click,
            'Exclude checked cell. Instead of clicking this button, you can also press the "G" key.',
        )
        self.btn_undo = _button(
            "Undo",
            self.undo_on_click,
            'Undo last selection. Instead of clicking this button, you can also press the "H" key.',
        )
        self.btn_cancel = _button("Cancel", self.cancel_on_click)
        self.btn_show_included = _button(
            "Show Included", self.show_included_on_click
        )
        self.btn_show_excluded = _button(
            "Show Excluded", self.show_excluded_on_click
        )
        self.btn_show_remaining = _button(
            "Show Remaining", self.show_remaining_on_click
        )
        self.btn_segment = _button("Draw own cell", self.draw_own_cell)
        self.btn_include_multiple = _button(
            "Include multiple", self.include_multiple_on_click
        )

        # nothing can be decided before the analysis has started
        for button in (
            self.btn_start_analysis,
            self.btn_export,
            self.btn_include,
            self.btn_exclude,
            self.btn_undo,
            self.btn_show_included,
            self.btn_show_excluded,
            self.btn_show_remaining,
            self.btn_segment,
            self.btn_include_multiple,
        ):
            button.setEnabled(False)
        self.btn_cancel.setVisible(False)

    def create_line_edits(self):
        self.lineedit_next_id = QLineEdit()
        self.lineedit_include = QLineEdit()

    def create_roi_group(self) -> QGroupBox:
        """The group of widgets to export the cells inside of a region."""
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
        self.btn_export_roi = _button("Export ROI", self.export_roi_on_click)

        groupbox = QGroupBox("ROI Analysis")
        groupbox.setStyleSheet(
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
        grid = QGridLayout()
        groupbox.setLayout(grid)
        grid.addWidget(label_range_y, 0, 0, 1, 1)
        grid.addWidget(self.lineedit_y_low, 0, 1, 1, 1)
        grid.addWidget(QLabel("-"), 0, 2, 1, 1)
        grid.addWidget(self.lineedit_y_high, 0, 3, 1, 1)

        grid.addWidget(label_range_x, 1, 0, 1, 1)
        grid.addWidget(self.lineedit_x_low, 1, 1, 1, 1)
        grid.addWidget(QLabel("-"), 1, 2, 1, 1)
        grid.addWidget(self.lineedit_x_high, 1, 3, 1, 1)

        grid.addWidget(label_threshold_size, 2, 0, 1, 1)
        grid.addWidget(self.lineedit_threshold_size, 2, 1, 1, -1)

        grid.addWidget(self.btn_export_roi, 3, 0, 1, -1)
        return groupbox

    def create_content(self, groupbox_roi: QWidget) -> QWidget:
        """Arrange all widgets."""
        title = QLabel("<h1>MMV_H4Cells</h1>")
        label_include = QLabel("Include:")
        label_included = QLabel("Included:")
        label_excluded = QLabel("Excluded:")
        label_remaining = QLabel("Remaining:")
        label_mean = QLabel("Mean size [px]:")
        label_std = QLabel("Std size [px]:")
        label_mean.setToolTip(
            "Only accounting for cells which have been included"
        )
        label_std.setToolTip(
            "Only accounting for cells which have been included"
        )

        content = QWidget()
        grid = QGridLayout()
        content.setLayout(grid)
        grid.addWidget(title, 0, 0, 1, -1)

        grid.addWidget(_separator(visible=False), 1, 0, 1, -1)

        grid.addWidget(self.btn_import, 2, 0, 1, 1)
        grid.addWidget(self.btn_segment, 2, 1, 1, 1)
        grid.addWidget(self.btn_export, 2, 2, 1, 1)

        grid.addWidget(self.btn_start_analysis, 3, 0, 1, 1)
        grid.addWidget(self.label_next_id, 3, 1, 1, 1)
        grid.addWidget(self.lineedit_next_id, 3, 2, 1, 1)

        grid.addWidget(_separator(), 4, 0, 1, -1)

        grid.addWidget(self.btn_exclude, 5, 0, 1, 1)
        grid.addWidget(self.btn_undo, 5, 1, 1, 1)
        grid.addWidget(self.btn_cancel, 5, 1, 1, 1)
        grid.addWidget(self.btn_include, 5, 2, 1, 1)

        grid.addWidget(label_included, 6, 0, 1, 1)
        grid.addWidget(self.label_amount_included, 6, 2, 1, 1)

        grid.addWidget(label_mean, 7, 0, 1, 1)
        grid.addWidget(self.label_mean_included, 7, 2, 1, 1)

        grid.addWidget(label_std, 8, 0, 1, 1)
        grid.addWidget(self.label_std_included, 8, 2, 1, 1)

        # content.layout().addWidget(self.label_metric_included, 9, 0, 1, 1)

        grid.addWidget(label_excluded, 9, 0, 1, 1)
        grid.addWidget(self.label_amount_excluded, 9, 2, 1, 1)

        grid.addWidget(label_remaining, 10, 0, 1, 1)
        grid.addWidget(self.label_amount_remaining, 10, 2, 1, 1)

        grid.addWidget(_separator(), 11, 0, 1, -1)

        grid.addWidget(label_include, 12, 0, 1, 1)
        grid.addWidget(self.lineedit_include, 12, 1, 1, 1)
        grid.addWidget(self.btn_include_multiple, 12, 2, 1, 1)

        grid.addWidget(self.btn_show_included, 14, 0, 1, 1)
        grid.addWidget(self.btn_show_excluded, 14, 1, 1, 1)
        grid.addWidget(self.btn_show_remaining, 14, 2, 1, 1)

        grid.addWidget(_separator(), 15, 0, 1, -1)

        grid.addWidget(groupbox_roi, 16, 0, 1, -1)
        return content


    def get_label_layer(self, event):
        logger.debug("New potential label layer detected...")
        if not self.layer_to_evaluate is None:
            logger.debug("Label layer already set")
            return
        if not isinstance(event.value, Labels):
            logger.debug("New layer invalid")
            return
        logger.debug("Label layer is valid")
        self.set_label_layer(event.value)

    def watch_layer_edits(self, layer):
        """Invalidate the boxes when the label layer is painted or replaced.

        Code that modifies the array of the layer in place (without napari
        noticing) has to call invalidate_boxes afterwards.
        """
        if layer is self._watched_layer:
            return
        for event_name in ("paint", "data"):
            if self._watched_layer is not None:
                try:
                    getattr(
                        self._watched_layer.events, event_name
                    ).disconnect(self.session.invalidate_boxes)
                except (AttributeError, TypeError, ValueError, RuntimeError):
                    pass
            event = getattr(getattr(layer, "events", None), event_name, None)
            if event is not None:
                event.connect(self.session.invalidate_boxes)
        self._watched_layer = layer

    def center_on_current_cell(self):
        """Move the camera to the displayed cell and zoom in."""
        label = self.current_cell_layer.selected_label
        labels, box = self.session.get_current_cell()
        if box is not None and label in labels:
            mask = self.current_cell_layer.data[box] == label
            self.viewer.camera.center = self.session.centroid_in_box(mask, box)
        self.viewer.camera.zoom = 7.5

    def set_label_layer(self, layer):
        logger.debug("Setting label layer...")
        self.watch_layer_edits(layer)
        self.btn_start_analysis.setEnabled(True)
        self.session.set_layer(layer)
        self.show_next_id()
        self.update_labels()

    def update_undo_button(self):
        """Enable Undo only if there is something to undo."""
        showing_other_layer = any(
            layer is not None
            for layer in (
                self.included_layer,
                self.excluded_layer,
                self.remaining_layer,
            )
        )
        self.btn_undo.setEnabled(
            len(self.undo_stack) > 0
            and self.current_cell_layer is not None
            and not showing_other_layer
        )

    def update_labels(self):
        logger.debug("Updating labels...")
        self.label_amount_excluded.setText(str(len(self.excluded)))
        self.label_amount_included.setText(str(len(self.included)))
        self.label_amount_remaining.setText(str(len(self.remaining)))
        self.label_mean_included.setText(str(self.mean_size))
        self.label_std_included.setText(str(self.std_size))

    def start_analysis_on_click(self):
        logger.debug("Analysis started...")
        label_layers = [
            layer.name for layer in self.viewer.layers if isinstance(layer, Labels)
        ]
        if len(label_layers) > 1:
            dialog = ChoiceDialog(label_layers, self.layer_to_evaluate.name)
            choice = dialog.exec_()
            if choice < 0:
                logger.debug("Layer selection cancelled")
                return
            layer = self.viewer.layers[label_layers[choice]]
            self.set_label_layer(layer)
        logger.debug(f"Using label layer: {self.layer_to_evaluate.name}")
        try:
            start_id = int(self.lineedit_next_id.text())
        except ValueError:
            logger.warning("Invalid start id")
            self.show_message("Invalid start id")
            return
        if len(self.remaining) < 1:
            logger.info("No cell to evaluate")
            self.show_message("No cells to evaluate")
            return
        self.btn_start_analysis.setEnabled(False)
        self.btn_exclude.setEnabled(True)
        self.btn_include.setEnabled(True)
        self.btn_import.setEnabled(False)
        self.btn_export.setEnabled(True)
        self.btn_show_included.setEnabled(True)
        self.btn_show_excluded.setEnabled(True)
        self.btn_show_remaining.setEnabled(True)
        self.btn_segment.setEnabled(True)
        self.btn_include_multiple.setEnabled(True)
        self.label_next_id.setText("Next cell:")
        self.layer_to_evaluate.opacity = 0.3
        self.session.use_compact_label_dtype()

        self.session.set_current_cell_layer(
            self.viewer.add_labels(
                np.zeros_like(self.layer_to_evaluate.data), name="Current Cell"
            )
        )
        self.update_undo_button()
        start_id = self.session.start(start_id)
        self.show_next_id()
        # start iterating through ids to create label layer for and zoom into centroid of label
        self.display_cell(start_id)

    def display_cell(self, cell_id: int):
        logger.debug(f"Displaying cell {cell_id}")
        centroid = self.session.show_cell(cell_id)
        self.current_cell_layer.opacity = 0.7
        self.current_cell_layer.refresh()
        if centroid is not None:
            self.viewer.camera.center = centroid
        logger.debug(f"Centroid: {centroid}")
        self.viewer.camera.zoom = 7.5  # !!
        self.current_cell_layer.selected_label = cell_id

    def import_on_click(self):
        logger.debug("Importing data...")
        csv_filepath = Path(
            open_dialog(
                self,
                title="Import: select the .csv file of an exported session",
            )
        )
        if str(csv_filepath) == ".":
            logger.debug("No csv file selected. Aborting.")
            return
        zarr_filepath = csv_filepath.with_suffix(".zarr")
        try:
            imported = read(zarr_filepath)
        except FileNotFoundError:
            logger.debug("No zarr next to the csv. Asking for location.")
            self.show_message(
                f"The session folder '{zarr_filepath.name}' was not found "
                f"next to '{csv_filepath.name}'.\n\n"
                "In the next window, please select the exported session "
                "folder. It was created by Export, next to the .csv file, "
                "and its name ends in .zarr."
            )
            zarr_filepath = Path(
                open_dialog(
                    self,
                    dir=True,
                    title="Select the exported session folder (ends in .zarr)",
                )
            )
            if str(zarr_filepath) == ".":
                logger.debug("No zarr file selected. Aborting.")
                return
            try:
                imported = read(zarr_filepath)
            except (FileNotFoundError, KeyError, ValueError):
                logger.warning("Selected folder is not a valid session")
                self.show_message(
                    "The selected folder is not an exported session. Please "
                    "select the folder ending in .zarr that Export created."
                )
                return
        (
            data_to_evaluate,
            accepted_cells,
            rejected_cells,
            data,
            metrics,
            undo_stack,
            selfdrawn_lower_bound,
        ) = imported
        logger.debug(f"Imported labels of shape {data_to_evaluate.shape}")
        self.session.load_state(
            accepted_cells,
            rejected_cells,
            data,
            metrics,
            undo_stack,
            selfdrawn_lower_bound,
        )
        self.update_undo_button()
        self.btn_export.setEnabled(True)

        layer = self.viewer.add_labels(data_to_evaluate, name="Imported Data")
        self.set_label_layer(layer)

        logger.debug("Filling in values for imported data")
        self.session.update_remaining()
        self.show_next_id()
        self.btn_start_analysis.setEnabled(True)

        self.update_labels()

    def export_on_click(self):
        logger.debug("Exporting data...")
        csv_filepath = Path(save_dialog(self))
        if csv_filepath.name == ".csv":
            logger.debug("No file selected. Aborting.")
            return
        zarr_filepath = csv_filepath.with_suffix(".zarr")
        tiff_filepath = csv_filepath.with_suffix(".tiff")
        self.session.sort_metric_data()
        write(
            csv_filepath,
            self.metric_data,
            (self.mean_size, self.std_size, 0),
        )
        logger.debug("Metrics written to csv")
        write(tiff_filepath, self.accepted_cells)
        logger.debug("Accepted cells written to tiff")
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
        logger.debug("Data written to zarr")

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
        self.restore_missing_layers()
        result = self.session.include_current(self_drawn)
        if result.status is Status.NO_REMAINING:
            self.show_message("No remaining cells")
            return False
        if result.status is Status.OVERLAP:
            self.handle_overlap(result.overlap, self_drawn)
            return False
        if result.status is Status.EMPTY:
            self.show_message("No cell found in the current cell layer.")
            return False
        if result.status is Status.MULTIPLE_LABELS:
            if self_drawn:
                text = (
                    "More than one label was drawn, but only one cell can be "
                    "confirmed at a time.\n\n"
                    "Please erase the extra labels (or use the same label for "
                    "all parts of the cell) and press Confirm again. Draw "
                    "further cells afterwards with another 'Draw own cell'."
                )
            else:
                text = "More than one label detected in current cell layer. Please remove the added label."
            self.show_message(text)
            return False

        self.update_labels()
        self.update_undo_button()
        if len(self.remaining) > 0:
            self.display_next_cell()
        return True

    def exclude_on_click(self):
        self.restore_missing_layers()
        result = self.session.exclude_current()
        if result.status is Status.NO_REMAINING:
            self.show_message("No remaining cells")
            return
        if result.status is Status.EMPTY:
            self.show_message("No cell found in the current cell layer.")
            return
        if result.status is Status.MULTIPLE_LABELS:
            self.show_message(
                "More than one label detected in current cell layer. Please remove the added label."
            )
            return

        self.update_undo_button()
        self.layer_to_evaluate.refresh()
        self.update_labels()
        if len(self.remaining) > 0:
            self.display_next_cell()

    def undo_on_click(self):
        self.restore_missing_layers()
        undone = self.session.undo()
        if undone is None:
            logger.info("No actions to undo")
            return
        if undone.layer_changed:
            self.layer_to_evaluate.refresh()
        self.lineedit_next_id.setText(str(undone.cell_id))

        self.update_labels()
        self.update_undo_button()
        if undone.selfdrawn:
            self.redisplay_current_cell()
        else:
            self.display_next_cell(True)

    def cancel_on_click(self):
        logger.debug("Cancelling draw own cell...")
        self.btn_include.setEnabled(True)
        self.btn_exclude.setEnabled(True)
        self.btn_undo.setVisible(True)
        self.btn_cancel.setVisible(False)
        self.current_cell_layer.mode = "pan_zoom"
        self.btn_segment.setText("Draw own cell")
        self.display_next_cell()

    def toggle_cell_view(self, kind: str):
        """Show the included, excluded or remaining cells; "Back" returns.

        kind is one of CELL_VIEWS. Only one of the views is shown at a time.
        """
        button = getattr(self, f"btn_show_{kind}")
        if button.text() == f"Show {kind.capitalize()}":
            logger.debug(f"Showing {kind} cells...")
            self.btn_include.setEnabled(False)
            self.btn_exclude.setEnabled(False)
            self.btn_undo.setEnabled(False)
            button.setText("Back")
            self.set_visibility_label_layers(False)
            for other in CELL_VIEWS:
                other_layer = getattr(self, f"{other}_layer")
                if other != kind and other_layer is not None:
                    self.viewer.layers.remove(other_layer)
                    setattr(self, f"{other}_layer", None)
                    getattr(self, f"btn_show_{other}").setText(
                        f"Show {other.capitalize()}"
                    )
            layer = self.viewer.add_labels(
                self.session.cells_to_show(kind), name=f"{kind.capitalize()} Cells"
            )
            setattr(self, f"{kind}_layer", layer)
            self.viewer.camera.zoom = 1
        else:
            logger.debug(f"Hiding {kind} cells...")
            self.viewer.layers.remove(getattr(self, f"{kind}_layer"))
            setattr(self, f"{kind}_layer", None)
            self.set_visibility_label_layers(True)
            button.setText(f"Show {kind.capitalize()}")
            self.viewer.layers.selection.active = self.current_cell_layer
            self.center_on_current_cell()
            self.btn_include.setEnabled(True)
            self.btn_exclude.setEnabled(True)
            self.update_undo_button()

    def show_included_on_click(self):
        self.toggle_cell_view("included")

    def show_excluded_on_click(self):
        self.toggle_cell_view("excluded")

    def show_remaining_on_click(self):
        self.toggle_cell_view("remaining")

    def include_multiple_on_click(self):
        self.restore_missing_layers()
        logger.debug(
            f"Including multiple cells for input {self.lineedit_include.text()}"
        )
        given_ids = self.get_ids_to_include()
        if given_ids is None:
            logger.debug("No valid ids in input")
            return
        logger.debug(f"Given ids: {given_ids}")
        included, ignored, overlapped, faulty = self.session.include_multiple(
            given_ids
        )
        self.update_labels()
        self.lineedit_include.setText("")
        self.update_undo_button()
        next_id = str(min(self.remaining)) if len(self.remaining) > 0 else ""
        self.lineedit_next_id.setText(next_id)
        self.display_next_cell(True)
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
        self.show_message(msgtext)

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
            logger.debug("Invalid input")
            self.show_message(
                "Please enter a comma separated list of integers, for example: 1, 5, 2, 8, 199, 5000."
            )
            return None
        return ids

    def draw_own_cell(self):
        self.restore_missing_layers()
        if self.btn_segment.text() == "Draw own cell":
            logger.debug("Draw own cell initialized")
            self.btn_segment.setText("Confirm")
            self.btn_exclude.setEnabled(False)
            self.btn_include.setEnabled(False)
            self.btn_undo.setVisible(False)
            self.btn_cancel.setVisible(True)
            # Set next id label to current cell layer id
            current_id = str(np.max(self.current_cell_layer.data))
            self.lineedit_next_id.setText(current_id)
            # Display empty current cell layer
            self.session.clear_current_cell()
            self.current_cell_layer.refresh()
            # Select current cell layer, set mode to paint
            self.viewer.layers.select_all()
            self.viewer.layers.selection.select_only(self.current_cell_layer)
            self.current_cell_layer.mode = "paint"
            # Select unique id
            self.current_cell_layer.selected_label = (
                self.session.new_drawn_label()
            )
        else:
            logger.debug("Draw own cell confirmed")
            self.current_cell_layer.mode = "pan_zoom"
            labels, _ = self.session.get_current_cell()
            if not labels:
                logger.debug("No label drawn")
                self.show_message("No cell annotated.")
                self.current_cell_layer.mode = "paint"
                return
            if self.include_on_click(True):
                self.btn_segment.setText("Draw own cell")
                self.btn_exclude.setEnabled(True)
                self.btn_include.setEnabled(True)
                self.btn_undo.setVisible(True)
                self.btn_cancel.setVisible(False)
            elif self.btn_segment.text() == "Confirm":
                # confirming failed but the user is still drawing
                self.current_cell_layer.mode = "paint"

    def show_next_id(self):
        """Show the id of the next cell in the text field."""
        next_id = self.session.next_id
        self.lineedit_next_id.setText("" if next_id is None else str(next_id))

    def display_next_cell(self, ignore_jump_back: bool = False):
        logger.debug("Displaying next cell...")
        if len(self.remaining) < 1:
            logger.debug("No cells left to evaluate")
            self.show_message("No more cells to evaluate.")
            return

        try:
            given_id = int(self.lineedit_next_id.text())
        except ValueError:
            given_id = None
        next_id, jumped_back = self.session.choose_next(
            given_id, ignore_jump_back
        )
        if jumped_back:
            self.show_message("Dataset is finished. Jumping to earlier cells.")
        self.display_cell(next_id)
        self.session.set_next_after(next_id)
        self.show_next_id()

    def redisplay_current_cell(self):
        logger.debug("Redisplaying current cell...")
        id_ = self.current_cell_layer.selected_label
        self.display_cell(id_)
        self.session.set_next_after(id_)
        self.show_next_id()

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
        logger.debug("Handling overlap...")
        overlap_indices = tuple(np.array(list(overlap)).T)
        self.layer_to_evaluate.opacity = 0.2
        self.current_cell_layer.opacity = 0.3
        logger.debug("Displaying overlap...")
        overlap_layer = self.viewer.add_labels(
            np.zeros_like(self.layer_to_evaluate.data),
            name="Overlap",
            opacity=1,
        )
        highlight = int(np.amax(self.current_cell_layer.data)) + 1
        overlap_layer.data[overlap_indices] = min(
            highlight, np.iinfo(overlap_layer.data.dtype).max
        )
        overlap_layer.refresh()
        if user_drawn:
            return_value = self.show_message(
                "Overlap detected and highlighted.\n\n"
                "OK: continue drawing and remove the overlap.\n"
                "Cancel: discard this drawing.",
                buttons=QMessageBox.Ok | QMessageBox.Cancel,
                default_button=QMessageBox.Ok,
            )
        else:
            return_value = self.show_message(
                "Overlap detected and highlighted. Please remove the overlap!"
            )
        self.current_cell_layer.opacity = 0.7
        self.layer_to_evaluate.opacity = 0.3
        self.viewer.layers.remove(overlap_layer)
        logger.debug("Overlap display removed")
        self.viewer.layers.select_all()
        self.viewer.layers.selection.select_only(self.current_cell_layer)
        if user_drawn and return_value == QMessageBox.Cancel:
            self.cancel_on_click()

    def set_visibility_label_layers(self, visible: bool):
        logger.debug(
            f"Setting visibility of label layers to {visible}..."
        )
        for layer in self.viewer.layers:
            if isinstance(layer, Labels):
                layer.visible = visible

    def toggle_visibility_label_layers(self):
        logger.debug("Toggling visibility of label layers...")
        for layer in self.viewer.layers:
            if isinstance(layer, Labels):
                layer.visible = not layer.visible

    def export_roi_on_click(self):
        logger.debug("Exporting ROI data...")
        try:
            lower_y, upper_y, lower_x, upper_x, threshold = (
                self.validate_roi_params()
            )
        except ValueError:
            return
        logger.debug("Valid ROI parameters")
        logger.debug(
            f"ROI parameters: {lower_y}, {upper_y}, {lower_x}, {upper_x}, {threshold}"
        )
        csv_filepath = Path(save_dialog(self, "(*.csv);; (*.tiff *.tif)"))
        if csv_filepath.name == ".csv":
            logger.debug("No file selected. Aborting.")
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
        logger.debug("Validating ROI parameters...")

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

        logger.debug(f"ROI parameters: {params}")
        if None in params:
            if params[-1] is None:
                msg_text = "Threshold must be a positive integer."
            else:
                msg_text = "All values for Range y and Range x must be set."
            self.show_message(msg_text)
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
        logger.debug("Exporting ROI data...")
        image, df, paths, threshold = params
        csv_filepath, tiff_filepath = paths
        data = df.itertuples(index=False)
        metrics = (
            np.round(df["count [px]"].mean(), 3),
            np.round(df["count [px]"].std(), 3),
            threshold,
        )
        write(csv_filepath, data, metrics)
        write(tiff_filepath, image)
        logger.debug("ROI data exported.")
        self.show_message("ROI data exported.")


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
