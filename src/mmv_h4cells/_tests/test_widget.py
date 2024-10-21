"""Tests for main widget"""

import pytest

from unittest.mock import patch, call

import numpy as np
from pathlib import Path
from aicsimageio import AICSImage
from qtpy.QtWidgets import QMessageBox

from mmv_h4cells import CellAnalyzer

PATH = Path(__file__).parent / "data"


@pytest.fixture
def create_widget(make_napari_viewer):
    yield CellAnalyzer(make_napari_viewer())


@pytest.fixture
def create_started_widget(create_widget):
    widget = create_widget
    file = Path(PATH / "ex-seg.tiff")
    segmentation = AICSImage(file).get_image_data("YX")
    widget.viewer.add_labels(segmentation, name="segmentation")
    widget.start_analysis_on_click()
    yield widget


@pytest.fixture
def create_widget_in_analysis(create_started_widget):
    widget = create_started_widget
    widget.include_on_click()
    widget.exclude_on_click()
    yield widget


# widget instanzieren
# label layer laden
# analyse starten
# zelle(n) akzeptieren
# self.accepted_cells np.unique prüfen (equivalent mit self.evaluated_ids wenn keine abgelehnt)


def test_initialize(make_napari_viewer):
    try:
        CellAnalyzer(make_napari_viewer())
    except Exception as e:
        assert False, e
    assert True


def hotkeys_old(widget, custom_keys: list):
    # for napari version < 0.5.0
    hotkeys = widget.viewer.keymap.keys()
    for key in custom_keys:
        if not key in hotkeys:
            return False

    return True


def hotkeys_new(widget, custom_keys: list):
    # for napari version >= 0.5.0
    hotkeys = [bind.to_text() for bind in widget.viewer.keymap]
    for key in custom_keys:
        if not key in hotkeys:
            return False

    return True


def test_hotkeys(create_widget):
    widget = create_widget
    custom_keys = ["K", "G", "H", "J"]
    assert hotkeys_old(widget, custom_keys) or hotkeys_new(widget, custom_keys)


@patch.object(CellAnalyzer, "set_label_layer")
def test_get_label_layer(mock_set, create_widget):
    widget = create_widget
    file = Path(PATH / "ex-seg.tiff")
    segmentation = AICSImage(file).get_image_data("ZYX")

    layer = widget.viewer.add_labels(segmentation, name="segmentation")
    mock_set.assert_called_once_with(layer)


def test_set_label_layer(create_widget):
    widget = create_widget
    file = Path(PATH / "ex-seg.tiff")
    segmentation = AICSImage(file).get_image_data("ZYX")

    widget.viewer.layers.events.inserted.disconnect(widget.get_label_layer)
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    layer = widget.viewer.add_labels(segmentation, name="segmentation")
    widget.set_label_layer(layer)
    assert widget.layer_to_evaluate is layer
    assert widget.remaining == {1, 2, 3, 4, 5, 6, 7}
    assert widget.accepted_cells.shape == layer.data.shape
    assert np.max(widget.accepted_cells) == 0


@pytest.mark.parametrize(
    "params",
    [
        ((set(), set(), set()), 0),
        (({2}, set(), {2}), 3.5),
        (({1}, {2}, {3}), 1),
        (({1, 2}, {3, 4}, {5, 6}), 2),
    ],
)
def test_update_labels(create_widget, params):
    widget = create_widget
    widget.label_amount_excluded.setText("-8")
    widget.label_amount_included.setText("NOTHING")
    widget.label_amount_remaining.setText("9000.1")
    widget.label_mean_included.setText("take on me")
    widget.label_std_included.setText("take me on")
    widget.excluded = params[0][0]
    widget.included = params[0][1]
    widget.remaining = params[0][2]
    widget.mean_size = params[1]
    widget.std_size = params[1]
    widget.update_labels()
    assert widget.label_amount_excluded.text() == str(len(widget.excluded))
    assert widget.label_amount_included.text() == str(len(widget.included))
    assert widget.label_amount_remaining.text() == str(len(widget.remaining))
    assert widget.label_mean_included.text() == str(params[1])
    assert widget.label_std_included.text() == str(params[1])


def test_start_analysis(create_widget):
    widget = create_widget
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    file = Path(PATH / "ex-seg.tiff")
    segmentation = AICSImage(file).get_image_data("ZYX")

    layer = widget.viewer.add_labels(segmentation, name="segmentation")
    widget.start_analysis_on_click()
    assert widget.current_cell_layer is not None
    assert widget.current_cell_layer.data.shape == layer.data.shape
    assert np.array_equal(
        np.unique(widget.current_cell_layer.data), np.array([0, 1])
    )
    assert np.array_equal(
        np.where(widget.current_cell_layer.data == 1),
        np.where(layer.data == 1),
    )
    assert widget.lineedit_next_id.text() == "2"


def test_display_cell(create_widget):
    widget = create_widget
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    file = Path(PATH / "ex-seg.tiff")
    segmentation = AICSImage(file).get_image_data("ZYX")

    widget.viewer.add_labels(segmentation, name="segmentation")
    widget.start_analysis_on_click()
    widget.display_cell(4)
    assert np.max(widget.current_cell_layer.data) == 4


@patch("mmv_h4cells._widget.open_dialog", return_value="test.csv")
@patch("mmv_h4cells._widget.read")
@patch.object(CellAnalyzer, "update_labels")
@pytest.mark.parametrize(
    "csv_data",
    [
        ([], (0, 0)),
        ([(1, 1, (1, 1))], (1, 1)),
        (
            [(1, 1, (1, 1)), (2, 2, (2, 2))],
            (1.5, 0.5),
        ),
        ([], (0, 0)),
    ],
)
@pytest.mark.parametrize("layer_loaded", [True, False])
@pytest.mark.skip("Old test")
# TODO: Update test
def test_import(
    mock_update, mock_read, mock_open, create_widget, csv_data, layer_loaded
):
    widget = create_widget
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    if layer_loaded:
        file = Path(PATH / "ex-seg.tiff")
        segmentation = AICSImage(file).get_image_data("YX")
        widget.viewer.add_labels(segmentation, name="segmentation")
        mock_update.reset_mock()

    def mock_side(path):
        if path.suffix == ".csv":
            return csv_data
        elif path.suffix == ".tiff":
            img = np.zeros((10, 10), dtype=np.int32)
            if 1 in csv_data[4] and not 1 in csv_data[3]:
                img[1, 1] = 1
            if 2 in csv_data[4] and not 2 in csv_data[3]:
                img[2, 2] = 2
            if 3 in csv_data[4] and not 3 in csv_data[3]:
                img[3, 3] = 3
            return img

    mock_read.side_effect = mock_side
    widget.import_on_click()
    mock_open.assert_called_once()
    mock_read.assert_has_calls(
        [call(Path("test.csv")), call(Path("test.tiff"))], any_order=True
    )
    assert widget.metric_data == csv_data[0]
    assert widget.mean_size == csv_data[1][0]
    assert widget.std_size == csv_data[1][1]
    assert widget.lineedit_conversion_rate.text() == str(csv_data[2][0])
    assert (
        widget.combobox_conversion_unit.currentText() == csv_data[2][1]
        if csv_data[2][1] != "pixel"
        else "mm"
    )
    assert widget.excluded == csv_data[3]
    assert widget.undo_stack == csv_data[4]
    assert widget.accepted_cells.shape == (10, 10)
    for id_ in csv_data[4]:
        if id_ not in csv_data[3]:
            assert id_ in widget.accepted_cells
    assert widget.included == set(csv_data[4]) - csv_data[3]
    assert widget.btn_export.isEnabled()
    if layer_loaded:
        assert widget.remaining == {1, 2, 3, 4, 5, 6, 7} - set(csv_data[4])
        assert widget.lineedit_next_id.text() == str(min(widget.remaining))
        assert widget.btn_start_analysis.isEnabled()
    mock_update.assert_called_once()


@patch("mmv_h4cells._widget.open_dialog", return_value=".")
@patch("mmv_h4cells._widget.read")
@patch.object(CellAnalyzer, "update_labels")
def test_import_no_csv(mock_update, mock_read, mock_open, create_widget):
    widget = create_widget
    widget.import_on_click()
    mock_open.assert_called_once()
    mock_read.assert_not_called()
    mock_update.assert_not_called()


@patch("mmv_h4cells._widget.open_dialog")
@patch("mmv_h4cells._widget.read")
@patch.object(CellAnalyzer, "update_labels")
@pytest.mark.skip("Old test")
# TODO: Update test
def test_import_no_tiff(mock_update, mock_read, mock_open, create_widget):
    widget = create_widget

    def side_effect(_, filetype=None):
        if filetype is None:
            return "test.csv"
        return "."

    mock_open.side_effect = side_effect
    mock_read.side_effect = FileNotFoundError()
    widget.import_on_click()
    assert mock_open.call_count == 2
    assert mock_read.call_count == 2
    mock_read.assert_has_calls(
        [call(Path("test.tiff")), call(Path("test.tif"))]
    )
    mock_update.assert_not_called()


@patch("mmv_h4cells._widget.open_dialog", return_value="test.csv")
@patch("mmv_h4cells._widget.read")
@patch.object(CellAnalyzer, "update_labels")
@pytest.mark.skip("Old test")
# TODO Update test
def test_import_tif(mock_update, mock_read, mock_open, create_widget):
    def side_effect(path):
        if path.suffix == ".csv":
            return [], (0, 0), (1, "pixel"), set(), []
        elif path.suffix == ".tif":
            return np.zeros((10, 10), dtype=np.int32)
        raise FileNotFoundError()

    mock_read.side_effect = side_effect
    widget = create_widget
    widget.import_on_click()
    mock_open.assert_called_once()
    assert mock_read.call_count == 3
    mock_read.assert_has_calls(
        [
            call(Path("test.tiff")),
            call(Path("test.tif")),
            call(Path("test.csv")),
        ]
    )
    mock_update.assert_called_once()


@patch("mmv_h4cells._widget.open_dialog")
@patch("mmv_h4cells._widget.read")
@patch.object(CellAnalyzer, "update_labels")
@pytest.mark.parametrize("filename", ["test.tiff", "test.tif"])
@pytest.mark.skip("Old test")
# TODO: Update test
def test_import_tiff_prompt(
    mock_update, mock_read, mock_open, create_widget, filename
):
    widget = create_widget

    def open_side_effect(_, filetype=None):
        if filetype is None:
            return "test.csv"
        return f"subdirectory/{filename}"

    mock_open.side_effect = open_side_effect

    def read_side_effect(path):
        if path.suffix == ".csv":
            return [], (0, 0), (1, "pixel"), set(), []
        elif path.parent == Path("subdirectory"):
            return np.zeros((10, 10), dtype=np.int32)
        raise FileNotFoundError()

    mock_read.side_effect = read_side_effect
    widget.import_on_click()
    assert mock_open.call_count == 2
    assert mock_read.call_count == 4
    mock_read.assert_has_calls(
        [
            call(Path("test.tiff")),
            call(Path("test.tif")),
            call(Path(f"subdirectory/{filename}")),
            call(Path("test.csv")),
        ]
    )
    mock_update.called_once()


@patch("mmv_h4cells._widget.save_dialog", return_value="test.csv")
@patch("mmv_h4cells._widget.write")
def test_export(
    mock_write,
    mock_save_dialog,
    create_widget_in_analysis,
):
    widget = create_widget_in_analysis
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    widget.export_on_click()
    mock_save_dialog.assert_called_once()
    threshold = 0

    assert len(mock_write.call_args_list) == 3
    mock_write.assert_has_calls(
        [
            call(
                Path("test.csv"),
                widget.metric_data,
                (widget.mean_size, widget.std_size, threshold),
                # (
                #     float(conversion_rate) if conversion_rate else 1,
                #     (
                #         widget.combobox_conversion_unit.currentText()
                #         if conversion_rate
                #         else "pixel"
                #     ),
                # ),
                # widget.excluded,
                # widget.undo_stack,
            ),
            call(Path("test.tiff"), widget.accepted_cells),
        ],
        any_order=True,
    )


@patch("mmv_h4cells._widget.save_dialog", return_value=".csv")
@patch("mmv_h4cells._widget.write")
def test_export_no_file(
    mock_write, mock_save_dialog, create_widget_in_analysis
):
    widget = create_widget_in_analysis
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    widget.export_on_click()
    mock_save_dialog.assert_called_once()
    mock_write.assert_not_called()


@patch.object(CellAnalyzer, "check_for_overlap", return_value=False)
@patch.object(CellAnalyzer, "include")
@patch.object(CellAnalyzer, "display_next_cell")
@patch.object(QMessageBox, "exec_")
@pytest.mark.parametrize("user_drawn", [True, False])
@pytest.mark.parametrize("remaining", [set(), {1}, {1, 2}])
def test_include_on_click(
    mock_msg,
    mock_display,
    mock_include,
    mock_check,
    create_started_widget,
    user_drawn,
    remaining,
):
    widget = create_started_widget
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    widget.remaining = remaining.copy()
    undo_amount = len(widget.undo_stack)
    widget.include_on_click(user_drawn)
    if remaining == set():
        mock_display.assert_not_called()
        mock_include.assert_not_called()
        mock_check.assert_not_called()
        mock_msg.assert_called_once()
        return
    mock_check.assert_called_once()
    mock_include.assert_called_once_with(
        1, widget.current_cell_layer.data, not user_drawn
    )

    if remaining == {1, 2}:
        mock_display.assert_called_once()

    assert len(widget.undo_stack) == undo_amount + 1
    assert widget.undo_stack[-1] == 1


@patch.object(CellAnalyzer, "check_for_overlap", return_value=True)
@patch.object(CellAnalyzer, "include")
@patch.object(CellAnalyzer, "display_next_cell")
@patch.object(QMessageBox, "exec_")
@pytest.mark.parametrize("remaining", [set(), {1}])
def test_include_on_click_overlap(
    mock_msg,
    mock_display,
    mock_include,
    mock_check,
    create_started_widget,
    remaining,
):
    widget = create_started_widget
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    widget.remaining = remaining.copy()
    widget.include_on_click(False)

    if remaining == set():
        mock_display.assert_not_called()
        mock_include.assert_not_called()
        mock_check.assert_not_called()
        mock_msg.assert_called_once()
    else:
        mock_check.assert_called_once()
        mock_include.assert_not_called()
        mock_display.assert_not_called()
        mock_msg.assert_not_called()


@patch.object(CellAnalyzer, "update_labels")
@patch.object(CellAnalyzer, "display_next_cell")
@patch.object(QMessageBox, "exec_")
@pytest.mark.parametrize("remaining", [set(), {1}, {1, 2}])
def test_exclude_on_click(
    mock_msg, mock_display, mock_update, create_started_widget, remaining
):
    widget = create_started_widget
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    widget.remaining = remaining.copy()
    widget.exclude_on_click()
    if remaining == set():
        mock_display.assert_not_called()
        mock_update.assert_not_called()
        mock_msg.assert_called_once()
        return
    assert widget.excluded == {1}
    assert widget.remaining == ({2} if remaining == {1, 2} else set())
    assert widget.undo_stack == [1]
    mock_update.assert_called_once()
    mock_msg.assert_not_called()
    if remaining == {1, 2}:
        mock_display.assert_called_once()


@patch.object(CellAnalyzer, "calculate_metrics")
@patch.object(CellAnalyzer, "update_labels")
@patch.object(CellAnalyzer, "display_next_cell")
@patch.object(CellAnalyzer, "redisplay_current_cell")
@pytest.mark.parametrize("operation", ["include", "drawn", "exclude"])
@pytest.mark.parametrize("empty", [True, False])
def test_undo_on_click(
    mock_redisplay,
    mock_display,
    mock_update,
    mock_calculate,
    create_started_widget,
    operation,
    empty,
):
    widget = create_started_widget
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    id_ = 2
    if not empty:
        widget.include_on_click()
        widget.display_cell(id_)
        id_ += 1
    if operation == "exclude":
        widget.exclude_on_click()
        widget.display_cell(id_)
    elif operation == "include":
        widget.include_on_click()
        widget.display_cell(id_)
    else:
        widget.draw_own_cell()
        widget.current_cell_layer.data[(0, 0)] = (
            widget.current_cell_layer.selected_label
        )
        widget.draw_own_cell()
    last_id = widget.undo_stack[-1]
    amount_remaining = len(widget.remaining)
    amount_included = len(widget.included)
    amount_excluded = len(widget.excluded)
    sum_accepted = np.sum(widget.accepted_cells)
    amount_undo_stack = len(widget.undo_stack)

    mock_calculate.reset_mock()
    mock_update.reset_mock()
    mock_display.reset_mock()

    widget.undo_on_click()
    assert len(widget.undo_stack) == amount_undo_stack - 1
    if empty:
        assert not widget.undo_stack
    else:
        assert last_id not in widget.undo_stack

    if operation == "include":
        assert last_id in widget.remaining
        assert last_id not in widget.accepted_cells
        assert last_id not in widget.included
        assert last_id not in widget.excluded
        assert len(widget.included) == amount_included - 1
        assert len(widget.remaining) == amount_remaining + 1
        assert len(widget.excluded) == amount_excluded
        assert np.sum(widget.accepted_cells) < sum_accepted
    elif operation == "exclude":
        assert last_id in widget.remaining
        assert last_id not in widget.accepted_cells
        assert last_id not in widget.included
        assert last_id not in widget.excluded
        assert len(widget.included) == amount_included
        assert len(widget.remaining) == amount_remaining + 1
        assert len(widget.excluded) == amount_excluded - 1
        assert np.sum(widget.accepted_cells) == sum_accepted
    else:
        assert last_id not in widget.remaining
        assert last_id not in widget.accepted_cells
        assert last_id not in widget.included
        assert last_id not in widget.excluded
        assert len(widget.included) == amount_included - 1
        assert len(widget.remaining) == amount_remaining
        assert len(widget.excluded) == amount_excluded
        assert np.sum(widget.accepted_cells) < sum_accepted

    if operation == "drawn":
        # TODO: magic number
        assert widget.lineedit_next_id.text() == "8"
        mock_redisplay.assert_called_once()
        # assert widget.lineedit_next_id.text() == str(last_id)
    else:
        assert widget.lineedit_next_id.text() == str(last_id)
        mock_display.assert_called_once()

    mock_calculate.assert_called_once()
    mock_update.assert_called_once()


def test_double_undo_next_ids(create_started_widget):
    widget = create_started_widget
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    assert widget.lineedit_next_id.text() == "2"
    widget.include_on_click()
    assert widget.lineedit_next_id.text() == "3"
    widget.exclude_on_click()
    assert widget.lineedit_next_id.text() == "4"
    widget.undo_on_click()
    assert widget.lineedit_next_id.text() == "3"
    widget.undo_on_click()
    assert widget.lineedit_next_id.text() == "2"


@patch.object(CellAnalyzer, "calculate_metrics")
@patch.object(CellAnalyzer, "update_labels")
@patch.object(CellAnalyzer, "display_next_cell")
def test_undo_on_click_no_undo_stack(
    mock_display, mock_update, mock_calculate, create_started_widget
):
    widget = create_started_widget
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    assert not widget.undo_stack
    amount_remaining = len(widget.remaining)
    amount_included = len(widget.included)
    amount_excluded = len(widget.excluded)
    sum_accepted = np.sum(widget.accepted_cells)
    widget.undo_on_click()
    assert len(widget.remaining) == amount_remaining
    assert len(widget.included) == amount_included
    assert len(widget.excluded) == amount_excluded
    assert np.sum(widget.accepted_cells) == sum_accepted
    mock_calculate.assert_not_called()
    mock_update.assert_not_called()
    mock_display.assert_not_called()


@pytest.mark.skip("Old test")
# TODO: Update test
@patch.object(CellAnalyzer, "set_visibility_label_layers")
def test_show_included_on_click(mock_toggle, create_started_widget):
    widget = create_started_widget
    # widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    assert widget.btn_show_included.text() == "Show Included"
    assert not widget.included_layer
    amount_of_layers = len(widget.viewer.layers)
    widget.show_included_on_click()
    assert widget.btn_show_included.text() == "Back"
    assert len(widget.viewer.layers) == amount_of_layers + 1
    assert widget.included_layer
    mock_toggle.assert_called_once_with(False)


@pytest.mark.skip("Old test")
# TODO: Update test
@patch.object(CellAnalyzer, "set_visibility_label_layers")
def test_hide_included_on_click(mock_toggle, create_started_widget):
    widget = create_started_widget
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    # This assumes that show_included_on_click works correctly
    # for displaying the included cells
    widget.show_included_on_click()
    mock_toggle.reset_mock()
    assert widget.btn_show_included.text() == "Back"
    assert widget.included_layer
    amount_of_layers = len(widget.viewer.layers)
    widget.show_included_on_click()
    assert widget.btn_show_included.text() == "Show Included"
    assert len(widget.viewer.layers) == amount_of_layers - 1
    assert not widget.included_layer
    mock_toggle.assert_called_once_with(True)


@pytest.mark.skip("Old test")
# TODO: Update test
@patch.object(CellAnalyzer, "toggle_visibility_label_layers")
def test_show_excluded_on_click(mock_toggle, create_started_widget):
    widget = create_started_widget
    assert widget.btn_show_excluded.text() == "Show Excluded"
    assert not widget.excluded_layer
    amount_of_layers = len(widget.viewer.layers)
    widget.show_excluded_on_click()
    assert widget.btn_show_excluded.text() == "Back"
    assert len(widget.viewer.layers) == amount_of_layers + 1
    assert widget.excluded_layer
    mock_toggle.assert_called_once()


@pytest.mark.skip("Old test")
# TODO: Update test
@patch.object(CellAnalyzer, "toggle_visibility_label_layers")
def test_hide_excluded_on_click(mock_toggle, create_started_widget):
    widget = create_started_widget
    # This assumes that show_excluded_on_click works correctly
    # for displaying the excluded cells
    widget.show_excluded_on_click()
    mock_toggle.reset_mock()
    assert widget.btn_show_excluded.text() == "Back"
    assert widget.excluded_layer
    amount_of_layers = len(widget.viewer.layers)
    widget.show_excluded_on_click()
    assert widget.btn_show_excluded.text() == "Show Excluded"
    assert len(widget.viewer.layers) == amount_of_layers - 1
    assert not widget.excluded_layer
    mock_toggle.assert_called_once()


@pytest.mark.skip("Old test")
# TODO: Update test
@patch.object(CellAnalyzer, "toggle_visibility_label_layers")
def test_show_remaining_on_click(mock_toggle, create_started_widget):
    widget = create_started_widget
    assert widget.btn_show_remaining.text() == "Show Remaining"
    assert not widget.remaining_layer
    amount_of_layers = len(widget.viewer.layers)
    widget.show_remaining_on_click()
    assert widget.btn_show_remaining.text() == "Back"
    assert len(widget.viewer.layers) == amount_of_layers + 1
    assert widget.remaining_layer
    mock_toggle.assert_called_once()


@pytest.mark.skip("Old test")
# TODO: Update test
@patch.object(CellAnalyzer, "toggle_visibility_label_layers")
def test_hide_remaining_on_click(mock_toggle, create_started_widget):
    widget = create_started_widget
    # This assumes that show_remaining_on_click works correctly
    # for displaying the remaining cells
    widget.show_remaining_on_click()
    mock_toggle.reset_mock()
    assert widget.btn_show_remaining.text() == "Back"
    assert widget.remaining_layer
    amount_of_layers = len(widget.viewer.layers)
    widget.show_remaining_on_click()
    assert widget.btn_show_remaining.text() == "Show Remaining"
    assert len(widget.viewer.layers) == amount_of_layers - 1
    assert not widget.remaining_layer
    mock_toggle.assert_called_once()


@patch.object(QMessageBox, "exec_")
@patch.object(QMessageBox, "setText")
@patch.object(CellAnalyzer, "display_next_cell")
@patch.object(CellAnalyzer, "include_multiple")
@pytest.mark.parametrize(
    "ids",
    [
        None,
        [1],
        [2],
        [3],
        [4],
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 3],
        [2, 4],
        [3, 4],
        [1, 2, 3],
        [1, 2, 4],
        [1, 3, 4],
        [2, 3, 4],
        [1, 2, 3, 4],
    ],
)
def test_include_multiple_on_click(
    mock_include_multiple,
    mock_display_next,
    mock_setText,
    mock_exec,
    create_started_widget,
    ids,
):
    widget = create_started_widget
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)

    def mock_side(ids):
        included = {1} if 1 in ids else set()
        ignored = {2} if 2 in ids else set()
        overlapped = {3} if 3 in ids else set()
        faulty = {4} if 4 in ids else set()
        return included, ignored, overlapped, faulty

    mock_include_multiple.side_effect = mock_side
    with patch.object(
        widget, "get_ids_to_include", return_value=ids
    ) as mock_get_ids:
        widget.include_multiple_on_click()
        mock_get_ids.assert_called_once()
    if not ids:
        mock_include_multiple.assert_not_called()
        mock_display_next.assert_not_called()
        mock_setText.assert_not_called()
        mock_exec.assert_not_called()
        return
    mock_include_multiple.assert_called_once_with(ids)
    assert widget.lineedit_include.text() == ""
    assert widget.lineedit_next_id.text() == str(min(widget.remaining))
    mock_display_next.assert_called_once_with(True)
    text = ""
    if 1 in ids:
        text += f"Cells included: {{{1}}}\n"
    if 2 in ids:
        text += f"Cells ignored as they are already evaluated: {{{2}}}\n"
        text += "Only unprocessed cells can be included.\n"
    if 4 in ids:
        text += f"Cells ignored due to nonexistance: {{{4}}}\n"
        text += "Please only enter existing cell ids.\n"
    if 3 in ids:
        text += f"Cells not included due to overlap: {{{3}}}\n"
        text += "Please remove the overlap(s)."
    mock_setText.assert_called_once_with(text)
    mock_exec.assert_called_once()


@pytest.mark.parametrize(
    "ids",
    [
        ("", None),
        ("   1", [1]),
        ("1, 2", [1, 2]),
        ("1,2, 3", [1, 2, 3]),
        ("4 , 2 ,3, 1", [4, 2, 3, 1]),
    ],
)
def test_get_ids_to_include(create_widget, ids):
    widget = create_widget
    widget.lineedit_include.setText(ids[0])
    returned_ids = widget.get_ids_to_include()
    assert returned_ids == ids[1]


@patch.object(QMessageBox, "exec_")
def test_get_ids_to_include_invalid_input(mock_exec, create_widget):
    widget = create_widget
    widget.lineedit_include.setText("a")
    widget.get_ids_to_include()
    mock_exec.assert_called_once()


@patch.object(CellAnalyzer, "include")
@pytest.mark.parametrize("ids", [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4]])
def test_include_multiple(mock_include, create_widget_in_analysis, ids):
    widget = create_widget_in_analysis
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    included, ignored, overlapped, faulty = widget.include_multiple(ids)
    print(included, ignored, overlapped, faulty)
    assert overlapped == set()
    if 1 in ids:
        assert 1 in ignored
    if 2 in ids:
        assert 2 in faulty
    assert included == set(ids) - {1, 2}
    for id_ in included:
        id_ in mock_include.call_args_list


def test_include_multiple_overlap(create_widget_in_analysis):
    widget = create_widget_in_analysis
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    widget.layer_to_evaluate.data[:] = 0
    widget.layer_to_evaluate.data[0, 0] = 3
    widget.layer_to_evaluate.data[1, 1] = 3
    widget.layer_to_evaluate.data[2, 2] = 4
    widget.layer_to_evaluate.data[3, 3] = 4
    widget.accepted_cells[:] = 0
    widget.accepted_cells[0, 0] = 5
    widget.accepted_cells[1, 1] = 5
    widget.accepted_cells[2, 2] = 5
    widget.accepted_cells[3, 3] = 5
    included, ignored, overlapped, faulty = widget.include_multiple([3, 4])
    print(included, ignored, overlapped, faulty)
    assert included == set()
    assert ignored == set()
    assert overlapped == {3, 4}
    assert faulty == set()


@pytest.mark.parametrize("btn_text", ["Draw own cell", "Confirm"])
def test_draw_own_cell(create_started_widget, btn_text):
    widget = create_started_widget
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    widget.btn_segment.setText(btn_text)
    current_id = np.amax(widget.current_cell_layer.data)
    with patch.object(widget, "include_on_click") as mock_include:
        widget.draw_own_cell()
        if btn_text == "Draw own cell":
            assert widget.btn_segment.text() == "Confirm"
            assert widget.lineedit_next_id.text() == str(current_id)
            assert np.amax(widget.current_cell_layer.data) == 0
            assert len(widget.viewer.layers.selection) == 1
            assert (
                widget.viewer.layers.selection.active
                == widget.current_cell_layer
            )
            assert widget.current_cell_layer.mode == "paint"
            assert (
                widget.current_cell_layer.selected_label not in widget.included
            )
            assert (
                widget.current_cell_layer.selected_label
                not in widget.layer_to_evaluate.data
            )
            # TODO: NOT DONE
        else:
            mock_include.assert_called_once_with(True)
            assert widget.btn_segment.text() == "Draw own cell"


class TestDisplayNextCell:
    @patch.object(QMessageBox, "exec_")
    @patch.object(QMessageBox, "setText")
    def test_no_remaining_cells(self, mock_set_text, mock_exec, create_widget):
        widget = create_widget
        # remaining starts out as empty set
        widget.display_next_cell()
        mock_exec.assert_called_once()
        mock_set_text.assert_called_once_with("No more cells to evaluate.")

    @patch.object(QMessageBox, "exec_")
    @patch.object(QMessageBox, "setText")
    @patch.object(CellAnalyzer, "display_cell")
    @pytest.mark.parametrize(
        "remaining", [{10, 11}, {10}, {9, 10}, {11, 12}, {8, 9}, {9, 21}]
    )
    @pytest.mark.parametrize("last_evaluated_id", [1, 20])
    @pytest.mark.parametrize("check_lowered", [True, False])
    def test_display_next_cell(
        self,
        mock_display_cell,
        mock_set_text,
        mock_exec,
        create_widget,
        remaining,
        last_evaluated_id,
        check_lowered,
    ):
        widget = create_widget
        widget.lineedit_next_id.setText("10")
        widget.remaining = remaining
        higher_numbers = [value for value in remaining if value > last_evaluated_id]
        if len(higher_numbers) == 0:
            widget.next_id = min(remaining)
        else:
            widget.next_id = min(higher_numbers)
        widget.undo_stack = [last_evaluated_id]
        widget.display_next_cell(check_lowered)
        if last_evaluated_id == 1 or not check_lowered:
            if 10 in remaining:
                try:
                    mock_display_cell.assert_called_once_with(10)
                except AssertionError:
                    raise AssertionError(
                        f"Expected display_cell to be called once with 10, but was called {mock_display_cell.call_count} times."
                    )
                if 9 in remaining:
                    assert widget.lineedit_next_id.text() == "9"
                elif 11 in remaining:
                    assert widget.lineedit_next_id.text() == "11"
                else:
                    assert (
                        widget.lineedit_next_id.text() == ""
                    ), f"No next cell id should be set, but {widget.lineedit_next_id.text()} was set."
            elif min(remaining) > 10:  # 11 and 12 in remaining
                mock_exec.assert_not_called()
                mock_set_text.assert_not_called()
                try:
                    mock_display_cell.assert_called_once_with(11)
                except AssertionError:
                    raise AssertionError(
                        f"Expected display_cell to be called once with 11, but was called {mock_display_cell.call_count} times with {mock_display_cell.call_args_list}."
                    )
                assert widget.lineedit_next_id.text() == "12"
            elif max(remaining) < 10:  # 8 and 9 in remaining
                mock_exec.assert_not_called()
                mock_set_text.assert_not_called()
                try:
                    mock_display_cell.assert_called_once_with(9)
                except AssertionError:
                    raise AssertionError(
                        f"Expected display_cell to be called once with 9, but was called {mock_display_cell.call_count} times with {mock_display_cell.call_args_list}."
                    )
                assert widget.lineedit_next_id.text() == "8"
            else:  # 9 and 21 in remaining
                mock_exec.assert_not_called()
                mock_set_text.assert_not_called()
                try:
                    mock_display_cell.assert_called_once_with(9)
                except AssertionError:
                    raise AssertionError(
                        f"Expected display_cell to be called once with 9, but was called {mock_display_cell.call_count} time swith {mock_display_cell.call_args_list}."
                    )
                assert widget.lineedit_next_id.text() == "21"
        else:
            mock_exec.assert_not_called()
            mock_set_text.assert_not_called()


@patch.object(CellAnalyzer, "add_cell_to_accepted")
@pytest.mark.parametrize("remove", [True, False])
def test_include(mock_add, create_started_widget, remove):
    widget = create_started_widget
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    id_ = 1
    indices = np.where(widget.current_cell_layer.data == id_)
    data_array = np.zeros_like(widget.layer_to_evaluate.data)
    data_array[indices] = id_
    accepted_cells_sum = np.sum(widget.accepted_cells)
    widget.include(id_, data_array, remove)
    assert np.sum(widget.accepted_cells) > accepted_cells_sum
    if remove:
        assert 1 not in widget.remaining
    assert 1 in widget.included
    # mock_add.assert_called_once_with(1, widget.layer_to_evaluate.data)
    # Can't test this because mock can not compare the numpy arrays
    # So we just test if the function was called
    mock_add.assert_called_once()


@patch.object(CellAnalyzer, "handle_overlap")
@pytest.mark.parametrize("overlap", [{}, {(0, 0)}])
@pytest.mark.parametrize("user_drawn", [True, False])
def test_check_for_overlap(
    mock_handle_overlap, create_started_widget, overlap, user_drawn
):
    widget = create_started_widget
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    with patch.object(
        widget, "get_overlap", return_value=overlap
    ) as mock_get_overlap:
        retval = widget.check_for_overlap(user_drawn)
        mock_get_overlap.assert_called_once()

    if overlap:
        mock_handle_overlap.assert_called_once_with(overlap, user_drawn)
        assert retval
    else:
        assert not retval

@pytest.mark.skip("Old test")
# TODO: Update test
@pytest.mark.parametrize("current_ids", [[4], [3, 4], [2, 3, 4]])
def test_get_overlap(create_started_widget, current_ids):
    widget = create_started_widget
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    widget.current_cell_layer.data[:] = 0
    for id_ in current_ids:
        indices = np.where(widget.layer_to_evaluate.data == id_)
        widget.current_cell_layer.data[indices] = 100
    widget.accepted_cells[:] = 0
    accepted_ids = [1, 2, 3]
    for id_ in accepted_ids:
        indices = np.where(widget.layer_to_evaluate.data == id_)
        widget.accepted_cells[indices] = 101
    overlap_ids = set(current_ids) & set(accepted_ids)
    print(overlap_ids)
    combined_layer = widget.current_cell_layer.data + widget.accepted_cells
    expected_overlap = set(
        map(tuple, np.transpose(np.where(combined_layer == 201)))
    )
    assert widget.get_overlap() == expected_overlap


@patch.object(CellAnalyzer, "calculate_metrics")
@patch.object(CellAnalyzer, "update_labels")
@pytest.mark.parametrize("cell_id", [1, 2])
def test_add_cell_to_accepted(
    mock_update_labels, mock_calculate_metrics, create_started_widget, cell_id
):
    widget = create_started_widget
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    data = np.array([[0, 1, 1], [0, 1, 2], [0, 2, 1], [0, 2, 2]])
    assert cell_id not in widget.included
    metrics_len = len(widget.metric_data)
    widget.add_cell_to_accepted(cell_id, data)
    assert cell_id in widget.included
    assert len(widget.metric_data) == metrics_len + 1
    mock_update_labels.assert_called_once()
    mock_calculate_metrics.assert_called_once()


@patch.object(QMessageBox, "setText")
@patch.object(QMessageBox, "setStandardButtons")
@patch.object(QMessageBox, "setDefaultButton")
@pytest.mark.parametrize("overlap", [{}, {(0, 0)}])
@pytest.mark.parametrize("user_drawn", [True, False])
@pytest.mark.parametrize("return_value", [QMessageBox.Ok, QMessageBox.Cancel])
@pytest.mark.parametrize("remaining", [{}, {10}])
@pytest.mark.skip("Old test")
# TODO: Update test
def test_handle_overlap(
    mock_set_default_button,
    mock_set_standard_buttons,
    mock_set_text,
    create_started_widget,
    overlap,
    user_drawn,
    return_value,
    remaining,
):
    widget = create_started_widget
    widget.viewer.layers.events.removed.disconnect(widget.slot_layer_deleted)
    widget.remaining = remaining

    def test_side_effects():
        assert widget.layer_to_evaluate.opacity < 0.3
        assert widget.current_cell_layer.opacity < 0.7

    with patch.object(
        QMessageBox, "exec_", side_effect=test_side_effects
    ) as mock_exec:
        widget.handle_overlap(overlap, user_drawn)
        mock_exec.assert_called_once()
    mock_set_text.assert_called_once_with(
        "Overlap detected and highlighted. Please remove the overlap!"
    )
    if user_drawn:
        mock_set_standard_buttons.assert_called_once_with(
            QMessageBox.Ok | QMessageBox.Cancel
        )
        mock_set_default_button.assert_called_once_with(QMessageBox.Cancel)
        if return_value == QMessageBox.Cancel and len(widget.remaining) > 0:
            assert widget.btn_segment.text() == "Draw own cell"
            assert widget.current_cell_layer.mode == "pan_zoom"
    assert widget.current_cell_layer.opacity == 0.7
    assert widget.layer_to_evaluate.opacity == 0.3
    with pytest.raises(ValueError):
        widget.viewer.layers.index("Overlap")
    assert len(widget.viewer.layers.selection) == 1
    assert widget.viewer.layers.selection.active == widget.current_cell_layer


class TestCalculateMetrics:
    def test_values_initially_zero(self, create_widget):
        widget = create_widget
        assert widget.mean_size == 0
        assert widget.std_size == 0

    def test_no_metrics(self, create_widget):
        widget = create_widget
        widget.mean_size = 10
        widget.std_size = 10
        widget.calculate_metrics()
        assert widget.mean_size == 0
        assert widget.std_size == 0

    def test_with_metrics(self, create_widget):
        widget = create_widget
        widget.metric_data = [(1, 100, (100, 100)), (2, 200, (200, 200))]
        widget.calculate_metrics()
        assert widget.mean_size == 150
        assert widget.std_size == 50
