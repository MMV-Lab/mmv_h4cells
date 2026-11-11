"""Tests for main widget"""

import pytest

from unittest.mock import patch, call

import numpy as np
from pathlib import Path
from bioio import BioImage
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
    segmentation = BioImage(file).get_image_data("YX")
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


def _bound_callback(viewer, key):
    for bind, callback in viewer.keymap.items():
        if bind.to_text() == key:
            return callback
    return None


def test_hotkeys_follow_the_newest_widget(make_napari_viewer):
    """Reopening the panel creates a new widget; hotkeys must use it."""
    viewer = make_napari_viewer()
    first = CellAnalyzer(viewer)
    second = CellAnalyzer(viewer)

    assert _bound_callback(viewer, "K") == second.on_hotkey_include
    assert _bound_callback(viewer, "G") == second.on_hotkey_exclude
    assert _bound_callback(viewer, "H") == second.on_hotkey_undo
    assert _bound_callback(viewer, "J") == second.toggle_visibility_label_layers_hotkey

    # showing the older widget again hands the hotkeys back to it
    from qtpy.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendEvent(first, QEvent(QEvent.Show))
    assert _bound_callback(viewer, "K") == first.on_hotkey_include


def test_hotkeys(create_widget):
    widget = create_widget
    custom_keys = ["K", "G", "H", "J"]
    assert hotkeys_old(widget, custom_keys) or hotkeys_new(widget, custom_keys)


@patch.object(CellAnalyzer, "set_label_layer")
def test_get_label_layer(mock_set, create_widget):
    widget = create_widget
    file = Path(PATH / "ex-seg.tiff")
    segmentation = BioImage(file).get_image_data("ZYX")

    layer = widget.viewer.add_labels(segmentation, name="segmentation")
    mock_set.assert_called_once_with(layer)


def test_set_label_layer(create_widget):
    widget = create_widget
    file = Path(PATH / "ex-seg.tiff")
    segmentation = BioImage(file).get_image_data("ZYX")

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
    segmentation = BioImage(file).get_image_data("ZYX")

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
    segmentation = BioImage(file).get_image_data("ZYX")

    widget.viewer.add_labels(segmentation, name="segmentation")
    widget.start_analysis_on_click()
    widget.display_cell(4)
    assert np.max(widget.current_cell_layer.data) == 4


@patch("mmv_h4cells._widget.open_dialog", return_value=".")
@patch("mmv_h4cells._widget.read")
@patch.object(CellAnalyzer, "update_labels")
def test_import_no_csv(mock_update, mock_read, mock_open, create_widget):
    widget = create_widget
    widget.import_on_click()
    mock_open.assert_called_once()
    mock_read.assert_not_called()
    mock_update.assert_not_called()


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


# --- show / hide included, excluded, remaining ------------------------------

SHOW_KINDS = [
    pytest.param("included", {1}, id="included"),
    pytest.param("excluded", {2}, id="excluded"),
    pytest.param("remaining", {3}, id="remaining"),
]


@pytest.fixture
def widget_with_decisions(create_widget):
    """Cell 1 included, cell 2 excluded, cell 3 remaining."""
    widget = create_widget
    widget.viewer.add_labels(_labels(1, 2, 3), name="labels")
    widget.start_analysis_on_click()  # shows cell 1
    widget.include_on_click()  # shows cell 2
    widget.exclude_on_click()  # shows cell 3
    assert widget.included == {1}
    assert widget.excluded == {2}
    assert widget.remaining == {3}
    return widget


def _show_click(widget, kind):
    getattr(widget, f"show_{kind}_on_click")()


@pytest.mark.parametrize("kind, ids", SHOW_KINDS)
def test_show_cells(widget_with_decisions, kind, ids):
    widget = widget_with_decisions
    layers_before = len(widget.viewer.layers)

    _show_click(widget, kind)

    layer = getattr(widget, f"{kind}_layer")
    assert getattr(widget, f"btn_show_{kind}").text() == "Back"
    assert layer is not None
    assert layer in widget.viewer.layers
    assert len(widget.viewer.layers) == layers_before + 1
    assert set(np.unique(layer.data)) - {0} == ids
    assert layer.visible
    assert not widget.layer_to_evaluate.visible
    assert not widget.current_cell_layer.visible
    assert not widget.btn_include.isEnabled()
    assert not widget.btn_exclude.isEnabled()
    assert not widget.btn_undo.isEnabled()


@pytest.mark.parametrize("kind, ids", SHOW_KINDS)
def test_hide_cells(widget_with_decisions, kind, ids):
    widget = widget_with_decisions
    layers_before = len(widget.viewer.layers)
    _show_click(widget, kind)

    _show_click(widget, kind)  # "Back"

    assert (
        getattr(widget, f"btn_show_{kind}").text()
        == f"Show {kind.capitalize()}"
    )
    assert getattr(widget, f"{kind}_layer") is None
    assert len(widget.viewer.layers) == layers_before
    assert widget.layer_to_evaluate.visible
    assert widget.current_cell_layer.visible
    assert widget.viewer.layers.selection.active is widget.current_cell_layer
    assert widget.btn_include.isEnabled()
    assert widget.btn_exclude.isEnabled()
    assert widget.btn_undo.isEnabled()


def test_showing_one_kind_replaces_the_other(widget_with_decisions):
    widget = widget_with_decisions
    layers_before = len(widget.viewer.layers)
    _show_click(widget, "included")

    _show_click(widget, "excluded")

    assert widget.included_layer is None
    assert widget.btn_show_included.text() == "Show Included"
    assert widget.excluded_layer is not None
    assert widget.btn_show_excluded.text() == "Back"
    assert len(widget.viewer.layers) == layers_before + 1


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

@pytest.fixture
def widget_for_overlap(create_widget):
    """Cell 1 is displayed; cell 2 occupies rows 3-4, columns 0-1."""
    widget = create_widget
    widget.viewer.add_labels(_labels(1, 2), name="labels")
    widget.start_analysis_on_click()
    assert widget.current_cell_layer.selected_label == 1
    return widget


@pytest.mark.parametrize(
    "painted, expected",
    [
        ([], set()),  # the cell itself does not overlap
        ([(3, 0), (4, 1)], {(3, 0), (4, 1)}),  # painted over cell 2
        ([(10, 10)], set()),  # painted on background
    ],
    ids=["none", "other_cell", "background"],
)
def test_get_overlap(widget_for_overlap, painted, expected):
    widget = widget_for_overlap
    for row, col in painted:
        widget.current_cell_layer.data[row, col] = 1

    assert widget.get_overlap() == expected


@pytest.mark.parametrize(
    "user_drawn, choice",
    [(False, "Ok"), (True, "Ok"), (True, "Cancel")],
)
def test_handle_overlap(widget_for_overlap, user_drawn, choice):
    widget = widget_for_overlap
    widget.btn_segment.setText("Confirm")
    widget.current_cell_layer.mode = "paint"
    during = {}

    with patch("mmv_h4cells._widget.QMessageBox") as box:

        def dialog_shown():
            during["evaluate_opacity"] = widget.layer_to_evaluate.opacity
            during["current_opacity"] = widget.current_cell_layer.opacity
            overlap_layers = [
                layer
                for layer in widget.viewer.layers
                if layer.name == "Overlap"
            ]
            during["overlap_layers"] = len(overlap_layers)
            during["overlap_pixels"] = np.argwhere(
                overlap_layers[0].data
            ).tolist()
            return getattr(box, choice)

        box.return_value.exec_.side_effect = dialog_shown
        widget.handle_overlap({(3, 0)}, user_drawn)

    box.return_value.setText.assert_called_once()
    text = box.return_value.setText.call_args.args[0]
    assert text.startswith("Overlap detected and highlighted.")
    assert ("Cancel" in text) == user_drawn
    # while the dialog is shown the overlap is highlighted
    assert during["evaluate_opacity"] < 0.3
    assert during["current_opacity"] < 0.7
    assert during["overlap_layers"] == 1
    assert during["overlap_pixels"] == [[3, 0]]
    # afterwards everything is restored
    assert widget.current_cell_layer.opacity == 0.7
    assert widget.layer_to_evaluate.opacity == 0.3
    assert [layer.name for layer in widget.viewer.layers].count("Overlap") == 0
    assert len(widget.viewer.layers.selection) == 1
    assert widget.viewer.layers.selection.active is widget.current_cell_layer
    if user_drawn:
        box.return_value.setStandardButtons.assert_called_once()
        box.return_value.setDefaultButton.assert_called_once_with(box.Ok)
    else:
        box.return_value.setStandardButtons.assert_not_called()
        box.return_value.setDefaultButton.assert_not_called()
    if choice == "Cancel":
        assert widget.btn_segment.text() == "Draw own cell"
        assert widget.current_cell_layer.mode == "pan_zoom"
    else:
        assert widget.btn_segment.text() == "Confirm"
        assert widget.current_cell_layer.mode == "paint"


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

# --- import ---------------------------------------------------------------


def _write_session(zarr_path):
    from mmv_h4cells._writer import write_zarr

    data = np.zeros((10, 10), dtype=np.int32)
    data[0:2, 0:2] = 1
    data[4:6, 4:6] = 2
    data[8:10, 8:10] = 3
    accepted = np.where(data == 1, data, 0)
    rejected = np.where(data == 2, data, 0)
    write_zarr(
        zarr_path,
        data,
        accepted,
        rejected,
        [(1, 4, (0, 0))],
        (4.0, 0.0),
        [1, 2],
        4,
    )


def _assert_imported(widget):
    assert widget.layer_to_evaluate is not None
    assert widget.included == {1}
    assert widget.excluded == {2}
    assert widget.remaining == {3}
    assert widget.btn_start_analysis.isEnabled()


def test_import_zarr_next_to_csv(create_widget, tmp_path):
    widget = create_widget
    _write_session(tmp_path / "session.zarr")
    csv_path = str(tmp_path / "session.csv")

    with patch("mmv_h4cells._widget.open_dialog", return_value=csv_path):
        widget.import_on_click()

    _assert_imported(widget)


def test_import_restores_session_state(create_widget, tmp_path):
    widget = create_widget
    _write_session(tmp_path / "session.zarr")
    csv_path = str(tmp_path / "session.csv")

    with patch("mmv_h4cells._widget.open_dialog", return_value=csv_path):
        widget.import_on_click()

    assert widget.metric_data == [(1, 4, (0, 0))]
    assert widget.mean_size == 4.0
    assert widget.std_size == 0.0
    assert widget.undo_stack == [1, 2]
    assert widget.selfdrawn_lower_bound == 4
    assert widget.accepted_cells.shape == (10, 10)
    assert set(np.unique(widget.accepted_cells)) == {0, 1}
    assert set(np.unique(widget.rejected_cells)) == {0, 2}
    assert widget.btn_export.isEnabled()
    assert widget.lineedit_next_id.text() == "3"


def test_import_asks_for_zarr_when_missing_next_to_csv(
    create_widget, tmp_path
):
    widget = create_widget
    zarr_dir = tmp_path / "elsewhere" / "moved.zarr"
    zarr_dir.parent.mkdir()
    _write_session(zarr_dir)
    csv_path = str(tmp_path / "session.csv")  # no session.zarr beside it

    with patch(
        "mmv_h4cells._widget.open_dialog",
        side_effect=[csv_path, str(zarr_dir)],
    ) as mock_open, patch("mmv_h4cells._widget.QMessageBox") as mock_box:
        widget.import_on_click()

    # the user is told what is asked before the folder dialog opens
    text = mock_box.return_value.setText.call_args.args[0]
    assert "session.zarr" in text
    assert "session.csv" in text
    assert "ends in .zarr" in text
    second_call = mock_open.call_args_list[1]
    assert second_call.kwargs["dir"] is True
    assert ".zarr" in second_call.kwargs["title"]
    assert ".csv" in mock_open.call_args_list[0].kwargs["title"]
    _assert_imported(widget)


def test_import_missing_zarr_and_dialog_cancelled(create_widget, tmp_path):
    widget = create_widget
    csv_path = str(tmp_path / "session.csv")

    with patch(
        "mmv_h4cells._widget.open_dialog", side_effect=[csv_path, ""]
    ), patch("mmv_h4cells._widget.QMessageBox"):
        widget.import_on_click()

    assert widget.layer_to_evaluate is None
    assert len(widget.viewer.layers) == 0


def test_import_invalid_folder_shows_message(create_widget, tmp_path):
    widget = create_widget
    csv_path = str(tmp_path / "session.csv")
    not_a_session = tmp_path / "empty_folder"
    not_a_session.mkdir()

    with patch(
        "mmv_h4cells._widget.open_dialog",
        side_effect=[csv_path, str(not_a_session)],
    ), patch("mmv_h4cells._widget.QMessageBox") as mock_box:
        widget.import_on_click()

    # first the explanation, then the "not a valid session" message
    assert mock_box.return_value.setText.call_count == 2
    last_text = mock_box.return_value.setText.call_args.args[0]
    assert "not an exported session" in last_text
    assert widget.layer_to_evaluate is None
    assert len(widget.viewer.layers) == 0

# --- start analysis / next cell edge cases --------------------------------


def _labels(*ids):
    """Label image with one 2x2 square per id."""
    data = np.zeros((20, 20), dtype=np.int32)
    for n, id_ in enumerate(ids):
        data[n * 3 : n * 3 + 2, 0:2] = id_
    return data


@pytest.mark.parametrize("text", ["", "abc"])
def test_display_next_cell_invalid_next_id(create_started_widget, text):
    widget = create_started_widget
    widget.lineedit_next_id.setText(text)

    widget.display_next_cell()

    assert widget.current_cell_layer.selected_label in widget.remaining


def test_start_analysis_on_highest_remaining_id(create_widget):
    widget = create_widget
    widget.viewer.add_labels(_labels(1, 2, 3), name="labels")
    widget.lineedit_next_id.setText("3")

    widget.start_analysis_on_click()

    assert widget.current_cell_layer.selected_label == 3
    assert widget.next_id == 1  # wraps around
    assert widget.lineedit_next_id.text() == "1"


def test_start_analysis_with_single_cell(create_widget):
    widget = create_widget
    widget.viewer.add_labels(_labels(5), name="labels")

    widget.start_analysis_on_click()

    assert widget.current_cell_layer.selected_label == 5
    assert widget.next_id is None
    assert widget.lineedit_next_id.text() == ""


def test_display_next_cell_with_single_cell_left(create_widget):
    widget = create_widget
    widget.viewer.add_labels(_labels(5), name="labels")
    widget.start_analysis_on_click()
    widget.lineedit_next_id.setText("3")  # not in remaining, next_id is None

    widget.display_next_cell()

    assert widget.current_cell_layer.selected_label == 5


def test_start_analysis_cancelled_layer_choice(create_widget):
    widget = create_widget
    widget.viewer.add_labels(_labels(1, 2), name="first")
    widget.viewer.add_labels(_labels(3, 4), name="second")

    with patch("mmv_h4cells._widget.ChoiceDialog") as mock_dialog:
        mock_dialog.return_value.exec_.return_value = -1
        widget.start_analysis_on_click()

    assert widget.current_cell_layer is None
    assert widget.btn_start_analysis.isEnabled()
    assert not widget.btn_include.isEnabled()


def _jump_back_setup(widget):
    widget.viewer.add_labels(_labels(1, 2, 3), name="labels")
    widget.start_analysis_on_click()
    widget.undo_stack = [3]  # last evaluated was the highest id
    widget.next_id = 1
    widget.lineedit_next_id.setText("1")


def test_display_next_cell_jump_back_message(create_widget):
    widget = create_widget
    _jump_back_setup(widget)

    with patch("mmv_h4cells._widget.QMessageBox") as mock_box:
        widget.display_next_cell()

    mock_box.return_value.setText.assert_called_once_with(
        "Dataset is finished. Jumping to earlier cells."
    )
    mock_box.return_value.exec_.assert_called_once()


def test_display_next_cell_no_message_when_ignoring_jump_back(create_widget):
    widget = create_widget
    _jump_back_setup(widget)

    with patch("mmv_h4cells._widget.QMessageBox") as mock_box:
        widget.display_next_cell(True)

    mock_box.return_value.exec_.assert_not_called()

# --- layer events ----------------------------------------------------------


def _send(widget, event_type):
    from qtpy.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendEvent(widget, QEvent(event_type))


def test_layer_events_reconnected_after_hide_and_show(create_widget):
    from qtpy.QtCore import QEvent

    widget = create_widget
    _send(widget, QEvent.Hide)
    _send(widget, QEvent.Hide)  # hiding twice must not raise
    layer = widget.viewer.add_labels(_labels(1, 2), name="labels")
    assert widget.layer_to_evaluate is None  # ignored while hidden

    _send(widget, QEvent.Show)

    assert widget.layer_to_evaluate is layer
    other = widget.viewer.add_labels(_labels(3), name="other")
    assert widget.layer_to_evaluate is layer  # still connected, no change
    widget.viewer.layers.remove(layer)
    assert widget.layer_to_evaluate is other  # removal handled again


def test_remove_label_layer_before_analysis_resets(create_widget):
    widget = create_widget
    layer = widget.viewer.add_labels(_labels(1, 2), name="labels")
    assert widget.btn_start_analysis.isEnabled()

    with patch("mmv_h4cells._widget.QMessageBox") as mock_box:
        widget.viewer.layers.remove(layer)

    mock_box.assert_not_called()
    assert len(widget.viewer.layers) == 0
    assert widget.layer_to_evaluate is None
    assert widget.remaining == set()
    assert widget.lineedit_next_id.text() == ""
    assert not widget.btn_start_analysis.isEnabled()


def test_clearing_layers_during_analysis_does_not_loop(create_widget):
    widget = create_widget
    widget.viewer.add_labels(_labels(1, 2, 3), name="labels")
    widget.start_analysis_on_click()

    with patch("mmv_h4cells._widget.QMessageBox") as mock_box:
        # a re-add loop would show dialogs forever: turn that into a failure
        mock_box.return_value.exec_.side_effect = [None] * 3 + [
            RuntimeError("layer removal loop")
        ]
        widget.viewer.layers.clear()

    mock_box.assert_not_called()
    assert len(widget.viewer.layers) == 0


def test_removed_layers_are_restored_before_next_action(create_widget):
    widget = create_widget
    widget.viewer.add_labels(_labels(1, 2, 3), name="labels")
    widget.start_analysis_on_click()
    evaluate_layer = widget.layer_to_evaluate
    current_layer = widget.current_cell_layer

    with patch("mmv_h4cells._widget.QMessageBox") as mock_box:
        widget.viewer.layers.clear()
        mock_box.assert_not_called()  # nothing shown on removal itself

        widget.exclude_on_click()

    mock_box.return_value.setText.assert_called_once()
    assert evaluate_layer in widget.viewer.layers
    assert current_layer in widget.viewer.layers
    assert len(widget.excluded) == 1

# --- undo ----------------------------------------------------------------


def _include_in_order(widget, ids):
    widget.viewer.add_labels(_labels(1, 2, 3), name="labels")
    widget.start_analysis_on_click()
    widget.include_multiple(ids)


def test_undo_removes_metrics_of_undone_cell(create_widget):
    widget = create_widget
    _include_in_order(widget, [3, 1])

    widget.undo_on_click()  # undoes cell 1

    assert [entry[0] for entry in widget.metric_data] == [3]
    assert widget.included == {3}
    assert 1 in widget.remaining


def test_undo_after_export_removes_metrics_of_undone_cell(create_widget):
    widget = create_widget
    _include_in_order(widget, [3, 1])  # last included is not the highest id

    # export sorts the metric data by id
    with patch(
        "mmv_h4cells._widget.save_dialog", return_value="out.csv"
    ), patch("mmv_h4cells._widget.write"):
        widget.export_on_click()
    widget.undo_on_click()  # undoes cell 1

    assert [entry[0] for entry in widget.metric_data] == [3]
    assert widget.included == {3}
    assert widget.mean_size == 4  # only cell 3 (4 px) is left


# --- draw own cell ---------------------------------------------------------


@pytest.fixture
def widget_drawing(widget_for_overlap):
    """Widget in "draw own cell" mode (cell 2 is at rows 3-4, columns 0-1)."""
    widget = widget_for_overlap
    widget.draw_own_cell()
    assert widget.btn_segment.text() == "Confirm"
    assert widget.btn_cancel.isVisibleTo(widget) or not widget.btn_cancel.isHidden()
    assert widget.current_cell_layer.mode == "paint"
    return widget


def _paint(widget, *pixels, label=None):
    label = label or widget.current_cell_layer.selected_label
    for row, col in pixels:
        widget.current_cell_layer.data[row, col] = label


def test_draw_own_cell_overlap_cancel_discards_the_drawing(widget_drawing):
    widget = widget_drawing
    _paint(widget, (3, 0), (10, 10))  # (3, 0) lies on cell 2

    with patch("mmv_h4cells._widget.QMessageBox") as box:
        box.return_value.exec_.return_value = box.Cancel
        widget.draw_own_cell()  # confirm

    box.return_value.setStandardButtons.assert_called_once()  # Ok and Cancel
    assert widget.btn_segment.text() == "Draw own cell"
    assert widget.btn_include.isEnabled()
    assert widget.btn_exclude.isEnabled()
    assert widget.btn_cancel.isHidden()
    assert not widget.btn_undo.isHidden()
    assert widget.current_cell_layer.mode == "pan_zoom"
    assert widget.included == set()


def test_draw_own_cell_overlap_ok_keeps_drawing(widget_drawing):
    widget = widget_drawing
    _paint(widget, (3, 0))

    with patch("mmv_h4cells._widget.QMessageBox") as box:
        box.return_value.exec_.return_value = box.Ok
        widget.draw_own_cell()  # confirm

    assert widget.btn_segment.text() == "Confirm"
    assert not widget.btn_include.isEnabled()  # still in drawing mode
    assert not widget.btn_cancel.isHidden()
    assert widget.current_cell_layer.mode == "paint"  # can keep painting
    assert widget.included == set()


def test_draw_own_cell_with_several_labels_explains_and_keeps_drawing(
    widget_drawing,
):
    widget = widget_drawing
    label = widget.current_cell_layer.selected_label
    _paint(widget, (10, 10), label=label)
    _paint(widget, (12, 12), label=label + 1)

    with patch("mmv_h4cells._widget.QMessageBox") as box:
        widget.draw_own_cell()  # confirm

    text = box.return_value.setText.call_args.args[0]
    assert "only one cell can be confirmed at a time" in text
    assert widget.btn_segment.text() == "Confirm"
    assert widget.current_cell_layer.mode == "paint"
    assert widget.included == set()


def test_draw_own_cell_without_drawing_keeps_drawing(widget_drawing):
    widget = widget_drawing

    with patch("mmv_h4cells._widget.QMessageBox") as box:
        widget.draw_own_cell()  # confirm without painting anything

    box.return_value.setText.assert_called_once_with("No cell annotated.")
    assert widget.btn_segment.text() == "Confirm"
    assert widget.current_cell_layer.mode == "paint"


def test_draw_own_cell_confirm_includes_the_cell(widget_drawing):
    widget = widget_drawing
    label = widget.current_cell_layer.selected_label
    _paint(widget, (10, 10), (10, 11))

    widget.draw_own_cell()  # confirm

    assert label in widget.included
    assert widget.btn_segment.text() == "Draw own cell"
    assert widget.btn_include.isEnabled()

# --- undo button -----------------------------------------------------------


def test_undo_button_follows_the_undo_stack(create_widget):
    widget = create_widget
    widget.viewer.add_labels(_labels(1, 2, 3), name="labels")
    assert not widget.btn_undo.isEnabled()

    widget.start_analysis_on_click()
    assert not widget.btn_undo.isEnabled()  # nothing to undo yet

    widget.include_on_click()
    assert widget.btn_undo.isEnabled()
    widget.exclude_on_click()
    assert widget.btn_undo.isEnabled()

    widget.undo_on_click()  # one decision left
    assert widget.btn_undo.isEnabled()
    widget.undo_on_click()  # stack is empty again
    assert not widget.btn_undo.isEnabled()


def test_undo_button_enabled_after_include_multiple(create_widget):
    widget = create_widget
    widget.viewer.add_labels(_labels(1, 2, 3), name="labels")
    widget.start_analysis_on_click()
    widget.lineedit_include.setText("2")

    with patch("mmv_h4cells._widget.QMessageBox"):
        widget.include_multiple_on_click()

    assert widget.btn_undo.isEnabled()


def test_undo_button_while_showing_cells(widget_with_decisions):
    widget = widget_with_decisions
    assert widget.btn_undo.isEnabled()

    _show_click(widget, "included")
    assert not widget.btn_undo.isEnabled()
    _show_click(widget, "included")  # back
    assert widget.btn_undo.isEnabled()


def test_undo_button_stays_disabled_after_showing_with_empty_stack(
    create_widget,
):
    widget = create_widget
    widget.viewer.add_labels(_labels(1, 2, 3), name="labels")
    widget.start_analysis_on_click()

    _show_click(widget, "remaining")
    _show_click(widget, "remaining")  # back

    assert not widget.btn_undo.isEnabled()


def test_undo_button_after_import(create_widget, tmp_path):
    widget = create_widget
    _write_session(tmp_path / "session.zarr")  # undo stack [1, 2]

    with patch(
        "mmv_h4cells._widget.open_dialog",
        return_value=str(tmp_path / "session.csv"),
    ):
        widget.import_on_click()
    assert not widget.btn_undo.isEnabled()  # analysis not started yet

    widget.start_analysis_on_click()
    assert widget.btn_undo.isEnabled()


# --- self drawn id bound / session state -----------------------------------


def test_switching_label_layer_updates_selfdrawn_bound(create_widget):
    widget = create_widget
    widget.viewer.add_labels(_labels(1, 2), name="first")
    assert widget.selfdrawn_lower_bound == 3
    second = np.zeros((30, 30), dtype=np.int32)
    second[0:2, 0:2] = 9
    second[5:7, 5:7] = 4
    widget.viewer.add_labels(second, name="second")

    with patch("mmv_h4cells._widget.ChoiceDialog") as mock_dialog:
        mock_dialog.return_value.exec_.return_value = 1  # "second"
        widget.start_analysis_on_click()

    assert widget.layer_to_evaluate.name == "second"
    assert widget.selfdrawn_lower_bound == 10
    assert widget.accepted_cells.shape == (30, 30)
    assert widget.remaining == {4, 9}


def test_removing_the_label_layer_forgets_the_selfdrawn_bound(create_widget):
    widget = create_widget
    layer = widget.viewer.add_labels(_labels(1, 2), name="first")
    assert widget.selfdrawn_lower_bound == 3
    big = np.zeros((20, 20), dtype=np.int32)
    big[0:2, 0:2] = 50

    widget.viewer.layers.remove(layer)
    widget.viewer.add_labels(big, name="other")

    assert widget.selfdrawn_lower_bound == 51


def test_import_session_with_only_excluded_cells(create_widget, tmp_path):
    from mmv_h4cells._writer import write_zarr

    widget = create_widget
    data = np.zeros((10, 10), dtype=np.int32)
    data[0:2, 0:2] = 1  # remaining
    data[8:10, 8:10] = 3  # remaining
    rejected = np.zeros_like(data)
    rejected[4:6, 4:6] = 2  # excluded, no cell has been included
    write_zarr(
        tmp_path / "session.zarr",
        data,
        np.zeros_like(data),
        rejected,
        [],
        (0.0, 0.0),
        [2],
        4,
    )

    with patch(
        "mmv_h4cells._widget.open_dialog",
        return_value=str(tmp_path / "session.csv"),
    ):
        widget.import_on_click()

    assert widget.excluded == {2}
    assert set(np.unique(widget.rejected_cells)) == {0, 2}
    assert widget.remaining == {1, 3}


def test_import_keeps_the_stored_selfdrawn_bound(create_widget, tmp_path):
    from mmv_h4cells._writer import write_zarr

    widget = create_widget
    widget.viewer.add_labels(_labels(1, 2), name="other layer")
    data = np.zeros((10, 10), dtype=np.int32)
    data[0:2, 0:2] = 1
    data[4:6, 4:6] = 3
    accepted = np.where(data == 1, data, 0)
    write_zarr(
        tmp_path / "session.zarr",
        data,
        accepted,
        np.zeros_like(data),
        [(1, 4, (0, 0))],
        (4.0, 0.0),
        [1],
        7,  # differs from max id + 1 of the data
    )

    with patch(
        "mmv_h4cells._widget.open_dialog",
        return_value=str(tmp_path / "session.csv"),
    ):
        widget.import_on_click()

    assert widget.selfdrawn_lower_bound == 7
