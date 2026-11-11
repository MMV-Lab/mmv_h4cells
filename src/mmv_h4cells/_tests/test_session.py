"""Tests for the Session: the cells and the decisions about them.

None of these tests needs napari or a viewer, which makes them fast.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mmv_h4cells._session import Result, Session, Status, UndoResult


class ArrayLayer:
    """The part of a napari labels layer that the Session uses."""

    def __init__(self, data):
        self.data = data
        self.selected_label = 0


def cells(*ids, size=20):
    """Label image with one 2x2 cell per id: rows 0-1, 3-4, 6-7, ..."""
    data = np.zeros((size, size), dtype=np.int32)
    for n, id_ in enumerate(ids):
        data[n * 3 : n * 3 + 2, 0:2] = id_
    return data


def make_session(*ids):
    session = Session()
    session.set_layer(ArrayLayer(cells(*ids)))
    session.set_current_cell_layer(
        ArrayLayer(np.zeros_like(session.layer_to_evaluate.data))
    )
    return session


def show(session, cell_id):
    """What the widget does when it displays a cell."""
    session.show_cell(cell_id)
    session.current_cell_layer.selected_label = cell_id


def draw(session, label, rows, cols):
    """Draw a cell with the given label into the current cell layer."""
    session.current_cell_layer.data[rows, cols] = label
    session.current_cell_layer.selected_label = label


# --- the label layer ---------------------------------------------------------


def test_set_layer_collects_the_cells():
    session = make_session(5, 7, 9)

    assert session.remaining == {5, 7, 9}
    assert session.next_id == 5
    assert session.selfdrawn_lower_bound == 10
    assert not session.accepted_cells.any()
    assert not session.rejected_cells.any()
    assert session.accepted_cells.shape == session.layer_to_evaluate.data.shape


def test_set_layer_computes_the_bound_again_for_another_layer():
    session = make_session(1, 2)
    assert session.selfdrawn_lower_bound == 3

    session.set_layer(ArrayLayer(cells(1, 50)))

    assert session.selfdrawn_lower_bound == 51
    assert session.remaining == {1, 50}


def test_set_layer_keeps_the_bound_once_cells_were_evaluated():
    session = make_session(1, 2, 3)
    show(session, 1)
    session.include_current()
    session.selfdrawn_lower_bound = 77  # e.g. of an imported session

    session.set_layer(ArrayLayer(cells(1, 2, 3)))

    assert session.selfdrawn_lower_bound == 77
    assert session.included == {1}  # nothing was forgotten


def test_the_highest_uint16_id_does_not_wrap_around():
    session = Session()
    data = np.zeros((10, 10), dtype=np.uint16)
    data[0, 0] = 65535

    session.set_layer(ArrayLayer(data))

    assert session.selfdrawn_lower_bound == 65536


def test_reset_forgets_everything():
    session = make_session(1, 2, 3)
    show(session, 1)
    session.include_current()

    session.reset()

    assert session.layer_to_evaluate is None
    assert session.accepted_cells is None
    assert session.remaining == set() and session.included == set()
    assert session.metric_data == [] and session.undo_stack == []
    assert session.next_id is None and session.selfdrawn_lower_bound is None
    assert not session.has_session_state()


def test_load_state_continues_an_exported_session():
    session = Session()
    layer_data = cells(1, 3)  # cell 2 was excluded: it is not in the layer
    accepted = np.where(layer_data == 1, layer_data, 0)
    rejected = np.zeros_like(layer_data)
    rejected[3:5, 0:2] = 2

    session.load_state(
        accepted,
        rejected,
        [(1, 4, (0, 0))],
        (4.0, 0.0),
        np.array([1, 2]),
        4,
    )
    session.set_layer(ArrayLayer(layer_data))
    session.update_remaining()

    assert session.included == {1}
    assert session.excluded == {2}
    assert session.remaining == {3}
    assert session.undo_stack == [1, 2]
    assert session.mean_size == 4.0
    assert session.selfdrawn_lower_bound == 4  # the stored one is kept
    assert session.next_id == 3
    assert set(np.unique(session.rejected_cells)) == {0, 2}  # still there


# --- the cells --------------------------------------------------------------


def test_cell_boxes_are_exact():
    session = make_session(1, 2, 3)

    assert session.cell_box(2) == (slice(3, 5), slice(0, 2))
    assert session.cell_box(99) is None


def test_cell_boxes_include_excluded_cells():
    session = make_session(1, 3)
    session.rejected_cells[10:12, 5:8] = 2
    session.invalidate_boxes()

    assert session.cell_box(2) == (slice(10, 12), slice(5, 8))


def test_cell_box_falls_back_to_searching():
    session = make_session(1, 2)

    with patch("mmv_h4cells._session.MAX_CACHED_LABEL", 0):
        session.invalidate_boxes()
        assert session.get_boxes() == {}
        assert session.cell_box(2) == (slice(3, 5), slice(0, 2))
    assert session.get_boxes()[2] == (slice(3, 5), slice(0, 2))  # remembered


def test_show_cell_puts_only_that_cell_into_the_current_layer():
    session = make_session(1, 2, 3)

    centroid = session.show_cell(2)

    shown = session.current_cell_layer.data
    assert set(np.unique(shown)) == {0, 2}
    assert np.count_nonzero(shown) == 4
    assert centroid == pytest.approx((3.5, 0.5))
    session.show_cell(3)  # the previous one is removed
    assert set(np.unique(session.current_cell_layer.data)) == {0, 3}


def test_show_cell_of_a_missing_cell():
    session = make_session(1, 2)

    assert session.show_cell(99) is None
    assert not session.current_cell_layer.data.any()


def test_the_displayed_cell_is_not_scanned_for():
    session = make_session(1, 2, 3)
    show(session, 1)

    with patch("numpy.nonzero", side_effect=AssertionError("scanned")):
        labels, box = session.get_current_cell()

    assert labels == {1}
    assert box == (slice(0, 2), slice(0, 2))


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda d: d.__setitem__((15, 15), 1), id="add_pixel"),
        pytest.param(lambda d: d.__setitem__((0, 0), 0), id="erase_pixel"),
        pytest.param(lambda d: d.__setitem__((15, 15), 2), id="other_label"),
        pytest.param(lambda d: d.__setitem__(slice(None), 0), id="erase_all"),
        pytest.param(
            lambda d: (d.__setitem__((0, 0), 0), d.__setitem__((15, 15), 1)),
            id="move_pixel_same_count",
        ),
    ],
)
def test_manual_edits_of_the_current_cell_are_noticed(edit):
    session = make_session(1, 2, 3)
    show(session, 1)
    edit(session.current_cell_layer.data)

    data = session.current_cell_layer.data
    coords = np.nonzero(data)
    if len(coords[0]) == 0:
        expected = (set(), None)
    else:
        expected = (
            {int(v) for v in np.unique(data[coords])},
            tuple(slice(int(c.min()), int(c.max()) + 1) for c in coords),
        )
    assert session.get_current_cell() == expected


# --- including ----------------------------------------------------------------


def test_include_the_displayed_cell():
    session = make_session(1, 2, 3)
    show(session, 1)

    result = session.include_current()

    assert result == Result(Status.INCLUDED, cell_id=1)
    assert session.included == {1}
    assert session.remaining == {2, 3}
    assert session.undo_stack == [1]
    assert session.accepted_cells[0:2, 0:2].tolist() == [[1, 1], [1, 1]]
    assert session.metric_data == [(1, 4, (0, 0))]
    assert session.mean_size == 4.0
    assert session.std_size == 0.0


def test_include_a_self_drawn_cell():
    session = make_session(1, 2, 3)
    draw(session, 10, slice(15, 17), slice(15, 17))

    result = session.include_current(self_drawn=True)

    assert result == Result(Status.INCLUDED, cell_id=10)
    assert session.included == {10}
    assert session.remaining == {1, 2, 3}  # it was not remaining
    assert session.layer_to_evaluate.data[15, 15] == 10  # now part of the layer
    assert session.accepted_cells[16, 16] == 10
    assert session.cell_box(10) == (slice(15, 17), slice(15, 17))


def test_nothing_is_included_if_no_cells_remain():
    session = make_session(1)
    show(session, 1)
    session.include_current()

    assert session.include_current() == Result(Status.NO_REMAINING)


def test_nothing_is_included_if_the_current_layer_is_empty():
    session = make_session(1, 2)
    show(session, 1)
    session.current_cell_layer.data[:] = 0

    assert session.include_current() == Result(Status.EMPTY)
    assert session.included == set()


def test_nothing_is_included_if_there_are_several_labels():
    session = make_session(1, 2)
    show(session, 1)
    session.current_cell_layer.data[15, 15] = 2  # on the background

    assert session.include_current() == Result(Status.MULTIPLE_LABELS)
    assert session.included == set()


def test_nothing_is_included_if_the_cell_overlaps():
    session = make_session(1, 2)
    show(session, 1)
    session.current_cell_layer.data[3, 0] = 1  # on cell 2

    result = session.include_current()

    assert result.status is Status.OVERLAP
    assert result.overlap == {(3, 0)}
    assert session.included == set()
    assert session.undo_stack == []


def test_include_multiple():
    session = make_session(1, 2, 3)
    show(session, 1)
    session.include_current()  # cell 1 is evaluated

    included, ignored, overlapped, faulty = session.include_multiple(
        [0, 1, 2, 99]
    )

    assert included == {2}
    assert ignored == {1}  # already evaluated
    assert faulty == {99}  # does not exist
    assert overlapped == set()
    assert session.undo_stack == [1, 2]
    assert session.included == {1, 2}


def test_include_multiple_skips_overlapping_cells():
    session = make_session(1, 2, 3)
    session.accepted_cells[6, 0] = 5  # an accepted cell on cell 3
    session.invalidate_boxes()

    included, _, overlapped, _ = session.include_multiple([2, 3])

    assert included == {2}
    assert overlapped == {3}


def test_include_with_the_whole_image():
    session = make_session(1, 2)
    data = np.zeros_like(session.layer_to_evaluate.data)
    data[0:2, 0:2] = 1

    session.include(1, data)

    assert session.included == {1}
    assert session.remaining == {2}
    assert session.metric_data == [(1, 4, (0, 0))]


# --- excluding --------------------------------------------------------------


def test_exclude_the_displayed_cell():
    session = make_session(1, 2, 3)
    show(session, 2)

    result = session.exclude_current()

    assert result == Result(Status.EXCLUDED, cell_id=2)
    assert session.excluded == {2}
    assert session.remaining == {1, 3}
    assert session.undo_stack == [2]
    assert session.rejected_cells[3:5, 0:2].tolist() == [[2, 2], [2, 2]]
    assert not (session.layer_to_evaluate.data == 2).any()  # removed


def test_exclude_reports_why_it_did_nothing():
    session = make_session(1, 2)
    show(session, 1)
    session.current_cell_layer.data[15, 15] = 2
    assert session.exclude_current() == Result(Status.MULTIPLE_LABELS)

    session.current_cell_layer.data[:] = 0
    assert session.exclude_current() == Result(Status.EMPTY)

    session.remaining.clear()
    assert session.exclude_current() == Result(Status.NO_REMAINING)
    assert session.excluded == set()


# --- undoing ----------------------------------------------------------------


def test_nothing_to_undo():
    assert make_session(1, 2).undo() is None


def test_undo_an_inclusion():
    session = make_session(1, 2, 3)
    show(session, 1)
    session.include_current()

    undone = session.undo()

    assert undone == UndoResult(1, selfdrawn=False, layer_changed=False)
    assert session.remaining == {1, 2, 3}
    assert session.included == set()
    assert not session.accepted_cells.any()
    assert session.metric_data == []
    assert session.mean_size == 0
    assert session.undo_stack == []


def test_undo_an_exclusion():
    session = make_session(1, 2, 3)
    show(session, 2)
    session.exclude_current()

    undone = session.undo()

    assert undone == UndoResult(2, selfdrawn=False, layer_changed=True)
    assert session.remaining == {1, 2, 3}
    assert session.excluded == set()
    assert not session.rejected_cells.any()
    assert session.layer_to_evaluate.data[3:5, 0:2].tolist() == [[2, 2], [2, 2]]


def test_undo_a_self_drawn_cell():
    session = make_session(1, 2, 3)
    draw(session, 10, slice(15, 17), slice(15, 17))
    session.include_current(self_drawn=True)

    undone = session.undo()

    assert undone == UndoResult(10, selfdrawn=True, layer_changed=True)
    assert session.remaining == {1, 2, 3}  # it never was remaining
    assert session.included == set()
    assert not session.layer_to_evaluate.data[15:17, 15:17].any()
    assert not session.accepted_cells.any()


def test_undo_removes_the_metrics_of_the_undone_cell():
    session = make_session(1, 2, 3)
    session.include_multiple([3, 1])
    session.sort_metric_data()  # an export sorts the metrics by id

    session.undo()  # cell 1 was included last

    assert [entry[0] for entry in session.metric_data] == [3]
    assert session.included == {3}


# --- the next cell ---------------------------------------------------------


def test_start_with_the_given_cell():
    session = make_session(1, 2, 3)

    assert session.start(2) == 2
    assert session.next_id == 3


def test_start_uses_a_remaining_cell_if_the_given_one_is_not():
    session = make_session(5, 8, 9)

    assert session.start(7) == 5  # the closest lower one
    assert session.next_id == 8
    assert session.start(1) == 5  # nothing lower: the lowest
    assert session.start(9) == 9
    assert session.next_id == 5  # wraps around


def test_the_only_cell_has_no_next_cell():
    session = make_session(4)

    assert session.start(4) == 4
    assert session.next_id is None


def test_choose_next_follows_the_computed_cell():
    session = make_session(1, 2, 3)
    session.start(1)  # next is 2

    assert session.choose_next(None) == (2, False)
    assert session.choose_next(3) == (3, False)  # the user typed an id
    session.set_next_after(3)
    assert session.next_id == 1


# --- showing the evaluated cells -----------------------------------------------


def test_cells_to_show():
    session = make_session(1, 2, 3)
    show(session, 1)
    session.include_current()  # 1 is included
    show(session, 2)
    session.exclude_current()  # 2 is excluded

    assert set(np.unique(session.cells_to_show("included"))) == {0, 1}
    assert set(np.unique(session.cells_to_show("excluded"))) == {0, 2}
    assert set(np.unique(session.cells_to_show("remaining"))) == {0, 3}


# --- the label dtype -----------------------------------------------------------


def test_labels_are_stored_as_uint16_if_the_ids_fit():
    session = make_session(1, 2, 3)
    original = session.layer_to_evaluate.data.copy()

    session.use_compact_label_dtype()

    for array in (
        session.layer_to_evaluate.data,
        session.accepted_cells,
        session.rejected_cells,
        session.current_cell_layer.data,
    ):
        assert array.dtype == np.uint16
    assert np.array_equal(session.layer_to_evaluate.data, original)


def test_labels_stay_wide_if_the_ids_do_not_fit():
    session = make_session(1, 65400)

    session.use_compact_label_dtype()

    assert session.layer_to_evaluate.data.dtype == np.int32


def test_small_label_types_are_left_alone():
    session = Session()
    session.set_layer(ArrayLayer(cells(1, 2).astype(np.uint8)))

    session.use_compact_label_dtype()

    assert session.layer_to_evaluate.data.dtype == np.uint8


def test_labels_are_widened_when_a_label_does_not_fit():
    session = Session()
    data = np.zeros((10, 10), dtype=np.uint16)
    data[0, 0] = 65535
    session.set_layer(ArrayLayer(data))
    session.set_current_cell_layer(ArrayLayer(np.zeros_like(data)))

    label = session.new_drawn_label()

    assert label == 65536
    for array in (
        session.layer_to_evaluate.data,
        session.accepted_cells,
        session.rejected_cells,
        session.current_cell_layer.data,
    ):
        assert array.dtype == np.int32


# --- independence of the GUI ---------------------------------------------------


def test_the_session_does_not_need_qt_or_napari():
    package = Path(__file__).parents[1]
    code = f"""
import sys, types
package = types.ModuleType("mmv_h4cells")
package.__path__ = [{str(package)!r}]
sys.modules["mmv_h4cells"] = package  # without running its __init__
sys.modules["napari"] = None  # importing these now fails
sys.modules["qtpy"] = None
import mmv_h4cells._session
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
