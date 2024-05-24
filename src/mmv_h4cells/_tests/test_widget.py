"""Tests for main widget"""
import pytest

from unittest.mock import patch

import numpy as np
from pathlib import Path
from aicsimageio import AICSImage
from qtpy.QtWidgets import QMessageBox

from mmv_h4cells import CellAnalyzer

PATH = Path(__file__).parent / "data"

@pytest.fixture
def create_widget(make_napari_viewer):
    yield CellAnalyzer(make_napari_viewer())

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

def test_hotkeys(create_widget):
    widget = create_widget
    hotkeys = widget.viewer.keymap.keys()
    assert "B" in hotkeys
    assert "F" in hotkeys
    assert "J" in hotkeys
    assert "V" in hotkeys

def test_set_label_layer(create_widget):
    widget = create_widget
    file = Path(PATH / "ex-seg.tiff")
    segmentation = AICSImage(file).get_image_data("ZYX")

    widget.viewer.layers.events.inserted.disconnect(widget.get_label_layer)
    layer = widget.viewer.add_labels(segmentation, name = "segmentation")
    widget.set_label_layer(layer)
    assert widget.layer_to_evaluate is layer
    assert widget.remaining == {1, 2, 3, 4, 5, 6, 7}
    assert widget.accepted_cells.shape == layer.data.shape
    assert np.max(widget.accepted_cells) == 0

def test_get_label_layer(create_widget):
    widget = create_widget
    file = Path(PATH / "ex-seg.tiff")
    segmentation = AICSImage(file).get_image_data("ZYX")

    layer = widget.viewer.add_labels(segmentation, name = "segmentation")
    assert widget.layer_to_evaluate is layer

def test_start_analysis(create_widget):
    widget = create_widget
    file = Path(PATH / "ex-seg.tiff")
    segmentation = AICSImage(file).get_image_data("ZYX")

    layer = widget.viewer.add_labels(segmentation, name = "segmentation")
    widget.start_analysis_on_click()
    assert widget.current_cell_layer is not None
    assert widget.current_cell_layer.data.shape == layer.data.shape
    assert np.array_equal(np.unique(widget.current_cell_layer.data), np.array([0,1]))
    assert np.array_equal(np.where(widget.current_cell_layer.data == 1), np.where(layer.data == 1))
    assert widget.lineedit_next_id.text() == "2"

def test_display_cell(create_widget):
    widget = create_widget
    file = Path(PATH / "ex-seg.tiff")
    segmentation = AICSImage(file).get_image_data("ZYX")

    widget.viewer.add_labels(segmentation, name = "segmentation")
    widget.start_analysis_on_click()
    widget.display_cell(4)
    assert np.max(widget.current_cell_layer.data) == 4

@pytest.mark.skip(reason="Not implementable without UI interaction")
def test_import(create_widget):
    widget = create_widget
    assert False

@pytest.mark.skip(reason="Not implementable without UI interaction")
def test_export(create_widget):
    widget = create_widget
    assert False

@pytest.mark.skip(reason="Not fully implementable without UI interaction")
def test_include_on_click(create_widget):
    widget = create_widget
    assert False

@pytest.mark.skip(reason="Not implemented yet")
def test_exclude_on_click(create_widget):
    widget = create_widget
    assert False

@pytest.mark.skip(reason="Not implemented yet")
def test_undo_on_click(create_widget):
    widget = create_widget
    assert False

@pytest.mark.skip(reason="Not implemented yet")
def test_show_included_on_click(create_widget):
    widget = create_widget
    assert False

@pytest.mark.skip(reason="Not implemented yet")
def test_show_excluded_on_click(create_widget):
    widget = create_widget
    assert False

@pytest.mark.skip(reason="Not implemented yet")
def test_show_remaining_on_click(create_widget):
    widget = create_widget
    assert False

@pytest.mark.skip(reason="Not implemented yet")
def test_include_multiple_on_click(create_widget):
    widget = create_widget
    assert False

@pytest.mark.skip(reason="Not implemented yet")
def test_get_ids_to_inclue(create_widget):
    widget = create_widget
    assert False

@pytest.mark.skip(reason="Not implemented yet")
def test_include_multiple(create_widget):
    widget = create_widget
    assert False

@pytest.mark.skip(reason="Not implemented yet")
def test_draw_own_cell(create_widget):
    widget = create_widget
    assert False

@pytest.mark.new
class TestDisplayNextCell:
    @patch.object(QMessageBox, "exec_")
    @patch.object(QMessageBox, "setText")
    def test_no_remaining_cells(self, mock_set_text, mock_exec, create_widget):
        widget = create_widget
        # remaining starts out as empty set
        widget.display_next_cell()
        mock_exec.assert_called_once()
        mock_set_text.assert_called_once_with("No more cells to evaluate.")

    @pytest.mark.skip(reason="Not implemented yet")
    def test_display_next_cell(self, create_widget):
        widget = create_widget
        assert False

@pytest.mark.skip(reason="Not implemented yet")
def test_include(create_widget):
    widget = create_widget
    assert False

@pytest.mark.skip(reason="Not implemented yet")
def test_check_for_overlap(create_widget):
    widget = create_widget
    assert False

@pytest.mark.skip(reason="Not implemented yet")
def test_get_overlap(create_widget):
    widget = create_widget
    assert False

@pytest.mark.skip(reason="Not implemented yet")
def test_add_cell_to_accepted(create_widget):
    widget = create_widget
    assert False

@pytest.mark.skip(reason="Not fully implementable without UI interaction")
def test_handle_overlap(create_widget):
    widget = create_widget
    assert False

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

def test_toggle_visibility_label_layers(create_widget):
    widget = create_widget
    file = Path(PATH / "ex-seg.tiff")
    segmentation = AICSImage(file).get_image_data("ZYX")
    labels1 = widget.viewer.add_labels(segmentation, name = "segmentation")
    labels2 = widget.viewer.add_labels(segmentation, name = "segmentation2")
    labels2.visible = False
    widget.toggle_visibility_label_layers()
    assert labels1.visible == False
    assert labels2.visible == True
