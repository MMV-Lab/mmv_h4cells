"""Tests for main widget"""

import pytest
from pathlib import Path
from aicsimageio import AICSImage
from glob import glob

from mmv_cell_analyzer import CellAnalyzer

PATH = Path(__file__).parent


@pytest.fixture
def create_widget(make_napari_viewer):
    yield CellAnalyzer(make_napari_viewer())


# widget instanzieren
# label layer laden
# analyse starten
# zelle(n) akzeptieren
# self.accepted_cells np.unique prüfen (equivalent mit self.evaluated_ids wenn keine abgelehnt)


@pytest.mark.justin
def test_learn(create_widget):
    widget = create_widget
    viewer = widget.viewer

    # load label layer
    print(PATH)
    print(Path.cwd())
    fn = next(Path(PATH / 'test_seg').glob('*.tiff'), None)
    labels = AICSImage(fn).get_image_data("YX")
    viewer.add_labels(labels, name=fn.name)
    assert widget.label_layer is not None

    # start analysis
    widget.start_analysis_on_click()

    # accept cell
    widget.lineedit_include.setText("1,2")
    included_ids = [int(id) for id in widget.lineedit_include.text().split(",")]  # get list of included ids (int)

    # include multiple cells
    widget.include_multiple_on_click()

    # test evaluated/remaining cells
    print(included_ids)
    assert all(id in widget.evaluated_ids for id in included_ids)
    assert not any(id in widget.remaining_ids for id in included_ids)
