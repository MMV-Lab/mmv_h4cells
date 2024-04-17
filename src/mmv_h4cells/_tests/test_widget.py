"""Tests for main widget"""
import pytest

from mmv_h4cells import CellAnalyzer


@pytest.fixture
def create_widget(make_napari_viewer):
    yield CellAnalyzer(make_napari_viewer())

# widget instanzieren
# label layer laden
# analyse starten
# zelle(n) akzeptieren
# self.accepted_cells np.unique prüfen (equivalent mit self.evaluated_ids wenn keine abgelehnt)

def learn_test():
    pass