"""Tests for main widget"""
import pytest

from mmv_cell_analyzer import CellAnalyzer


@pytest.fixture
def create_widget(make_napari_viewer):
    yield CellAnalyzer(make_napari_viewer())

