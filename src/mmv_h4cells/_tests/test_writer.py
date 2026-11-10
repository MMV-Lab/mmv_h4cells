"""Tests for writer"""

import pytest

import numpy as np

from unittest.mock import patch, Mock, call, mock_open
from pathlib import Path

from bioio import BioImage
from qtpy.QtWidgets import QFileDialog
from mmv_h4cells._writer import (
    save_dialog,
    write,
    get_writer,
    write_csv,
    write_tiff,
)


@patch.object(QFileDialog, "getSaveFileName", return_value=("test.csv", ""))
@pytest.mark.fatal
@pytest.mark.skip("This test is not working")
# TODO: Fix this test
def test_save_dialog(mock_getSaveFileName):
    retval = save_dialog(None)
    assert retval == "test.csv"
    mock_getSaveFileName.assert_called_once()


@patch.object(QFileDialog, "getSaveFileName", return_value=("test", ""))
@pytest.mark.fatal
@pytest.mark.skip("This test is not working")
# TODO: Fix this test
def test_save_dialog_no_extension(mock_getSaveFileName):
    retval = save_dialog(None)
    assert retval == "test.csv"
    mock_getSaveFileName.assert_called_once()


@patch.object(QFileDialog, "getSaveFileName", return_value=("", ""))
@pytest.mark.fatal
@pytest.mark.skip("This test is not working")
# TODO: Fix this test
def test_save_dialog_no_file(mock_getSaveFileName):
    retval = save_dialog(None)
    assert retval == ".csv"
    mock_getSaveFileName.assert_called_once()


@patch("mmv_h4cells._writer.get_writer")
def test_write(mock_get_writer):
    path = "test.csv"
    data1 = [(1, 2, (3, 4))]
    data2 = (5, 6)
    data3 = {7}
    data4 = 8
    mock_writer = Mock()
    mock_get_writer.return_value = mock_writer
    write(path, data1, data2, data3, data4)
    mock_get_writer.assert_called_once_with(path)
    mock_writer.assert_called_once_with(path, data1, data2, data3, data4)


@pytest.mark.parametrize(
    "filename, expected",
    [("test.csv", write_csv), ("test.tiff", write_tiff), ("test.txt", None)],
)
def test_get_writer(filename, expected):
    path = Path(filename)
    retval = get_writer(path)
    assert retval == expected


@patch("csv.writer")
def test_write_csv(mock_csv_writer):
    path = Path("test.csv")
    mock_file = mock_open()
    mock_file.return_value.__enter__.return_value = "arbitrary_string"
    mocked_writer = Mock()
    mock_writerow = Mock()
    mock_csv_writer.return_value = mocked_writer
    mocked_writer.writerow = mock_writerow
    data = [(1, 2, (3, 4))]
    metrics = (5, 6)
    # pixelsize = (7, "mm")
    # excluded = {8}
    # undo_stack = [9]
    with patch("builtins.open", mock_file):
        write_csv(path, data, metrics)
    mock_file.assert_called_once_with(path, "w", newline="")
    mock_csv_writer.assert_called_once_with("arbitrary_string", delimiter=",")
    # mock_writerow.assert_has_calls(
    #     [call(metrics), call(pixelsize)], any_order=True
    # )
    assert mock_writerow.call_count == 5


def test_write_tiff(tmp_path):
    array = np.arange(100).reshape(10, 10)
    path = tmp_path / "test.tiff"

    write_tiff(path, array)

    data = BioImage(path).get_image_data("YX")
    assert data.dtype == np.uint16
    assert np.array_equal(data, array)


@pytest.mark.parametrize(
    "default, current",
    [((None, None), (None, None)), ((None, "UTF-8"), (None, None))],
)
def test_write_csv_without_locale_information(tmp_path, default, current):
    path = tmp_path / "out.csv"

    with patch("locale.getdefaultlocale", return_value=default), patch(
        "locale.getlocale", return_value=current
    ):
        write_csv(path, [(1, 4, (0, 0))], (4.0, 0.0, 0))

    lines = path.read_text().splitlines()
    assert lines[0] == "ID,Size [px],Centroid,"


@pytest.mark.parametrize(
    "default, current",
    [(("de_DE", "cp1252"), (None, None)), ((None, None), ("German_Germany", "1252"))],
)
def test_write_csv_german_locale(tmp_path, default, current):
    path = tmp_path / "out.csv"

    with patch("locale.getdefaultlocale", return_value=default), patch(
        "locale.getlocale", return_value=current
    ):
        write_csv(path, [(1, np.float64(2.5), (0, 0))], (4.0, 0.0, 0))

    lines = path.read_text().splitlines()
    assert lines[0].startswith("ID;Size [px]")
    assert "2,5" in lines[1]
