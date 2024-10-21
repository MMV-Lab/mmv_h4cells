"""Tests for writer"""

import pytest

import numpy as np

from unittest.mock import patch, Mock, call, mock_open
from pathlib import Path

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


@patch("aicsimageio.writers.OmeTiffWriter.save")
def test_write_tiff(mock_save):
    array = np.zeros((10, 10))

    def mock_side_effect(data, _, dim_order_out):
        assert isinstance(data[0, 0], np.uint16)
        assert dim_order_out == "YX"

    mock_save.side_effect = mock_side_effect
    path = Path("test.tiff")
    write_tiff(path, array)
    mock_save.assert_called_once()
    args, arg3 = mock_save.call_args
    dim_order_out = arg3["dim_order_out"]
    arg1, arg2 = args
    assert np.array_equal(arg1, array)
    assert arg2 == path
    assert dim_order_out == "YX"
