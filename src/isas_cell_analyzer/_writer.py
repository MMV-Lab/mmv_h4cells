import numpy as np
import csv
from typing import List, Tuple
from aicsimageio.writers import OmeTiffWriter
from qtpy.QtWidgets import QFileDialog


def save_dialog(parent, filetype="*.csv", directory=""):
    """
    Opens a dialog to select a location to save a file

    Parameters
    ----------
    parent : QWidget
        Parent widget for the dialog
    filetype : str
        Only files of this file type will be displayed
    directory : str
        Opens view at the specified directory

    Returns
    -------
    str
        Path of selected file
    """
    dialog = QFileDialog()
    filepath, _ = dialog.getSaveFileName(
        parent,
        "Select location for CSV and TIFF-File to be created",
        directory,
        filetype,
        filetype,
    )
    if not filepath.endswith(".csv"):
        filepath += ".csv"
    return filepath


def write(path: str, *data):
    writer = get_writer(path)
    writer(path, *data)


def get_writer(path):
    if path.suffix == ".csv":
        return write_csv

    if path.suffix == ".tiff":
        return write_tiff

    return None


def write_csv(
    path: str, data: List[Tuple[int, int, Tuple[int, int]]], metrics: Tuple[float, float], pixelsize: Tuple[float, str]
):  # adjust if Metrics are added
    with open(path, "w", newline="") as file:
        csv_writer = csv.writer(file)

        csv_writer.writerow(["ID", "Size [px]", "Centroid"]) #, "metric name"
        for row in data:
            csv_writer.writerow(row)

        csv_writer.writerow([])
        csv_writer.writerow(["Mean size", "Std size"]) #, "metric name"
        csv_writer.writerow(metrics)
        csv_writer.writerow([])
        csv_writer.writerow(["1 pixel equals:"])
        csv_writer.writerow(pixelsize)


def write_tiff(path: str, data: np.ndarray):
    OmeTiffWriter.save(data, path, dim_order_out="YX") # breaks on linux
