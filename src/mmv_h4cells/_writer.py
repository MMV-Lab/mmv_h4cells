import numpy as np
import csv
from typing import List, Tuple, Set
from aicsimageio.writers import OmeTiffWriter
from pathlib import Path
from qtpy.QtWidgets import QFileDialog
import json
import zarr


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


def write(path: Path, *data):
    writer = get_writer(path)
    writer(path, *data)


def get_writer(path: Path):
    if path.suffix == ".csv":
        return write_csv

    if path.suffix == ".tiff":
        return write_tiff

    if path.suffix == ".zarr":
        return write_zarr

    return None


def write_csv(
    path: Path,
    data: List[Tuple[int, int, Tuple[int, int]]],
    metrics: Tuple[float, float, float],
    # pixelsize: Tuple[float, str],
    # excluded: Set[int],
    # undo_stack: List[int],
):  # adjust if Metrics are added
    with open(path, "w", newline="") as file:
        csv_writer = csv.writer(file)

        csv_writer.writerow(
            ["ID", "Size [px]", "Centroid", ""]
            # + [json.dumps(list(excluded))]
            # + [json.dumps(undo_stack)]
        )  # , "metric name"
        for row in data:
            csv_writer.writerow(row)

        csv_writer.writerow([])
        csv_writer.writerow(
            ["Mean size [px]", "Std size [px]", "Threshold size [px]"]
        )  # , "metric name"
        csv_writer.writerow(metrics)
        # csv_writer.writerow([])
        # csv_writer.writerow(["1 pixel equals:"])
        # csv_writer.writerow(pixelsize)


def write_tiff(path: Path, data: np.ndarray):
    data = data.astype(np.uint16)
    OmeTiffWriter.save(data, path, dim_order_out="YX")


def write_zarr(
    path: Path,
    data_to_evaluate: np.ndarray,
    accepted_cells: np.ndarray,
    rejected_cells: np.ndarray,
    data: List[Tuple[int, int, Tuple[int, int]]],
    metrics: Tuple[float, float],
    undo_stack: List[int],
    selfdrawn_lower_bound: int,
):
    zarr_file = zarr.open(str(path), mode="w")
    zarr_file.create_dataset(
        "data_to_valuate",
        shape=data_to_evaluate.shape,
        dtype="i4",
        data=data_to_evaluate,
    )
    zarr_file.create_dataset(
        "accepted_cells",
        shape=accepted_cells.shape,
        dtype="i4",
        data=accepted_cells,
    )
    zarr_file.create_dataset(
        "rejected_cells",
        shape=rejected_cells.shape,
        dtype="i4",
        data=rejected_cells,
    )
    flattened_data = [(id_, amount, centroid[0], centroid[1]) for id_, amount, centroid in data]
    zarr_file.create_dataset(
        "data",
        shape=(len(data), 4),
        dtype="i4",
        data=flattened_data,
    )
    zarr_file.create_dataset(
        "metrics",
        shape=(len(metrics),),
        dtype="f8",
        data=metrics,
    )
    zarr_file.create_dataset(
        "undo_stack",
        shape=(len(undo_stack),),
        dtype="i4",
        data=undo_stack,
    )
    zarr_file.attrs["selfdrawn_lower_bound"] = selfdrawn_lower_bound
