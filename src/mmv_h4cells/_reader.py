from qtpy.QtWidgets import QFileDialog
import csv
from aicsimageio import AICSImage
import json
import zarr


def open_dialog(parent, filetype="*.csv", directory="", dir: bool = False):
    """
    Opens a dialog to select a file to open

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
        Path of the selected file
    """
    dialog = QFileDialog()
    if dir:
        filepath = dialog.getExistingDirectory(
            parent, "Select Directory", directory=directory
        )
        return filepath
    else:
        filepath, _ = dialog.getOpenFileName(
            parent, "Select CSV-File", directory, filetype, filetype
        )
    return filepath


def read(path):
    reader = napari_get_reader(path)
    return reader(path)


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """

    # if path.endswith(".csv"):
    if path.suffix == ".csv":
        return read_csv

    # if path.endswith(".tiff")
    if path.suffix == ".tiff" or path.suffix == ".tif":
        return read_tiff
    
    if path.suffix == ".zarr":
        return read_zarr

    return None


def read_csv(path):  # adjust if needed if metrics are added # unused
    """
    Reads data from a CSV file and processes each row.

    Parameters:
    - path (str): The file path to the CSV file.

    Returns:
    - tuple: A tuple containing a list of data and a tuple representing the last row read.
      The list of data contains tuples for rows where the first element is an integer.
      The last row is returned separately if its first element is a float.
    """
    data = (
        []
    )  # List to store tuples of rows with the first element as an integer
    metrics = ()  # Tuple to store the metrics if its first element is a float

    with open(path, "r") as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            for i in range(len(row)):
                try:
                    number = float(row[i])
                    if number.is_integer():
                        row[i] = int(number)
                    else:
                        row[i] = number
                except ValueError:
                    pass
            # Skip empty rows
            if len(row) == 0:
                continue

            if isinstance(row[0], str):
                if row[0].startswith("ID"):
                    undo_stack = json.loads(row[4])
                continue

            # Check the type of the first element in the row
            if isinstance(row[1], int):
                data.append(tuple(row))
            elif isinstance(row[1], float):
                # If the second element is a float, store the row separately
                metrics = tuple(row[0:2])

    if metrics == ():
        # If the metric values happen to all be integers, they are now the last row of the data
        metrics = data.pop(-1)

    return data, metrics, undo_stack


def read_tiff(path): # unused
    data = AICSImage(path).get_image_data("YX")
    return data.astype("int32")

def read_zarr(path):
    zarr_file = zarr.open(path, mode="r")
    data_to_evaluate = zarr_file["data_to_valuate"][:]
    accepted_cells = zarr_file["accepted_cells"][:]
    rejected_cells = zarr_file["rejected_cells"][:]
    flattened_data = zarr_file["data"][:]
    data = [(int(id_), int(amount), (int(y), int(x))) for id_, amount, y, x in flattened_data]
    metrics = zarr_file["metrics"][:]
    undo_stack = zarr_file["undo_stack"][:]
    selfdrawn_lower_bound = zarr_file.attrs["selfdrawn_lower_bound"]
    return data_to_evaluate, accepted_cells, rejected_cells, data, metrics, undo_stack, selfdrawn_lower_bound
