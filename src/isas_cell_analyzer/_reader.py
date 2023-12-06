from qtpy.QtWidgets import QFileDialog
import csv
import tifffile


def open_dialog(parent, filetype="*.csv", directory=""):
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
    dialog.setNameFilter(filetype)
    filepath = dialog.getExistingDirectory(
        parent, "Select CSV-File", directory=directory
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

    if path.endswith(".csv"):
        return read_csv

    if path.endswith(".tiff"):
        return read_tiff

    return None


def read_csv(path):  # adjust if needed if metrics are added
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
    pixelsize = ()

    with open(path, "r") as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            # Skip empty rows
            if len(row) == 0 or isinstance(row[0], str):
                continue

            # Check the type of the first element in the row
            if isinstance(row[1], int):
                data.append(tuple(row))
            elif isinstance(row[1], float):
                # If the second element is a float, store the row separately
                metrics = tuple(row)
            elif isinstance(row[1], str):
                pixelsize = tuple(row)

    if metrics == ():
        # If the metric values happen to all be integers, they are now the last row of the data
        metrics = data.pop(-1)

    return data, metrics, pixelsize


def read_tiff(path):
    data = tifffile.imread(path)
    return data
