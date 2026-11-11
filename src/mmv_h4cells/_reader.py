from qtpy.QtWidgets import QFileDialog
from bioio import BioImage
import zarr

_ZARR_NOT_FOUND = getattr(zarr.errors, "PathNotFoundError", FileNotFoundError)


def open_dialog(
    parent,
    filetype="*.csv",
    directory="",
    dir: bool = False,
    title: str = None,
):
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
    dir : bool
        Select a directory instead of a file
    title : str, optional
        Title of the dialog window

    Returns
    -------
    str
        Path of the selected file
    """
    dialog = QFileDialog()
    if dir:
        filepath = dialog.getExistingDirectory(
            parent, title or "Select Directory", directory=directory
        )
        return filepath
    else:
        filepath, _ = dialog.getOpenFileName(
            parent, title or "Select CSV-File", directory, filetype, filetype
        )
    return filepath


def read(path):
    reader = napari_get_reader(path)
    if reader is None:
        raise ValueError(f"Unsupported file type: {path}")
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

    if path.suffix == ".tiff" or path.suffix == ".tif":
        return read_tiff
    
    if path.suffix == ".zarr":
        return read_zarr

    return None


def read_tiff(path): # unused
    data = BioImage(path).get_image_data("YX")
    return data.astype("int32")

def read_zarr(path):
    try:
        zarr_file = zarr.open(path, mode="r")
    except _ZARR_NOT_FOUND as e:
        # zarr does not raise FileNotFoundError for a missing store
        raise FileNotFoundError(f"No zarr store at {path}") from e
    # sessions exported by earlier versions used a misspelled dataset name
    name = (
        "data_to_evaluate" if "data_to_evaluate" in zarr_file else "data_to_valuate"
    )
    data_to_evaluate = zarr_file[name][:]
    accepted_cells = zarr_file["accepted_cells"][:]
    rejected_cells = zarr_file["rejected_cells"][:]
    flattened_data = zarr_file["data"][:]
    data = [(int(id_), int(amount), (int(y), int(x))) for id_, amount, y, x in flattened_data]
    metrics = zarr_file["metrics"][:]
    undo_stack = zarr_file["undo_stack"][:]
    selfdrawn_lower_bound = zarr_file.attrs["selfdrawn_lower_bound"]
    return data_to_evaluate, accepted_cells, rejected_cells, data, metrics, undo_stack, selfdrawn_lower_bound
