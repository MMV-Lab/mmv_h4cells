import numpy as np
import pytest

from pathlib import Path
from mmv_h4cells._reader import napari_get_reader


@pytest.mark.parametrize(
    "filename", ["session.csv", "image.tiff", "image.tif"]
)
def test_only_zarr_is_a_session_format(filename):
    """The csv is only a report and images are opened by napari itself."""
    assert napari_get_reader(Path(filename)) is None


def test_zarr_is_a_session_format():
    assert callable(napari_get_reader(Path("session.zarr")))


def test_get_reader_pass():
    no_file = Path("fake.file")
    reader = napari_get_reader(no_file)
    assert reader is None


def test_zarr_roundtrip_without_included_cells(tmp_path):
    """A session in which no cell has been included yet can be saved/read."""
    from mmv_h4cells._reader import read_zarr
    from mmv_h4cells._writer import write_zarr

    path = tmp_path / "session.zarr"
    data = np.zeros((5, 5), dtype=np.int32)
    data[1:3, 1:3] = 1

    write_zarr(path, data, np.zeros_like(data), np.zeros_like(data), [], (0.0, 0.0), [], 2)
    (
        data_to_evaluate,
        accepted,
        rejected,
        metric_data,
        metrics,
        undo_stack,
        selfdrawn_lower_bound,
    ) = read_zarr(path)

    assert np.array_equal(data_to_evaluate, data)
    assert not accepted.any()
    assert not rejected.any()
    assert metric_data == []
    assert len(undo_stack) == 0
    assert selfdrawn_lower_bound == 2


def _session(path):
    from mmv_h4cells._writer import write_zarr

    data = np.zeros((5, 5), dtype=np.int32)
    data[1:3, 1:3] = 1
    write_zarr(
        path,
        data,
        np.zeros_like(data),
        np.zeros_like(data),
        [],
        (0.0, 0.0),
        [],
        2,
    )
    return data


def test_zarr_dataset_name_is_spelled_correctly(tmp_path):
    path = tmp_path / "session.zarr"
    _session(path)

    assert (path / "data_to_evaluate").is_dir()
    assert not (path / "data_to_valuate").exists()


def test_read_zarr_written_with_the_old_dataset_name(tmp_path):
    """Sessions exported by earlier versions used a misspelled name."""
    from mmv_h4cells._reader import read_zarr

    path = tmp_path / "session.zarr"
    data = _session(path)
    (path / "data_to_evaluate").rename(path / "data_to_valuate")

    data_to_evaluate = read_zarr(path)[0]

    assert np.array_equal(data_to_evaluate, data)
