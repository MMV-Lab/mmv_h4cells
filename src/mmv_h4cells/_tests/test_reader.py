import numpy as np

from pathlib import Path
import csv
import tifffile
from mmv_h4cells._reader import napari_get_reader


# tmp_path is a pytest fixture
def test_reader_csv(tmp_path):
    """An example of how you might test your plugin."""

    # write some fake data using your supported file format
    my_test_file = Path(tmp_path / "myfile.csv")
    with open(my_test_file, "w", newline="") as file:
        csv_writer = csv.writer(file)
        
        csv_writer.writerow(["ID", "Size [px]", "Centroid", ""])
        
    # try to read it back in
    reader = napari_get_reader(my_test_file)
    assert callable(reader)
    
def test_reader_tiff(tmp_path):
    my_test_file = Path(tmp_path / "myfile.tiff")
    data = np.random.rand(10,10)
    tifffile.imwrite(my_test_file, data)
    
    reader = napari_get_reader(my_test_file)
    assert callable(reader)


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
