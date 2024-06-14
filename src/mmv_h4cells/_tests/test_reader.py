import numpy as np

from pathlib import Path
import csv
from aicsimageio.writers import OmeTiffWriter
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
    OmeTiffWriter.save(data, my_test_file, dim_order_out="YX")
    
    reader = napari_get_reader(my_test_file)
    assert callable(reader)


def test_get_reader_pass():
    no_file = Path("fake.file")
    reader = napari_get_reader(no_file)
    assert reader is None
