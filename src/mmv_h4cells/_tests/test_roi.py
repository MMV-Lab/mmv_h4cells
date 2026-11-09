import numpy as np

from mmv_h4cells._roi import analyse_roi


def _run(data, y, x, threshold):
    worker = analyse_roi(data, y, x, threshold, ("a.csv", "a.tiff"))
    return worker.work()


def _make_data():
    data = np.zeros((10, 10), dtype=np.int32)
    data[0:2, 0:2] = 1  # outside ROI
    data[4:7, 4:7] = 2  # inside ROI, 9 px
    data[8, 4] = 3  # inside ROI, 1 px (below threshold)
    return data


def test_analyse_roi_does_not_modify_input():
    data = _make_data()
    original = data.copy()

    _run(data, (3, 10), (3, 10), 1)

    np.testing.assert_array_equal(data, original)


def test_analyse_roi_returns_filtered_copy():
    data = _make_data()

    result, df, paths, threshold = _run(data, (3, 10), (3, 10), 1)

    assert result is not data
    assert set(np.unique(result)) == {0, 2}
    assert df["id"].tolist() == [2]
    assert df["count [px]"].tolist() == [9]
    assert paths == ("a.csv", "a.tiff")
    assert threshold == 1
