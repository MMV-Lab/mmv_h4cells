"""Regression test: a long session on a synthetic image.

The complete widget state (arrays, metrics, sets, undo stack, camera, ...) is
compared with a recorded session after every step. It protects behaviour
while the implementation is changed, e.g. to make it faster.

If the behaviour is changed on purpose, record the session again with
    RECORD_SESSION=1 pytest src/mmv_h4cells/_tests/test_session_regression.py
and review the diff of data/session_golden.json.
"""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
from qtpy.QtWidgets import QMessageBox

from mmv_h4cells import CellAnalyzer

GOLDEN = Path(__file__).parent / "data" / "session_golden.json"
RECORD = bool(os.environ.get("RECORD_SESSION"))


def make_image():
    """36 elliptical cells on a grid; deterministic, no random numbers."""
    size, step = 240, 40
    img = np.zeros((size, size), dtype=np.int32)
    ids = sorted(1 + (k * 37 + 11) % 499 for k in range(36))
    assert len(set(ids)) == 36
    yy, xx = np.mgrid[0:size, 0:size]
    n = 0
    for gy in range(6):
        for gx in range(6):
            k = gy * 6 + gx
            cy = gy * step + step // 2 + (k * 5) % 9 - 4
            cx = gx * step + step // 2 + (k * 7) % 9 - 4
            ry = 5 + (k * 3) % 12
            rx = 5 + (k * 5) % 12
            if (gy, gx) == (0, 0):  # touches the image border
                cy, cx = ry - 2, rx - 2
            blob = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
            img[blob & (img == 0)] = ids[n]
            n += 1
    return img, [int(i) for i in ids]


def free_spots(img, h=6, w=8, count=3):
    """Top-left corners of empty rectangles that do not touch any cell."""
    spots = []
    for y in range(0, img.shape[0] - h, 7):
        for x in range(0, img.shape[1] - w, 9):
            if not img[max(y - 2, 0) : y + h + 2, max(x - 2, 0) : x + w + 2].any():
                spots.append((y, x))
    assert len(spots) >= count, "image has too little free space"
    return spots[:: max(1, len(spots) // count)][:count]


def h(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def snap(widget):
    return {
        "layer": h(widget.layer_to_evaluate.data),
        "accepted": h(widget.accepted_cells),
        "rejected": h(widget.rejected_cells),
        "current": h(widget.current_cell_layer.data)
        if widget.current_cell_layer is not None
        else None,
        "metric": [
            [int(a), int(b), [int(x) for x in c]]
            for a, b, c in widget.metric_data
        ],
        "included": sorted(int(i) for i in widget.included),
        "excluded": sorted(int(i) for i in widget.excluded),
        "remaining": sorted(int(i) for i in widget.remaining),
        "undo": [int(i) for i in widget.undo_stack],
        "mean": round(float(widget.mean_size), 3),
        "std": round(float(widget.std_size), 3),
        "next": widget.lineedit_next_id.text(),
        "center": [round(float(x), 2) for x in tuple(widget.viewer.camera.center)[-2:]],
        "bound": int(widget.selfdrawn_lower_bound),
        "undo_enabled": bool(widget.btn_undo.isEnabled()),
        "label": int(widget.current_cell_layer.selected_label)
        if widget.current_cell_layer is not None
        else None,
        "layers": [l.name for l in widget.viewer.layers],
    }


def test_long_session_matches_the_recorded_one(make_napari_viewer, monkeypatch, tmp_path):
    results = []

    def step(name, widget):
        results.append((name, snap(widget)))

    monkeypatch.setattr(QMessageBox, "exec_", lambda *a: 0)
    img, ids = make_image()
    viewer = make_napari_viewer()
    widget = CellAnalyzer(viewer)
    viewer.add_labels(img, name="seg")
    step("loaded", widget)
    widget.start_analysis_on_click()
    step("started", widget)

    for i in range(4):
        widget.include_on_click()
        step(f"include {i}", widget)
    for i in range(3):
        widget.exclude_on_click()
        step(f"exclude {i}", widget)
    for i in range(2):
        widget.undo_on_click()
        step(f"undo {i}", widget)

    rem = sorted(widget.remaining)
    widget.lineedit_include.setText(
        f"{rem[3]}, {rem[7]}, 9999, {ids[0]}, 0, {rem[10]}"
    )
    widget.include_multiple_on_click()
    step("include multiple", widget)
    widget.include_on_click()
    step("include after multiple", widget)

    # draw an own cell in the background
    (y0, x0), (y1, x1), (y2, x2) = free_spots(img)
    widget.draw_own_cell()
    label = widget.current_cell_layer.selected_label
    widget.current_cell_layer.data[y0 : y0 + 6, x0 : x0 + 8] = label
    widget.draw_own_cell()  # confirm
    step("self drawn included", widget)
    widget.include_on_click()
    step("include after self drawn", widget)
    widget.exclude_on_click()
    step("exclude after self drawn", widget)
    for i in range(4):
        widget.undo_on_click()
        step(f"undo {i} (incl. self drawn)", widget)

    # drawing over another cell: overlap -> cancel
    other = [i for i in sorted(widget.remaining)][2]
    ys, xs = np.nonzero(widget.layer_to_evaluate.data == other)
    widget.draw_own_cell()
    label = widget.current_cell_layer.selected_label
    widget.current_cell_layer.data[ys[:5], xs[:5]] = label
    widget.current_cell_layer.data[y1 : y1 + 4, x1 : x1 + 4] = label
    monkeypatch.setattr(QMessageBox, "exec_", lambda *a: QMessageBox.Cancel)
    widget.draw_own_cell()
    step("overlap cancelled", widget)
    monkeypatch.setattr(QMessageBox, "exec_", lambda *a: 0)

    # overlap -> ok -> erase the overlap -> confirm
    widget.draw_own_cell()
    label = widget.current_cell_layer.selected_label
    widget.current_cell_layer.data[ys[:5], xs[:5]] = label
    widget.current_cell_layer.data[y1 : y1 + 4, x1 : x1 + 4] = label
    widget.draw_own_cell()
    step("overlap ok", widget)
    widget.current_cell_layer.data[ys[:5], xs[:5]] = 0
    widget.draw_own_cell()
    step("overlap removed and confirmed", widget)

    # several labels
    widget.draw_own_cell()
    label = widget.current_cell_layer.selected_label
    widget.current_cell_layer.data[y2 : y2 + 4, x2 : x2 + 4] = label
    widget.current_cell_layer.data[y2 + 4 : y2 + 6, x2 + 4 : x2 + 6] = label + 1
    widget.draw_own_cell()
    step("several labels", widget)
    widget.cancel_on_click()
    step("cancelled drawing", widget)

    for kind in ("included", "excluded", "remaining"):
        getattr(widget, f"show_{kind}_on_click")()
        step(f"show {kind}", widget)
        getattr(widget, f"show_{kind}_on_click")()
        step(f"back {kind}", widget)

    # export, import into a fresh widget, continue there
    csv = tmp_path / "golden.csv"
    with patch("mmv_h4cells._widget.save_dialog", return_value=str(csv)):
        widget.export_on_click()
    import zarr

    z = zarr.open(str(csv.with_suffix(".zarr")), mode="r")
    results.append(
        (
            "exported",
            {k: h(z[k][:]) for k in ("accepted_cells", "rejected_cells")}
            | {"metrics": [float(x) for x in z["metrics"][:]]},
        )
    )

    viewer2 = make_napari_viewer()
    widget2 = CellAnalyzer(viewer2)
    with patch("mmv_h4cells._widget.open_dialog", return_value=str(csv)):
        widget2.import_on_click()
    results.append(("imported", snap(widget2) | {"current": None, "label": None, "center": None}))
    widget2.start_analysis_on_click()
    step2 = lambda n: results.append((n, snap(widget2)))
    step2("import: started")
    widget2.include_on_click()
    step2("import: include")
    widget2.undo_on_click()
    step2("import: undo")
    # undo decisions made before the export (excluded cells, self drawn cell)
    for i in range(len(widget2.undo_stack)):
        widget2.undo_on_click()
        step2(f"import: undo old {i}")

    data = [(n, s) for n, s in results]
    if RECORD:
        GOLDEN.write_text(json.dumps(data, indent=0))

    else:
        golden = json.loads(GOLDEN.read_text())
        assert len(golden) == len(data), (len(golden), len(data))
        for (gn, gs), (n, s) in zip(golden, data):
            assert gn == n
            s = json.loads(json.dumps(s))
            for key in gs:
                assert gs[key] == s[key], f"step '{n}': {key}: {gs[key]} != {s[key]}"
