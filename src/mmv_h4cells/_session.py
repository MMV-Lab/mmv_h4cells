"""The evaluation of the cells of a label image.

This module knows nothing about Qt or napari. The label layers are only used
through their ``data`` (and ``selected_label`` for the current cell layer), so
any object with these attributes works.

Images can be very large (hundreds of megapixels) while a cell only covers a
tiny part of it. All per-cell work is therefore done on the cell's bounding
box instead of the whole image.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Tuple

import numpy as np
from scipy import ndimage

from mmv_h4cells._navigation import choose_next_cell, following_id

logger = logging.getLogger(__name__)

# Label images with larger ids than this are not indexed by bounding box
# (the lookup then searches the arrays for each cell)
MAX_CACHED_LABEL = 10_000_000

# Number of self drawn cells that still have to fit when the labels are stored
# with a smaller integer type (see Session.use_compact_label_dtype)
LABEL_HEADROOM = 256


class Status(Enum):
    """What happened when a cell was included or excluded."""

    INCLUDED = "included"
    EXCLUDED = "excluded"
    NO_REMAINING = "no cells remain"
    OVERLAP = "the cell overlaps with other cells"
    EMPTY = "the current cell layer is empty"
    MULTIPLE_LABELS = "the current cell layer has more than one label"


@dataclass
class Result:
    status: Status
    cell_id: Optional[int] = None
    overlap: set = field(default_factory=set)


@dataclass
class UndoResult:
    cell_id: int
    selfdrawn: bool  # the cell was drawn by the user
    layer_changed: bool  # pixels of the working layer were changed


class Session:
    """The cells of a label layer and what was decided about them."""

    def __init__(self):
        # label layer with the remaining and the included cells
        self.layer_to_evaluate = None
        # label layer consisting of the current cell
        self.current_cell_layer = None
        # label images of all accepted and all rejected cells
        self.accepted_cells: Optional[np.ndarray] = None
        self.rejected_cells: Optional[np.ndarray] = None
        # (id, size, centroid) of the included cells
        # (adjust if more metrics need to be saved)
        self.metric_data: List[Tuple[int, int, Tuple[int, ...]]] = []
        self.mean_size: float = 0  # mean size of all included cells
        self.std_size: float = 0  # standard deviation of the size
        self.remaining: Set[int] = set()
        self.included: Set[int] = set()
        self.excluded: Set[int] = set()
        self.undo_stack: List[int] = []  # ids in the order of the decisions
        # computed id of the next cell to evaluate (None: no next cell)
        self.next_id: Optional[int] = None
        # self drawn cells get ids from this one on
        self.selfdrawn_lower_bound: Optional[int] = None
        # bounding box of every cell (computed lazily, see get_boxes)
        self._boxes = None
        # (id, box, mask) of the cell shown in the current cell layer
        self._displayed = None

    # --- the label layer ----------------------------------------------------

    def has_session_state(self) -> bool:
        """Whether cells were already evaluated (e.g. an imported session)."""
        return bool(self.metric_data or self.excluded or self.undo_stack)

    def reset(self):
        """Forget the label layer and all evaluation state."""
        self.layer_to_evaluate = None
        self._boxes = None
        self.accepted_cells = None
        self.rejected_cells = None
        self.metric_data = []
        self.mean_size = 0
        self.std_size = 0
        self.remaining = set()
        self.included = set()
        self.excluded = set()
        self.undo_stack = []
        self.next_id = None
        self.selfdrawn_lower_bound = None

    def set_layer(self, layer):
        """Evaluate the cells of a label layer."""
        layer_changed = layer is not self.layer_to_evaluate
        self.layer_to_evaluate = layer
        self._boxes = None
        unique_ids = np.unique(self.layer_to_evaluate.data)
        logger.debug(f"{len(unique_ids)} unique ids found")
        if self.selfdrawn_lower_bound is None or (
            layer_changed and not self.has_session_state()
        ):
            # ids of self drawn cells must not collide with the ids of the
            # layer, so the bound belongs to the layer it was computed from
            self.selfdrawn_lower_bound = int(max(unique_ids)) + 1

        if not self.has_session_state():
            self.accepted_cells = np.zeros_like(self.layer_to_evaluate.data)
            self.rejected_cells = np.zeros_like(self.layer_to_evaluate.data)
            self.remaining = set(unique_ids) - {0}
        self.set_next_id(min(self.remaining) if self.remaining else None)
        logger.debug(f"Selfdrawn lower bound: {self.selfdrawn_lower_bound}")

    def set_current_cell_layer(self, layer):
        self.current_cell_layer = layer
        self._displayed = None

    def load_state(
        self,
        accepted_cells,
        rejected_cells,
        metric_data,
        metrics,
        undo_stack,
        selfdrawn_lower_bound,
    ):
        """Continue an exported session; call set_layer with its layer next."""
        self.selfdrawn_lower_bound = selfdrawn_lower_bound
        self.accepted_cells = accepted_cells
        self.rejected_cells = rejected_cells
        self.mean_size, self.std_size = metrics
        self.undo_stack = undo_stack.tolist()
        self.included = set(np.unique(accepted_cells)) - {0}
        self.excluded = set(np.unique(rejected_cells)) - {0}
        self.metric_data = metric_data

    def update_remaining(self):
        """The cells of the layer that are not evaluated yet."""
        unique_ids = np.unique(self.layer_to_evaluate.data)
        self.remaining = set(unique_ids) - (
            self.included | self.excluded | {0}
        )
        self.set_next_id(min(self.remaining) if self.remaining else None)

    def sort_metric_data(self):
        self.metric_data = sorted(self.metric_data, key=lambda x: x[0])

    # --- bounding boxes -----------------------------------------------------

    def invalidate_boxes(self, event=None):
        """Forget the boxes, e.g. if the label layer was changed.

        Code that modifies the array of the layer in place (without napari
        noticing) has to call this afterwards.
        """
        self._boxes = None

    @staticmethod
    def box_from_coords(coords) -> Tuple[slice, ...]:
        """Bounding box (tuple of slices) of the result of np.nonzero."""
        return tuple(slice(int(c.min()), int(c.max()) + 1) for c in coords)

    @staticmethod
    def union_boxes(box_a, box_b) -> Tuple[slice, ...]:
        return tuple(
            slice(min(a.start, b.start), max(a.stop, b.stop))
            for a, b in zip(box_a, box_b)
        )

    def get_boxes(self) -> dict:
        """Bounding box of each cell id, computed once per label layer.

        Boxes stay valid while cells move between the working layer and the
        accepted / rejected masks, as the pixels stay inside their box.
        """
        if self._boxes is not None:
            return self._boxes
        boxes = {}
        if self.layer_to_evaluate is not None:
            arrays = [self.layer_to_evaluate.data]
            if self.rejected_cells is not None:
                arrays.append(self.rejected_cells)
            for array in arrays:
                if array.size == 0 or int(array.max()) > MAX_CACHED_LABEL:
                    continue  # cell_box falls back to searching
                for label, box in enumerate(
                    ndimage.find_objects(array), start=1
                ):
                    if box is not None and label not in boxes:
                        boxes[label] = box
        self._boxes = boxes
        return boxes

    def cell_box(self, cell_id: int):
        """Bounding box of a cell, or None if the cell cannot be found."""
        boxes = self.get_boxes()
        box = boxes.get(cell_id)
        if box is not None:
            return box
        for array in (
            self.layer_to_evaluate.data,
            self.rejected_cells,
            self.accepted_cells,
        ):
            if array is None:
                continue
            coords = np.nonzero(array == cell_id)
            if len(coords[0]):
                box = self.box_from_coords(coords)
                boxes[cell_id] = box
                return box
        return None

    def remember_box(self, cell_id: int, box):
        """Make sure the cached box of a cell covers the given box."""
        boxes = self.get_boxes()
        known = boxes.get(cell_id)
        boxes[cell_id] = box if known is None else self.union_boxes(known, box)

    @staticmethod
    def centroid_in_box(mask, box) -> tuple:
        """Centroid of a mask that covers the given box, in image coordinates."""
        return tuple(
            float(c) + s.start
            for c, s in zip(ndimage.center_of_mass(mask), box)
        )

    # --- the label dtype ----------------------------------------------------
    # napari needs to convert labels wider than 16 bit before it can show them
    # (about 200 ms for a 6000 x 6000 image, for every refresh). 16 bit labels
    # are shown directly, and all label arrays need half the memory.

    def convert_label_arrays(self, dtype):
        """Store all label arrays with another integer type."""
        self.layer_to_evaluate.data = self.layer_to_evaluate.data.astype(dtype)
        for name in ("accepted_cells", "rejected_cells"):
            array = getattr(self, name)
            if array is not None:
                setattr(self, name, array.astype(dtype))
        if self.current_cell_layer is not None:
            self.current_cell_layer.data = self.current_cell_layer.data.astype(
                dtype
            )

    def use_compact_label_dtype(self):
        """Store the labels as uint16 if all ids fit (with some headroom)."""
        data = self.layer_to_evaluate.data
        if data.dtype.itemsize <= 2 or not np.issubdtype(
            data.dtype, np.integer
        ):
            return
        arrays = [
            array
            for array in (
                data,
                self.accepted_cells,
                self.rejected_cells,
            )
            if array is not None and array.size
        ]
        highest = max(int(array.max()) for array in arrays)
        lowest = min(int(array.min()) for array in arrays)
        if lowest < 0 or highest + LABEL_HEADROOM > np.iinfo(np.uint16).max:
            return
        logger.debug(f"Storing labels as uint16 (was {data.dtype})")
        self.convert_label_arrays(np.uint16)

    def ensure_label_capacity(self, label: int):
        """Make sure the label arrays can store the given label."""
        dtype = self.layer_to_evaluate.data.dtype
        if label <= np.iinfo(dtype).max:
            return
        for wider in (np.int32, np.int64):
            if label <= np.iinfo(wider).max:
                logger.debug(f"Storing labels as {wider.__name__}")
                self.convert_label_arrays(wider)
                return

    # --- the current cell ---------------------------------------------------

    def get_current_cell(self):
        """Labels in the current cell layer and their bounding box.

        Returns
        -------
        (set of int, tuple of slices or None)
            None as box if the layer is empty.
        """
        data = self.current_cell_layer.data
        if self._displayed is not None:
            # Scanning the whole layer is slow for large images. If it still
            # holds exactly what show_cell put there (same number of pixels,
            # and the same content inside the box), nothing else can be in it.
            cell_id, box, mask = self._displayed
            if np.count_nonzero(data) == np.count_nonzero(
                mask
            ) and np.array_equal(data[box], np.where(mask, cell_id, 0)):
                return {cell_id}, box
        coords = np.nonzero(data)
        if len(coords[0]) == 0:
            return set(), None
        labels = {int(v) for v in np.unique(data[coords])}
        return labels, self.box_from_coords(coords)

    def clear_current_cell(self):
        self.current_cell_layer.data[:] = 0
        self._displayed = None

    def show_cell(self, cell_id: int):
        """Put a cell into the current cell layer.

        Returns
        -------
        The centroid of the cell, None if it cannot be found.
        """
        self.clear_current_cell()
        centroid = None
        box = self.cell_box(cell_id)
        if box is not None:
            mask = self.layer_to_evaluate.data[box] == cell_id
            self.current_cell_layer.data[box][mask] = cell_id
            centroid = self.centroid_in_box(mask, box)
            self._displayed = (cell_id, box, mask)
        return centroid

    def new_drawn_label(self) -> int:
        """A label for a cell drawn by the user, which no other cell has."""
        new_label = 1 + max(
            int(np.max(self.accepted_cells)),
            int(np.max(self.rejected_cells)),
            int(np.max(self.layer_to_evaluate.data)),
        )
        self.ensure_label_capacity(new_label)
        return new_label

    def get_overlap(self, current=None):
        """
        Returns the indices of the pixels of the current cell that overlap with other cells.

        Parameters:
        -----------
        current: tuple, optional
            Result of get_current_cell, if it was already computed.

        Returns:
        --------
        overlap: set
            A set of tuples containing the indices of the overlapping pixels.
        """
        logger.debug("Calculating overlap...")
        _, box = current if current is not None else self.get_current_cell()
        if box is None:
            return set()
        current_cell = self.current_cell_layer.data[box]
        other_cells = self.layer_to_evaluate.data[box]
        id_ = self.current_cell_layer.selected_label
        overlapping = (
            (current_cell != 0) & (other_cells != 0) & (other_cells != id_)
        )
        offset = [s.start for s in box]
        return {
            tuple(int(c) + o for c, o in zip(coords, offset))
            for coords in np.argwhere(overlapping)
        }

    # --- the decisions ------------------------------------------------------

    def start(self, start_id: int) -> int:
        """Start with a cell, or the closest remaining one.

        Returns
        -------
        The id of the cell to start with.
        """
        if start_id not in self.remaining:
            logger.warning("Start id not in remaining ids")
            lower_ids = {value for value in self.remaining if value < start_id}
            if len(lower_ids) > 0:
                logger.info("Using lower id")
                start_id = max(lower_ids)
            else:
                logger.info("Using lowest remaining id")
                start_id = min(self.remaining)
        self.set_next_after(start_id)
        return start_id

    def include_current(self, self_drawn: bool = False) -> Result:
        """Include the cell in the current cell layer.

        Parameters
        ----------
        self_drawn
            Whether the cell was drawn by the user.
        """
        logger.debug("Including cell...")
        if len(self.remaining) < 1:
            logger.info("No cell to include")
            return Result(Status.NO_REMAINING)

        current = self.get_current_cell()
        overlap = self.get_overlap(current)
        if overlap:
            return Result(Status.OVERLAP, overlap=overlap)

        labels, box = current
        if not labels:
            logger.debug("Current cell layer is empty")
            return Result(Status.EMPTY)
        if len(labels) > 1:
            logger.debug("Multiple ids in current cell layer")
            return Result(Status.MULTIPLE_LABELS)
        id_ = max(labels)
        region = self.current_cell_layer.data[box]
        self.include(id_, region, not self_drawn, box=box)
        if self_drawn:
            self.layer_to_evaluate.data[box] += region

        self.undo_stack.append(id_)
        return Result(Status.INCLUDED, cell_id=id_)

    def exclude_current(self) -> Result:
        """Exclude the cell in the current cell layer."""
        logger.debug("Excluding cell...")
        if len(self.remaining) < 1:
            logger.info("No cell to exclude")
            return Result(Status.NO_REMAINING)

        labels, box = self.get_current_cell()
        if not labels:
            logger.debug("Current cell layer is empty")
            return Result(Status.EMPTY)
        if len(labels) > 1:
            logger.debug("Multiple ids in current cell layer")
            return Result(Status.MULTIPLE_LABELS)

        current_id = max(labels)
        self.excluded.add(current_id)
        self.remaining.remove(current_id)
        self.undo_stack.append(current_id)

        self.remember_box(current_id, box)
        mask = self.current_cell_layer.data[box] == current_id
        self.rejected_cells[box][mask] = current_id
        self.layer_to_evaluate.data[box][mask] = 0
        return Result(Status.EXCLUDED, cell_id=current_id)

    def undo(self) -> Optional[UndoResult]:
        """Take back the last decision; None if there is nothing to undo."""
        logger.debug("Undoing last action...")
        if len(self.undo_stack) == 0:
            return None
        logger.debug(f"Last evaluated: {self.undo_stack[-1]}")
        last_evaluated = self.undo_stack.pop(-1)
        box = self.cell_box(last_evaluated)
        selfdrawn = last_evaluated >= self.selfdrawn_lower_bound
        if not selfdrawn:
            logger.debug("Adding cell back to remaining")
            self.remaining.add(last_evaluated)
        layer_changed = False
        if last_evaluated in self.included:
            logger.debug("Removing cell from accepted")
            # metric_data is not ordered by inclusion (export sorts it by id)
            self.metric_data = [
                entry
                for entry in self.metric_data
                if entry[0] != last_evaluated
            ]
            self.included.remove(last_evaluated)
            if box is not None:
                accepted = self.accepted_cells[box]
                accepted[accepted == last_evaluated] = 0
            if selfdrawn and box is not None:
                working = self.layer_to_evaluate.data[box]
                working[working == last_evaluated] = 0
                layer_changed = True
        else:
            self.excluded.remove(last_evaluated)
            if box is not None:
                rejected = self.rejected_cells[box]
                mask = rejected == last_evaluated
                self.layer_to_evaluate.data[box][mask] = last_evaluated
                rejected[mask] = 0
                layer_changed = True
        self.calculate_metrics()
        return UndoResult(last_evaluated, selfdrawn, layer_changed)

    def include_multiple(
        self, ids: List[int]
    ) -> Tuple[Set[int], Set[int], Set[int], Set[int]]:
        """Include the cells with the given ids.

        Returns
        -------
        included, ignored, overlapped, faulty : sets of int
            Cells that were included, were ignored because they are already
            evaluated, were not included because they overlap with accepted
            cells, and ids that do not exist.
        """
        logger.debug("Including multiple cells...")
        included = set()
        ignored = set()
        overlapped = set()
        faulty = set()
        # the working layer contains the remaining and the included cells
        existing_ids = self.remaining | self.included
        for val in ids:
            if val == 0:
                continue
            if val not in existing_ids:
                faulty.add(val)
                continue
            if val not in self.remaining:
                ignored.add(val)
                continue
            box = self.cell_box(val)
            if box is None:
                faulty.add(val)
                continue
            mask = self.layer_to_evaluate.data[box] == val
            if self.accepted_cells[box][mask].any():
                overlapped.add(val)
                continue
            data_array = np.zeros(mask.shape, dtype=self.accepted_cells.dtype)
            data_array[mask] = val
            self.include(val, data_array, box=box)
            included.add(val)
            self.undo_stack.append(val)
        logger.debug("Multiple cells evaluated")
        return included, ignored, overlapped, faulty

    def include(
        self,
        id_: int,
        data_array: np.ndarray,
        remove_from_remaining: bool = True,
        box=None,
    ):
        """Include a cell.

        data_array is the cell's pixels: the whole image, or, if box (a tuple
        of slices) is given, only the region of the image inside that box.
        """
        logger.debug("Including cell...")
        if box is None:
            self.accepted_cells += data_array
        else:
            self.accepted_cells[box] += data_array
            self.remember_box(id_, box)
        if remove_from_remaining:
            self.remaining.remove(id_)
        self.included.add(id_)

        self.add_cell_to_accepted(id_, data_array, box)

    def add_cell_to_accepted(self, cell_id: int, data: np.ndarray, box=None):
        """Add a cell to the accepted ones.

        If box is given, data only covers that part of the image.
        """
        logger.debug("Adding cell to list of accepted...")
        self.included.add(cell_id)
        centroid = ndimage.center_of_mass(data)
        offset = [0] * len(centroid) if box is None else [s.start for s in box]
        centroid = tuple(int(c + o) for c, o in zip(centroid, offset))
        self.metric_data.append(  # TODO
            (
                cell_id,
                np.count_nonzero(data),
                centroid,
            )
        )

        self.calculate_metrics()

    def calculate_metrics(self):
        logger.debug("Calculating metrics...")
        sizes = [t[1] for t in self.metric_data]
        if len(sizes):
            self.mean_size = np.round(np.mean(sizes), 3)
            self.std_size = np.round(np.std(sizes), 3)
        else:
            self.mean_size = 0
            self.std_size = 0

    # --- the next cell ------------------------------------------------------

    def set_next_id(self, next_id: Optional[int]):
        self.next_id = None if next_id is None else int(next_id)

    def set_next_after(self, cell_id: int):
        """The cell after the given one is the next one to evaluate."""
        self.set_next_id(following_id(self.remaining, cell_id))

    def choose_next(
        self, given_id: Optional[int], ignore_jump_back: bool = False
    ) -> Tuple[int, bool]:
        """The cell to show next, taking the id given by the user into account.

        Returns
        -------
        (id of the cell, whether the dataset is finished and the cell is an
        earlier one than the last evaluated)
        """
        last_evaluated_id = self.undo_stack[-1] if self.undo_stack else 0
        # next_id is None when only one cell was left
        computed_id = (
            self.next_id if self.next_id is not None else min(self.remaining)
        )
        next_id, jumped_back = choose_next_cell(
            self.remaining,
            given_id,
            computed_id,
            last_evaluated_id,
            ignore_jump_back,
        )
        logger.debug(
            f"Next cell: {next_id} (given {given_id}, computed {computed_id}, "
            f"last evaluated {last_evaluated_id})"
        )
        return next_id, jumped_back

    # --- the evaluated cells ------------------------------------------------

    def cells_to_show(self, kind: str) -> np.ndarray:
        """Label image of the "included", "excluded" or "remaining" cells."""
        if kind == "included":
            return self.accepted_cells
        if kind == "excluded":
            return self.rejected_cells
        data = np.zeros_like(self.layer_to_evaluate.data)
        for cell_id in self.remaining:
            box = self.cell_box(cell_id)
            if box is not None:
                cell = self.layer_to_evaluate.data[box] == cell_id
                data[box][cell] = cell_id
        return data
