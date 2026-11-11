"""Decide which cell is shown next.

These functions do not depend on Qt or napari, so they can be tested directly.
"""

from typing import Iterable, Optional, Tuple


def following_id(remaining: Iterable[int], cell_id: int) -> Optional[int]:
    """The id of the cell to evaluate after cell_id.

    This is the next higher id that is still remaining, wrapping around to the
    lowest one. There is no following cell if at most one cell remains.
    """
    remaining = list(remaining)
    if len(remaining) <= 1:
        return None
    higher = [i for i in remaining if i > cell_id]
    return min(higher) if higher else min(remaining)


def choose_next_cell(
    remaining: Iterable[int],
    given_id: Optional[int],
    computed_id: int,
    last_evaluated_id: int,
    ignore_jump_back: bool = False,
) -> Tuple[int, bool]:
    """Choose the cell to show, taking the id typed by the user into account.

    Parameters
    ----------
    remaining
        Ids of the cells that are not evaluated yet.
    given_id
        The id in the "Next cell" field, None if it is empty or invalid.
    computed_id
        The id that was computed as the next one when the previous cell was
        shown.
    last_evaluated_id
        Id of the cell that was evaluated last (0 if there is none).
    ignore_jump_back
        Do not report jumping back to earlier cells.

    Returns
    -------
    (id of the cell to show, whether the dataset is finished and the cell is an
    earlier one than the last evaluated)
    """
    remaining = list(remaining)
    next_lower = next_higher = None
    if given_id is not None:
        next_lower = max((i for i in remaining if i < given_id), default=None)
        next_higher = min((i for i in remaining if i > given_id), default=None)

    jumped_back = False
    if given_id is None:
        # no valid id given
        next_id = computed_id
    elif given_id == computed_id:
        # id was not changed
        # jump back to lowest unprocessed id
        jumped_back = computed_id < last_evaluated_id and not ignore_jump_back
        next_id = computed_id
    elif given_id in remaining:
        # id was changed and is in remaining
        next_id = given_id
    elif given_id > computed_id:
        # id was increased and is not in remaining
        next_id = next_lower
    elif given_id > last_evaluated_id:
        # id was decreased and is not in remaining
        # maybe others < last < given < computed
        next_id = next_lower if next_lower is not None else computed_id
    elif computed_id < last_evaluated_id and not ignore_jump_back:
        # given < computed (smallest existing) < maybe others < last
        next_id = computed_id
    else:
        # others? < given < others? < computed < last (self drawn)
        # others? < given < others? < last < computed
        next_id = next_lower if next_lower is not None else next_higher

    if next_id is None:
        next_id = computed_id
    return next_id, jumped_back
