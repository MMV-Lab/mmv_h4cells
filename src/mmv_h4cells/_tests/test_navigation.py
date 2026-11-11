"""Tests for the choice of the next cell (no napari needed)"""

import random

import pytest

from mmv_h4cells._navigation import choose_next_cell, following_id


# --- the choice, as it was implemented inside the widget before --------------


def reference_choice(
    remaining, given_id, computed_id, last_evaluated_id, ignore_jump_back
):
    next_lower = next_higher = None
    if given_id is not None:
        next_lower = max([i for i in remaining if i < given_id], default=None)
        next_higher = min([i for i in remaining if i > given_id], default=None)
    jumped_back = False
    if given_id is None:
        next_id = computed_id
    elif given_id == computed_id:
        if computed_id < last_evaluated_id and not ignore_jump_back:
            jumped_back = True
        next_id = computed_id
    elif given_id != computed_id and given_id in remaining:
        next_id = given_id
    elif given_id > computed_id:
        next_id = next_lower
    else:
        if given_id > last_evaluated_id:
            next_id = next_lower if next_lower is not None else computed_id
        elif computed_id < last_evaluated_id and not ignore_jump_back:
            next_id = computed_id
        else:
            next_id = next_lower if next_lower is not None else next_higher
    if next_id is None:
        next_id = computed_id
    return next_id, jumped_back


def reference_following(remaining, next_id):
    if len(remaining) > 1:
        candidate_ids = sorted([i for i in remaining if i > next_id])
        if len(candidate_ids) > 0:
            return candidate_ids[0]
        return min(remaining)
    return None


def test_choice_matches_the_previous_implementation():
    rng = random.Random(1234)
    for _ in range(20000):
        remaining = set(rng.sample(range(1, 40), rng.randint(1, 15)))
        given = rng.choice([None] + list(range(0, 45)))
        computed = rng.randint(0, 44)
        last = rng.randint(0, 44)
        ignore = rng.random() < 0.5
        assert choose_next_cell(
            remaining, given, computed, last, ignore
        ) == reference_choice(remaining, given, computed, last, ignore), (
            remaining,
            given,
            computed,
            last,
            ignore,
        )


def test_following_id_matches_the_previous_implementation():
    rng = random.Random(99)
    for _ in range(5000):
        remaining = set(rng.sample(range(1, 40), rng.randint(0, 12)))
        cell_id = rng.randint(0, 44)
        assert following_id(remaining, cell_id) == reference_following(
            remaining, cell_id
        )


# --- the cases, described ------------------------------------------------------


@pytest.mark.parametrize(
    "given, computed, last, ignore, expected, jumped",
    [
        # nothing typed: use the computed id
        (None, 5, 2, False, 5, False),
        # typed id is the computed one: unchanged
        (5, 5, 2, False, 5, False),
        # ... but it is smaller than the last evaluated: dataset is finished
        (5, 5, 9, False, 5, True),
        (5, 5, 9, True, 5, False),
        # typed id exists: show it
        (8, 5, 2, False, 8, False),
        # typed id is larger and does not exist: the next lower one
        (7, 5, 2, False, 5, False),
        # typed id is smaller than the computed one and does not exist,
        # but larger than the last evaluated: the next lower remaining one
        (4, 5, 2, False, 3, False),
        # no lower id remaining: the computed one
        (1, 5, 0, False, 5, False),
    ],
)
def test_choose_next_cell(given, computed, last, ignore, expected, jumped):
    remaining = {3, 5, 8, 10}
    assert choose_next_cell(remaining, given, computed, last, ignore) == (
        expected,
        jumped,
    )


def test_a_cell_is_always_chosen_even_without_lower_ids():
    # id is smaller than everything remaining and than the last evaluated
    assert choose_next_cell({10, 20}, 1, 15, 30, False) == (15, False)
    assert choose_next_cell({10, 20}, 1, 15, 30, True) == (10, False)


@pytest.mark.parametrize(
    "remaining, cell_id, expected",
    [
        ({1, 3, 5}, 1, 3),  # the next higher one
        ({1, 3, 5}, 2, 3),  # the cell does not have to be remaining itself
        ({1, 3, 5}, 5, 1),  # wraps around to the lowest
        ({1, 3, 5}, 9, 1),
        ({7}, 7, None),  # nothing follows the only remaining cell
        (set(), 7, None),
    ],
)
def test_following_id(remaining, cell_id, expected):
    assert following_id(remaining, cell_id) == expected
