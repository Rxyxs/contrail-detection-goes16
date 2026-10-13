import numpy as np
import pytest

from contrails.data import ash_color, to_uint8_record
from contrails.dataset import frame_indices
from contrails.metrics import GlobalDice, global_dice
from contrails.rle import rle_decode, rle_encode


def test_rle_matches_competition_example():
    # '1 3 10 5' -> pixels 1,2,3,10..14, numbered down columns first
    mask = rle_decode("1 3 10 5", (4, 4))
    flat = mask.flatten(order="F")
    assert np.flatnonzero(flat).tolist() == [0, 1, 2, 9, 10, 11, 12, 13]
    assert rle_encode(mask) == "1 3 10 5"


def test_rle_column_major_order():
    mask = np.zeros((3, 3), np.uint8)
    mask[1, 0] = 1  # row 2, col 1 -> pixel 2
    mask[0, 1] = 1  # row 1, col 2 -> pixel 4
    assert rle_encode(mask) == "2 1 4 1"


def test_rle_empty_is_dash():
    assert rle_encode(np.zeros((5, 5))) == "-"
    assert rle_decode("-", (5, 5)).sum() == 0


@pytest.mark.parametrize("seed", range(5))
def test_rle_roundtrip(seed):
    m = (np.random.default_rng(seed).random((32, 32)) > 0.7).astype(np.uint8)
    assert np.array_equal(rle_decode(rle_encode(m), m.shape), m)


def test_global_dice_pools_pixels():
    # Per-image Dice would average 1.0 and 0.0; global Dice weights by pixels.
    pred = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])
    true = np.array([[1, 1, 1, 1], [1, 0, 0, 0]])
    assert global_dice(pred, true) == pytest.approx(2 * 4 / (4 + 5))


def test_meter_agrees_with_direct_dice():
    rng = np.random.default_rng(0)
    prob, true = rng.random((4, 1, 16, 16)), rng.random((4, 1, 16, 16)) > 0.8
    m = GlobalDice(thresholds=[0.5])
    m.update(prob[:2], true[:2])
    m.update(prob[2:], true[2:])
    assert m.scores()[0] == pytest.approx(global_dice(prob > 0.5, true))


def test_ash_color_range_and_shape():
    rng = np.random.default_rng(0)
    b = [rng.uniform(200, 320, (8, 8, 8)) for _ in range(3)]
    img = ash_color(*b)
    assert img.shape == (8, 8, 8, 3) and img.min() >= 0 and img.max() <= 1
    assert to_uint8_record(*b).shape == (8, 8, 8, 3)


def test_frame_indices_centre_on_labelled_frame():
    assert frame_indices(1) == [4]
    assert frame_indices(3) == [3, 4, 5]
    assert 4 in frame_indices(5)
