from pathlib import Path

import numpy as np
import pytest
import torch

from recognition.inference import crop_bounding_box, load_recognizer, recognise_plate

ROOT = Path(__file__).resolve().parents[1]


def test_inclusive_crop_and_clipping():
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    np.testing.assert_array_equal(crop_bounding_box(image, [-2, 1, 3, 8]), image[1:6, :4])


@pytest.mark.parametrize("box", [[8, 0, 10, 3], [4, 0, 2, 3], [0, 0, float('nan'), 2]])
def test_invalid_crop(box):
    with pytest.raises(ValueError):
        crop_bounding_box(np.zeros((6, 8, 3), dtype=np.uint8), box)


def test_bundled_checkpoint_loads_and_predicts_seven_characters():
    torch.set_num_threads(1)
    model = load_recognizer(ROOT / "recognition/checkpoints/seven_head_ccpd2019.pt")
    result = recognise_plate(np.zeros((48, 192, 3), dtype=np.uint8), [0, 0, 191, 47], model)
    assert len(result["text"]) == len(result["indices"]) == 7
    assert len(result["character_confidences"]) == 7
    assert all(0 <= p <= 1 for p in result["character_confidences"])


def test_cli_help_does_not_require_dataset_or_models():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, str(ROOT / "predict.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--image" in result.stdout
