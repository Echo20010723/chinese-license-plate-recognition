import pytest

from parse_tool import parse_filename

NAME = "025-90_85-10&20_110&60-110&60_10&60_10&20_110&20-0_0_24_25_26_27_28-100-5.jpg"


def test_valid_annotation_and_corner_order():
    result = parse_filename(NAME)
    assert result["plate_code"] == "皖A01234"
    assert result["bbox"] == [10, 20, 110, 60]
    assert result["four_points"] == [(10, 20), (110, 20), (110, 60), (10, 60)]
    assert (result["h_tilt"], result["v_tilt"], result["scale"]) == (0, -5, 0.025)


def test_eight_character_annotation_and_windows_path():
    name = NAME.replace("_27_28-100", "_27_28_29-100")
    assert parse_filename("C:\\data\\" + name)["plate_code"] == "皖A012345"


@pytest.mark.parametrize("name", [
    "bad.jpg", NAME.replace("0_0_24", "34_0_24"),
    NAME.replace("110&60-", "10&20-", 1),
    NAME.replace("10&20_110&20", "10&20&30_110&20"),
    NAME.replace("025-", "text-"),
])
def test_invalid_annotation_is_rejected(name):
    with pytest.raises(ValueError):
        parse_filename(name)
