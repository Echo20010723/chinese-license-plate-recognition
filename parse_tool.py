"""Parse the seven annotation fields embedded in a CCPD image filename."""
from __future__ import annotations

import os


PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
             "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
             "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
             "新", "警", "学", "O"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
             'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
       'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
       '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def parse_filename(file_name: str) -> dict[str, object]:
    stem = os.path.splitext(os.path.basename(str(file_name).replace("\\", "/")))[0]
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"expected 7 filename fields, got {len(parts)}: {file_name}")

    try:
        if not parts[0].isdigit():
            raise ValueError("area field must contain digits")
        scale = float("0." + parts[0])
        h_angle, v_angle = map(int, parts[1].split("_"))

        lt, rb = parts[2].split("_")
        x1, y1 = map(int, lt.split("&"))
        x2, y2 = map(int, rb.split("&"))
        if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError(f"invalid bounding box {(x1, y1, x2, y2)}")

        raw_points = [tuple(map(int, point.split("&"))) for point in parts[3].split("_")]
        if len(raw_points) != 4 or any(len(p) != 2 or min(p) < 0 for p in raw_points):
            raise ValueError(f"expected 4 vertices, got {len(raw_points)}")
        rb_point, lb_point, lt_point, rt_point = raw_points
        four_points = [lt_point, rt_point, rb_point, lb_point]

        char_indices = list(map(int, parts[4].split("_")))
        if len(char_indices) not in (7, 8):
            raise ValueError(f"expected 7 or 8 plate characters, got {len(char_indices)}")
        tables = [PROVINCES, ALPHABETS] + [ADS] * (len(char_indices) - 2)
        if any(not 0 <= index < len(table) for index, table in zip(char_indices, tables)):
            raise ValueError("character index outside the vocabulary")
        plate = PROVINCES[char_indices[0]] + ALPHABETS[char_indices[1]]
        plate += "".join(ADS[index] for index in char_indices[2:])
        brightness = int(parts[5])
        blurriness = int(parts[6])
        if min(brightness, blurriness) < 0:
            raise ValueError("brightness and blur must be nonnegative")
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid CCPD annotation in {file_name}: {exc}") from exc

    return {
        "scale": scale,
        "h_tilt": h_angle - 90,
        "v_tilt": v_angle - 90,
        "bbox": [x1, y1, x2, y2],
        "four_points": four_points,
        "plate_code": plate,
        "province": plate[0],
        "n_chars": len(char_indices),
        "bright": brightness,
        "blur": blurriness,
    }
