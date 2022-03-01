import json
from pathlib import Path
from typing import Dict
import numpy as np
from shapely import geometry


def iou_relative_quads(quad1, quad2):
    """
    Вычисляет Intersection over Union между двумя четырехугольниками.
    Четырехугольники должны быть представлены в относительном масштабе [0..1].
    В вычислении IoU учитывается только площадь в пределах значений [0..1].
    """
    quad1 = geometry.Polygon(quad1)
    quad2 = geometry.Polygon(quad2)

    # IoU вычисляется только в рамках кадра
    frame = geometry.Polygon([
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0]
    ])
    if not (quad1.is_valid and quad2.is_valid):
        return 0.0
    try:
        return (quad1 & quad2 & frame).area  / ((quad1 | quad2) & frame).area
    except Exception as exc:
        return 0.0

def dump_results_dict(unique_key2quad_dict: Dict, filename: Path):
    """
    Сохраняет значения словаря {"ключ": четырехугольник} в файл.
    """
    data = dict(
        (k, np.array(v).tolist())
        for (k, v) in unique_key2quad_dict.items()
    )
    with open(filename, 'w') as f:
        f.write(json.dumps(data))


def read_results_dict(filename: Path):
    """
    Читает значения словаря {"ключ": четырехугольник} из файла.
    """
    with open(filename, 'r') as f:
        return json.loads(f.read())


def measure_crop_accuracy(
    pred_filename: Path, gt_filename: Path,
    assert_same_keys:bool = False, iou_thr=0.95
    ):
    pred = read_results_dict(pred_filename)
    gt = read_results_dict(gt_filename)
    if assert_same_keys:
        assert set(pred.keys()) == set(gt.keys()), "Keys mismatch in gt/pred dicts"

    ious = []
    for key, gt_quad in gt.items():
        if key in pred:
            pred_quad = pred[key]
            ious.append(iou_relative_quads(gt_quad, pred_quad))
        else:
            ious.append(0)
    ious = np.array(ious)
    accuracy = (ious > iou_thr).astype(int).sum() / len(ious)
    return accuracy


def _run_evaluation():
    base = Path(__file__).absolute().parent.parent
    gt_path = base / 'gt.json'
    pred_path = base / 'pred.json'
    score = measure_crop_accuracy(pred_path, gt_path)
    print("Accuracy[IoU>0.95] = {:1.4f}".format(score))


if __name__ == "__main__":
    _run_evaluation()
