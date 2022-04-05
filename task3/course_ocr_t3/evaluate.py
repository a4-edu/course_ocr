#!/usr/bin/python3
# -*- coding: utf-8 -*-

from shapely.geometry import Polygon
import csv
from pathlib import Path
import os

def calculate_iou(markup, answer):
	sum_iou = 0.0
	for (k, v) in answer.items():
		a = Polygon(v[1])
		b = Polygon(markup[k][1])
		iou = a.intersection(b).area / a.union(b).area
		sum_iou += iou
	return sum_iou / len(markup)
	
	
def read_file(path):
	markup = {}
	with open(path, 'r', encoding='utf-16') as fd:
		rd = csv.reader(fd)
		next(rd)
		for row in rd:
			id = row[0]
			value = row[1]
			points = [(int(row[i]), int(row[i + 1])) for i in range(2, 10, 2)]
			markup[id] = [value, points]
	return markup


def calculate_accuracy(markup, answer):
	counter = 0
	for (k, v) in answer.items():
		markup_value = markup[k][0]
		if markup_value == v[0]:
			counter += 1
	return counter / len(markup)


ACCURACY_WEIGHT = 0.65
IOU_WEIGHT = 0.35


def main():
	base = Path(__file__).absolute().parent.parent
	gt_path = os.path.join(base, "markup.csv")
	answer_path = os.path.join(base, "answer.csv")
	print(f"Checking answer ({answer_path}) against markup({gt_path})")
	markup = read_file(gt_path)
	answer = read_file(answer_path)
	recognition_accuracy = calculate_accuracy(markup, answer)
	print("recognition_accuracy=", recognition_accuracy)
	detection_iou = calculate_iou(markup, answer)
	print("detection_iou=", detection_iou)
	score = ACCURACY_WEIGHT * recognition_accuracy + IOU_WEIGHT * detection_iou
	print("score=", score)


if __name__ == '__main__':
	main()