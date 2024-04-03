from pathlib import Path
import editdistance

def evaluate(gt_path, pred_path):
    gt = dict()
    with open(gt_path) as gt_f:
        for line in gt_f:
            name, gtruth = line.strip().split()
            gt[name] = gtruth
    
    ed_sum = 0
    len_sum = 0
    with open(pred_path) as pred_f:
        for line in pred_f:
            name, pred = line.strip().split()
            ed = editdistance.eval(pred, gt[name])
            ed_sum += ed
            len_sum += len(gtruth)
    
    return 1 - ed_sum / len_sum


def _run_evaluation():
    base = Path(__file__).absolute().parent.parent
    gt_path = base / 'gt.txt'
    pred_path = base / 'pred.txt'
    score = evaluate(gt_path, pred_path)
    print('Char Accuracy = {:1.4f}'.format(score * 100))


if __name__ == '__main__':
    _run_evaluation()