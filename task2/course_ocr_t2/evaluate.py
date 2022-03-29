from pathlib import Path

def evaluate(gt_path, pred_path):
    gt = dict()
    with open(gt_path) as gt_f:
        for line in gt_f:
            name, cls = line.strip().split()
            gt[name] = cls
    
    n_good = 0
    n_all = len(gt)
    with open(pred_path) as pred_f:
        for line in pred_f:
            name, cls = line.strip().split()
            if cls == gt[name]:
                n_good += 1
    
    return n_good / n_all


def _run_evaluation():
    base = Path(__file__).absolute().parent.parent
    gt_path = base / 'gt.txt'
    pred_path = base / 'pred.txt'
    score = evaluate(gt_path, pred_path)
    print('Accuracy = {:1.4f}'.format(score))


if __name__ == '__main__':
    _run_evaluation()