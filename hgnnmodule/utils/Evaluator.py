import torch as th
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score


class Evaluator:
    def __init__(self, seed):
        self.seed = seed
        self.threshold = None

    def f1_node_classification(self, y_label, y_pred, thresh=0.5):
        if self.threshold:
            thresh = self.threshold
        prec, rec, f1 = self.calc_metrics(y_pred, y_label, thresh)
        return dict(Micro_prec=prec, Micro_rec=rec, Micro_f1=f1)

    def calc_metrics(self, logits, labels, threshold=0.5):
        # Transform labels into one-hot encoded labels
        if len(labels.shape) == 1:
            labels = nn.functional.one_hot(labels)

        preds = logits >= threshold
        num_labels = th.sum(labels)

        # Sum of samples classified as True
        correct_v = labels * preds

        num_correct = th.sum(correct_v)
        num_total = th.sum(preds)

        prec = 100 * num_correct / num_total
        rec = 100 * num_correct / num_labels
        f1 = 2 * prec * rec / (prec + rec)

        return prec, rec, f1

    def fine_tune_threshold(self, preds, labels):
        best_threshold, best_f1 = 0, 0

        for threshold in range(0, 100):
            thresh = threshold / 100
            prec, rec, f1 = self.calc_metrics(preds, labels, thresh)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        self.threshold = best_threshold

    def cal_acc(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def cal_roc_auc(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)
