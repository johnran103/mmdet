import numpy as np

def miou(label, pred, thresh=0.3):

    pred = pred > thresh

    label = label.numpy()
    pred = pred.numpy()

    pred = pred == 1
    label = label == 1

    inter = pred[label].sum()
    union = pred.sum() + label.sum() - inter

    return float(inter)/float(union)
    
