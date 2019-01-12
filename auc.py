# -*- coding:utf-8 -*-

'''
Author:Samuel Chan
This file contain code to calculate the auc score.
'''

import sys
from constants import *
import numpy as np
import pandas as pd

def auc(labels, predicted_ctr):
    i_sorted = sorted(range(len(predicted_ctr)), key=lambda i: predicted_ctr[i], reverse=True)
    auc_temp = 0.0
    tp = 0.0
    fp = 0.0
    tp_pre = 0.0
    fp_pre = 0.0
    last_value = predicted_ctr[i_sorted[0]]
    for i in range(len(labels)):
        if labels[i_sorted[i]] > 0:
            tp += 1
        else:
            fp += 1
        if last_value != predicted_ctr[i_sorted[i]]:
            auc_temp += (tp + tp_pre) * (fp - fp_pre) / 2.0
            tp_pre = tp
            fp_pre = fp
            last_value = predicted_ctr[i_sorted[i]]
    auc_temp += (tp + tp_pre) * (fp - fp_pre) / 2.0
    return auc_temp / (tp * fp)

def scoreClickAUC(num_clicks, num_impressions, predicted_ctr):
    """
    Calculates the area under the ROC curve (AUC) for click rates

    Parameters
    ----------
    num_clicks : a list containing the number of clicks

    num_impressions : a list containing the number of impressions

    predicted_ctr : a list containing the predicted click-through rates

    Returns
    -------
    auc : the area under the ROC curve (AUC) for click rates
    """
    i_sorted = sorted(range(len(predicted_ctr)), key=lambda i: predicted_ctr[i],
                      reverse=True)
    auc_temp = 0.0
    click_sum = 0.0
    old_click_sum = 0.0
    no_click = 0.0
    no_click_sum = 0.0

    # treat all instances with the same predicted_ctr as coming from the
    # same bucket
    last_ctr = predicted_ctr[i_sorted[0]] + 1.0

    for i in range(len(predicted_ctr)):
        if last_ctr != predicted_ctr[i_sorted[i]]:
            auc_temp += (click_sum + old_click_sum) * no_click / 2.0
            old_click_sum = click_sum
            no_click = 0.0
            last_ctr = predicted_ctr[i_sorted[i]]
        no_click += num_impressions[i_sorted[i]] - num_clicks[i_sorted[i]]
        no_click_sum += num_impressions[i_sorted[i]] - num_clicks[i_sorted[i]]
        click_sum += num_clicks[i_sorted[i]]
    auc_temp += (click_sum + old_click_sum) * no_click / 2.0
    auc = auc_temp / (click_sum * no_click_sum)
    return auc

def load_click_imp(path):
    print('loading ' + path)
    data = pd.read_csv(path)
    clicks = data['clicks'].values
    imps = data['impressions'].values

    return clicks, imps


def load_preds(path):
    print('loading ' + path)
    data = pd.read_csv(path, header=None, dtype=np.float)
    preds = data[0].values

    return preds

def ensemble(path_result):
    paths_preds = ['deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2018-12-31 11:40:59.h5',  # auc: 0.774931 0.785993
                   'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2018-12-31 22:51:48.h5',  # auc: 0.774124 0.784411
                   'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-01 11:52:39.h5',  # auc: 0.773789 0.784705
                   # 'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-01 20:34:54.h5',  # auc: 0.772749 0.782517
                   'deep_fm_combined-ep006-loss0.127-val_loss0.127-bs18000-ee8-hz[512, 256, 256, 256, 128]-l2l1e-06-l2e1e-06-l2d0.001-t2019-01-02 10:21:56.h5',  # auc: 0.775915 0.786094
                    ]
    preds = np.zeros(NUM_TEST)

    # just average
    for path in paths_preds:
        preds += load_preds('data/' + path + '.csv')
    preds = preds / len(paths_preds)

    # writing preds to csv
    print('writing ......')
    with open(path_result, 'w') as fw:
        for i in range(len(preds)):
            # if i % 10000 == 0:
            #     print('pred: %f' % (preds[i]))
            to_write = str(preds[i]) + '\n'
            fw.write(to_write)
        fw.close()


if __name__ == "__main__":
    # n = 1 2 3
    n = '1'
    path_result = 'data/周栋梁+result+'+n+'.csv'  # 0.787509 786970
    ensemble(path_result)
    preds = load_preds(path_result)
    path_labels = PATH_SOLUTION

    clicks, imps = load_click_imp(path_labels)

    print('calculating auc ......')
    AUC = scoreClickAUC(clicks, imps, preds)
    print('scoreClickAUC: %f' % AUC)
