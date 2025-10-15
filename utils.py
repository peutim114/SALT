import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve,auc

target = 1
def compute_test(preds, label):
    #print('preds--------------',preds)
    print('label-----------',type(label))
    # fpr, tpr, threshold = roc_curve(preds.cpu(), label.cpu())
    # roc = auc(fpr, tpr)
    roc = roc_auc_score(label,preds)

    preds, label = torch.tensor(preds), torch.tensor(label)
    #print(label)
    TP = TN = FN = FP = 0
    # TP    predict 和 label 同时为1
    TP += ((preds == 1) & (label.data == 1)).sum()
    # TN    predict 和 label 同时为0
    TN += ((preds == 0) & (label.data == 0)).sum()
    # FN    predict 0 label 1
    FN += ((preds == 0) & (label.data == 1)).sum()
    # FP    predict 1 label 0
    FP += ((preds == 1) & (label.data == 0)).sum()
    # print("TP",TP,"TN",TN,"FN",FN,"FP",FP)
    sensitivity = TP * 1.0 / (TP + FN)
    specificity = TN * 1.0 / (TN + FP)
    # print("sensitivity:",sensitivity,"specificity:",specificity)

    if (TP + FN) == 0:  # 说明没有label=1的，即这个batch不含癫痫发作的片段
        pass
    else:
        L1 = TP * 1.0 / (TP + FN)
        L2 = TN * 1.0 / (TN + FP)

  #  F1 = 2 * specificity * sensitivity / (specificity + sensitivity)
   # F1 = 2*TP / (TP+TP  + FP + FN)
    # print("F1:", F1)
    acc = (TP + TN) * 1.0 / (TP + TN + FP + FN)
    Precision =TP/(TP+FP)
    Recall =TP/(TP+FN)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    acc,F1,specificity, sensitivity=acc.numpy(), F1.numpy(), specificity.numpy(), sensitivity.numpy()

    return acc, F1, specificity, sensitivity,roc
def last_relevant_pytorch(output, lengths, batch_first=True):
    lengths = np.array(lengths)
    lengths = torch.from_numpy(lengths)

    # masks of the true seq lengths
    masks = (lengths - 1).reshape(-1, 1).expand(len(lengths), output.size(2))
    time_dimension = 1 if batch_first else 0
    masks = masks.unsqueeze(time_dimension)
    masks = masks.to(output.device)
    last_output = output.gather(time_dimension, masks).squeeze(time_dimension)
    last_output.to(output.device)

    return last_output