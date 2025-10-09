from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def C_roc_auc(preds, trues):
    # 确定类别数
    classes = np.unique(trues)
    n_classes = len(classes)

    # 将预测和真实标签二值化
    trues_bin = label_binarize(trues, classes=classes)
    preds_bin = label_binarize(preds, classes=classes)
    
    # 计算每个类别的 ROC 曲线和 AUC 值
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(trues_bin[:, i], preds_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算宏平均 ROC 曲线和 AUC 值
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # 然后在这些点上插值所有 ROC 曲线
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # 最后求平均并计算 AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return roc_auc

def evaluate_classification(preds, trues):
    """
    Evaluate classification metrics based on predictions and true labels.
    
    Args:
    - preds (array-like): Predicted labels or probabilities.
    - trues (array-like): True labels.
    
    Returns:
    - A dictionary containing various classification metrics.
    """
    '''加一个模块把输入展平为一维的
    '''
    # 将输入展平为一维
    preds = np.ravel(preds)
    trues = np.ravel(trues)
    # 如果 preds 是概率，则将其转换为最可能的类别标签
    if preds.ndim > 1 and preds.shape[1] > 1:
        preds_labels = preds.argmax(axis=1)
        print('error')
        exit()

    else:
        preds_labels = preds
    
    # 计算各项指标
    accuracy = accuracy_score(trues, preds_labels)
    precision = precision_score(trues, preds_labels, average='macro')  # 修改为'macro'
    recall = recall_score(trues, preds_labels, average='macro')        # 修改为'macro'
    f1 = f1_score(trues, preds_labels, average='macro')  
    roc_auc = C_roc_auc(preds, trues)

    
    return accuracy,precision,recall,f1,roc_auc
