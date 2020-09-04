# -*- coding: utf-8 -*-
'''
@time: 2019/10/1 10:20

@ author: ys
'''
import torch
import numpy as np
import time,os
from sklearn.metrics import f1_score
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import config
import pandas as pd

weights_file = './evaluation/weights.csv'
mapping_score_file = './evaluation/dx_mapping_scored.csv'

def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

# Load a table with row and column names.
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table)-1
    if num_rows<1:
        raise Exception('The table {} is empty.'.format(table_file))

    num_cols = set(len(table[i])-1 for i in range(num_rows))
    if len(num_cols)!=1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(num_cols)
    if num_cols<1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [int(table[0][j+1]) for j in range(num_rows)]
    cols = [int(table[i+1][0]) for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values

# Load weights.
def load_weights(weight_file, classes):
    # Load the weight matrix.
    rows, cols, values = load_table(weight_file)
    assert(rows == cols)
    num_rows = len(rows)

    # Assign the entries of the weight matrix with rows and columns corresponding to the classes.
    num_classes = len(classes)
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i, a in enumerate(rows):
        if a in classes:
            k = classes.index(a)
            for j, b in enumerate(rows):
                if b in classes:
                    l = classes.index(b)
                    weights[k, l] = values[i, j]

    return weights

scored_classes = pd.read_csv(mapping_score_file)['SNOMED CT Code'].values.tolist()
weights = load_weights(weights_file,scored_classes)

# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py
class FocalLoss2d1(nn.Module):
    def __init__(self, gamma=2, class_weight=None, size_average=True):
        super(FocalLoss2d1, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.class_weight = class_weight

    def forward(self, logit, target, type='sigmoid'):
        target = target.view(-1, 1).long()
        if type=='sigmoid':
            if self.class_weight is None:
                self.class_weight = [1]*2 #[0.5, 0.5]
            prob   = F.sigmoid(logit)
            prob   = prob.view(-1, 1)
            prob   = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)
        elif  type=='softmax':
            B,C,H,W = logit.size()
            if self.class_weight is None:
                self.class_weight =[1]*C #[1/C]*C
            logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob    = F.softmax(logit,1)
            select  = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)
        class_weight = torch.FloatTensor(self.class_weight).cuda().view(-1,1)
        class_weight = torch.gather(self.class_weight, 0, target)
        prob       = (prob*select).sum(1).view(-1,1)
        prob       = torch.clamp(prob,1e-8,1-1e-8)
        batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss
        return loss

class FocalLoss1(nn.Module):

    def __init__(self, focusing_param=2, balance_param=0.25):
        super(FocalLoss, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')#reduction='none', reduce=True
        self.focusing_param = focusing_param
        self.balance_param = balance_param
        self.size_average = True

    def forward(self, output, target):
        # print(output)
        # print(target)
        logpt = self.cerition(output, target)
        # cross_entropy = F.cross_entropy(output, target)
        # cross_entropy_log = torch.log(cross_entropy)
        # logpt = - F.cross_entropy(output, target)
        pt    = torch.exp(logpt)
        focal_loss = ((1 - pt) ** self.focusing_param) * logpt

        balanced_focal_loss = self.balance_param * focal_loss

        if self.size_average:
            loss = balanced_focal_loss.mean()
        else:
            loss = balanced_focal_loss

        return loss

class FocalLoss(nn.Module):

    def __init__(self, gama=10, alpha=0.5, size_average =True):
        super(FocalLoss, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')#reduction='none', reduce=True
        self.gama = gama
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, output, target):
        #logpt = - F.binary_cross_entropy_with_logits(output, target,reduction='mean')#self.cerition(output, target)
        #pt    = torch.exp(logpt)
        p = output.sigmoid()

        focal_loss = -self.alpha*(1-p)**self.gama * p.log()*target - (1-self.alpha)*(p)**self.gama * (1-p).log()*(1-target) #.mean()

        #focal_loss = -((1 - pt) ** self.gama) * logpt
        #balanced_focal_loss = self.balance_param * focal_loss

        if self.size_average:
            loss = focal_loss.mean()
        else:
            loss = focal_loss.sum()

        loss = Variable(loss, requires_grad = True)

        return loss

class FocalLoss2d(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', balance_param=0.25):
        super(FocalLoss2d, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)
        
        weight = Variable(self.weight)
           
        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=weight, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -( (1-pt)**self.gamma ) * logpt
        balanced_focal_loss = self.balance_param * focal_loss


        return balanced_focal_loss

def compute_beta_score(labels, output, beta, num_classes, check_errors=True):

    # Check inputs for errors.
    if check_errors:
        if len(output) != len(labels):
            raise Exception('Numbers of outputs and labels must be the same.')

    # Populate contingency table.
    num_recordings = len(labels)

    fbeta_l = np.zeros(num_classes)
    gbeta_l = np.zeros(num_classes)
    fmeasure_l = np.zeros(num_classes)
    accuracy_l = np.zeros(num_classes)

    f_beta = 0
    g_beta = 0
    f_measure = 0
    accuracy = 0

    # Weight function
    C_l=np.ones(num_classes);

    for j in range(num_classes):
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for i in range(num_recordings):
            
            num_labels = np.sum(labels[i])
        
            if labels[i][j] and output[i][j]:
                tp += 1/num_labels
            elif not labels[i][j] and output[i][j]:
                fp += 1/num_labels
            elif labels[i][j] and not output[i][j]:
                fn += 1/num_labels
            elif not labels[i][j] and not output[i][j]:
                tn += 1/num_labels

        # Summarize contingency table.
        if ((1+beta**2)*tp + (fn*beta**2) + fp):
            fbeta_l[j] = float((1+beta**2)* tp) / float(((1+beta**2)*tp) + (fn*beta**2) + fp)
        else:
            fbeta_l[j] = 1.0

        if (tp + fp + beta * fn):
            gbeta_l[j] = float(tp) / float(tp + fp + beta*fn)
        else:
            gbeta_l[j] = 1.0

        if tp + fp + fn + tn:
            accuracy_l[j] = float(tp + tn) / float(tp + fp + fn + tn)
        else:
            accuracy_l[j] = 1.0

        if 2 * tp + fp + fn:
            fmeasure_l[j] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            fmeasure_l[j] = 1.0


    for i in range(num_classes):
        f_beta += fbeta_l[i]*C_l[i]
        g_beta += gbeta_l[i]*C_l[i]
        f_measure += fmeasure_l[i]*C_l[i]
        accuracy += accuracy_l[i]*C_l[i]


    f_beta = float(f_beta)/float(num_classes)
    g_beta = float(g_beta)/float(num_classes)
    f_measure = float(f_measure)/float(num_classes)
    accuracy = float(accuracy)/float(num_classes)


    return accuracy,f_measure,f_beta,g_beta,compute_challenge_metric(weights,labels,output,scored_classes,[426783006])

# def calc_metric(y_true, y_pre, threshold=0.5):
#     y_true = y_true.cpu().detach().numpy().astype(np.int)
#     y_pre = y_pre.cpu().detach().numpy() > threshold

#     #y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
#     #y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold

#     return compute_beta_score(y_true, y_pre,beta=2,num_classes=config.num_classes)

# Compute modified confusion matrix for multi-class, multi-label tasks.
def compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/normalization

    return A

# Compute the evaluation metric for the Challenge.
def compute_challenge_metric(weights, labels, outputs, classes, normal_class):
    num_recordings, num_classes = np.shape(labels)
    normal_index = 22#classes.index(normal_class)

    # Compute the observed score.
    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the normal class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    inactive_outputs[:, normal_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = float('nan')

    return normalized_score

def calc_metric(y_true, y_pre, threshold=0.5):
    y_true = y_true.cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.cpu().detach().numpy() > threshold

    #y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    #y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold

    return compute_beta_score(y_true, y_pre,beta=2,num_classes=config.num_classes)

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 计算F1score
def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return f1_score(y_true, y_pre)

# 计算F1score
def re_calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.cpu().detach().numpy().astype(np.int)
    # print(y_true.shape)
    y_prob = y_pre.cpu().detach().numpy()
    y_pre = y_prob > threshold #* (y_true.shape[0]//34)).astype(np.int)
    return y_true, y_prob, f1_score(y_true, y_pre,average='micro')
    
def fbeta(true_label, prediction):
    from sklearn.metrics import f1_score
    return f1_score(true_label, prediction, average='micro')#'micro', 'macro', 'weighted', 'samples'

def optimise_f1_thresholds_fast(y, p, iterations=20, verbose=True,num_classes=34):
    best_threshold = [0.2]*num_classes
    for t in range(num_classes):
        best_fbeta = 0
        temp_threshhold = [0.2]*num_classes
        for i in range(iterations):
            temp_value = i / float(iterations)
            temp_threshhold[t] = temp_value
            temp_fbeta = fbeta(y, p > temp_threshhold)
            if temp_fbeta > best_fbeta:
                best_fbeta = temp_fbeta
                best_threshold[t] = temp_value

        if verbose:
            print(t, best_fbeta, best_threshold[t])

    return best_threshold
#打印时间
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


# 调整学习率
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

#多标签使用类别权重
class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return (loss * self.weights).mean()
    
class Multilabel(nn.Module):
    def __init__(self):
        super(Multilabel, self).__init__()
        self.cerition = nn.MultiLabelSoftMarginLoss()
    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets.long())
        return loss

