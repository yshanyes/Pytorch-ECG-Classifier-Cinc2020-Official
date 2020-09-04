import os,sys
import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def get_labels(path):

    files = []
    # for path in path_list:
    for file in tqdm(os.listdir(path)):
        if file.endswith('.mat'):
            files.append(path +"/"+file.split(".")[0])

    label1 = []
    label2 = []
    label3 = []
    label4 = []
    sig_len = []
    sex = []
    age = []
    fs = []
    label_num = []
    class_mapping = {}
    classes_count = {}
    str_label = []
    label_files = []
    for _,file in tqdm(enumerate(files)):
        annotation = wfdb.rdheader(file)
        #print(annotation.__dict__)
        if annotation.fs < 500:
            continue

        label_files.append(file)
        sig_len.append(annotation.sig_len)
        fs.append(annotation.fs)

        sex.append(annotation.comments[1].split(": ")[1])
        try:
            age.append(int(annotation.comments[0].split(": ")[1]))
        except:
            age.append(np.nan)

        '''  '''
        l1 = annotation.comments[2].split(": ")[1]
        if l1 != "Unknown":
            str_label.append(l1)
            labels = l1.split(",")
            label_num.append(len(labels))

            for class_name in labels:
                if class_name not in class_mapping:
                    class_mapping[class_name] = len(class_mapping)

                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1

            # label1.append(labels[0])
            # try:
            #     label2.append(labels[1])
            # except:
            #     label2.append('')

            # try:
            #     label3.append(labels[2])
            # except:
            #     label3.append('')

            # try:
            #     label4.append(labels[3])
            # except:
            #     label4.append('')

    labels = pd.DataFrame({"record_name":label_files,
                           "length":sig_len,
                           "age":age,
                           "sex":sex,
                           "fs":fs,
                           "label_num":label_num,
                           "labels":str_label,
                           # "label1":label1,
                           # "label2":label2,
                           # "label3":label3,
                           # "label4":label4
                           })

    return labels

def get_dx_mapping():
    dx_mapping_scored = pd.read_csv('./evaluation/dx_mapping_scored.csv')
    dx_mapping_unscored = pd.read_csv('./evaluation/dx_mapping_unscored.csv')

    duplicate_scored_mapping = dict(zip(dx_mapping_scored['SNOMED CT Code'].values.tolist(),dx_mapping_scored.index.values.tolist()))
    duplicate_unscored_mapping = dict(zip(dx_mapping_unscored['SNOMED CT Code'].values.tolist(),dx_mapping_unscored.index.values.tolist()))

    return dx_mapping_scored,duplicate_scored_mapping


# weights_file = './evaluation/weights.csv'
# mapping_score_file = './evaluation/dx_mapping_scored.csv'
# Check if the input is a number.

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

    # print(values)

    assert(rows == cols)
    num_rows = len(rows)

    # Assign the entries of the weight matrix with rows and columns corresponding to the classes.
    num_classes = len(classes)
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i, a in enumerate(rows):
        # a = int(a)

        if a in classes:
            k = classes.index(a)
            for j, b in enumerate(rows):
                # b = int(b)
                if b in classes:
                    l = classes.index(b)
                    weights[k, l] = values[i, j]

    return weights

# normal_class = '426783006'
# equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

# name2idx = duplicate_unscored_mapping#{"AF":0,"I-AVB":1,"LBBB":2,"Normal":3,"PAC":4,"PVC":5,"RBBB":6,"STD":7,"STE":8}
# idx2name = dict(zip(dx_mapping_unscored.index.values.tolist(),dx_mapping_unscored['SNOMED CT Code'].values.tolist()))
# #{0:"AF",1:"I-AVB",2:"LBBB",3:"Normal",4:"PAC",5:"PVC",6:"RBBB",7:"STD",8:"STE"}


def split_data_cv(file2idx, num_classes=9, kfold=5):
    X_train_cv = []
    X_val_cv   = []
    X = []
    y = np.zeros([len(file2idx),num_classes])
    #print(y.shape)
    for i,(file, list_idx) in enumerate(file2idx.items()):
        #print(file, list_idx)
        X.append(file)
        y[i,file2idx[file]] = 1

    X = np.array(X)

    mskf = MultilabelStratifiedKFold(n_splits=kfold, random_state=42)
    for train_index, test_index in mskf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_val = X[train_index], X[test_index]
        X_train_cv.append(list(X_train))
        X_val_cv.append(list(X_val))
        #y_train, y_test = y[train_index], y[test_index]
    return X_train_cv,X_val_cv


def count_labels(data, file2idx, num_classes=9):
    '''
    统计每个类别的样本数
    :param data:
    :param file2idx:
    :return:
    '''
    cc = [0] * num_classes
    for fp in data:
        for i in file2idx[fp]:
            cc[i] += 1
    return np.array(cc)



def train_cv_data(name2idx, idx2name,labels,kfold=5,dx_mapping=None,num_classes=9):

    file2idx = dict()

    for i in range(labels.shape[0]):
        label = []
        idx = str(labels.iloc[i].record_name) + ".mat"
        label = [dx_mapping[int(lab)] for lab in labels.iloc[i].labels.split(',') if int(lab) in dx_mapping.keys()]

        if label == []:
            continue

#         if ~np.isnan(labels_round1.iloc[i].label1):
#             label.append(int(labels_round1.iloc[i].label1))
#         if ~np.isnan(labels_round1.iloc[i].label2):
#             label.append(int(labels_round1.iloc[i].label2))
#         if ~np.isnan(labels_round1.iloc[i].label3):
#             label.append(int(labels_round1.iloc[i].label3))

        file2idx[idx] = label
    print(len(file2idx))

    train_cv, val_cv = split_data_cv(file2idx,num_classes=num_classes,kfold=kfold)
    for i in range(kfold):
        wc=count_labels(train_cv[i],file2idx,num_classes=num_classes)
        print(len(train_cv[i]))
        print(len(val_cv[i]))
        print(wc)

        wc1=count_labels(val_cv[i],file2idx,num_classes=num_classes)
        print(wc1)
        print("**********************************************************************")
        dd = {'train': train_cv[i], 'val': val_cv[i], "idx2name": idx2name, 'file2idx': file2idx,'wc':wc}
        torch.save(dd, "./pth/round1_data_{}.pth".format(i))

def split(path):
    # labels_round1 = pd.read_csv("./labels.csv")
    labels = get_labels(path)
    #print(labels.shape)
    dx_mapping_scored,duplicate_scored_mapping = get_dx_mapping()
    name2idx = duplicate_scored_mapping
    idx2name = dict(zip(dx_mapping_scored.index.values.tolist(),dx_mapping_scored['SNOMED CT Code'].values.tolist()))
    train_cv_data(name2idx, idx2name, labels, kfold=5, dx_mapping=duplicate_scored_mapping, num_classes=len(name2idx))

if __name__ == '__main__':

    path = sys.argv[1]
    split(path)
