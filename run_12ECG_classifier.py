#!/usr/bin/env python

import numpy as np, os, sys
import joblib
from get_12ECG_features import get_12ECG_features

from config import config
import models
from dataset import transform
import torch
import pandas as pd
from scipy import signal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_12ECG_classifier(data,header_data,loaded_model):


    # Use your classifier here to obtain a label and score for each class.

    dx_mapping_scored = pd.read_csv('./evaluation/dx_mapping_scored.csv')['SNOMED CT Code'].values.tolist()

    with torch.no_grad():
        sig = data
        FS = 500
        SIGLEN = FS * 10

        fs = int(header_data[0].split(' ')[2])
        siglen = int(header_data[0].split(' ')[3])
        adc_gain = int(header_data[1].split(' ')[2].split('/')[0])

        # print(fs,siglen,adc_gain)

        if fs == FS * 2 :
            # sig = signal.resample(sig.T, int(annot.siglen/annot.fs * FS)).T
            sig = sig[:,::2]
        elif fs == FS:
            pass#raise ValueError("fs wrong")
        elif fs != FS:
            sig = signal.resample(sig.T, int(siglen/fs * FS)).T

        siglen = sig.shape[1]
        # print(siglen)

        if siglen !=  SIGLEN:
            sig_ext = np.zeros([12,SIGLEN])

        if siglen <  SIGLEN:
            sig_ext[:,:siglen] = sig
        if siglen >  SIGLEN:
            sig_ext = sig[:,:SIGLEN]

        if siglen !=  SIGLEN:
            sig = sig_ext

        sig = sig/adc_gain

        x = transform(sig.T,train=False).unsqueeze(0).to(device)

        # k-fold
        ''' '''
        output = 0
        kfold = 5
        for fold in range(kfold):
            output += torch.sigmoid(loaded_model[fold](x)).squeeze().cpu().numpy()
        output = output/kfold
        mapping = dict(zip([str(i) for i in dx_mapping_scored],output))
        output = [mapping[key] for key in sorted(mapping.keys())]
        ixs = [1 if out>0.25 else 0 for out in output]
        '''
        # 0.2 ——  0.990,0.881,0.629,0.792,0.826,0.605,0.792
        # 0.25 —— 0.990,0.881,0.669,0.806,0.825,0.620,0.796

        # one-fold
        output = torch.sigmoid(loaded_model(x)).squeeze().cpu().numpy()
        mapping = dict(zip([str(i) for i in dx_mapping_scored],output))
        output = [mapping[key] for key in sorted(mapping.keys())]
        ixs = [1 if out>0.25 else 0 for out in output]
        '''
        # 0.5 —— 0.975,0.745,0.621,0.679,0.657,0.460,0.654
        # 0.4 —— 0.975,0.745,0.610,0.697,0.689,0.477,0.696
        # 0.3 —— 0.975,0.745,0.579,0.704,0.713,0.482,0.723
        # 0.2 —— 0.975,0.745,0.517,0.694,0.728,0.470,0.733
        # 0.1 —— 0.975,0.745,0.396,0.637,0.711,0.417,0.701


    current_score = output
    current_label = ixs

    classes = sorted([str(i) for i in dx_mapping_scored])#[str(c) for c in dx_mapping_scored]

    return current_label, current_score, classes

'''
def load_12ECG_model(input_directory):

    # filename = os.path.join(input_directory,config.best_w)
    # filename = os.path.join("submit_model",config.best_w)
    filename = os.path.join(input_directory,"best_weight_fold0.pth")

    model = getattr(models, 'iresnest50_predict')()

    model.load_state_dict(torch.load(filename, map_location='cpu')['state_dict'])
    model = model.to(device)
    model.eval()

    return model
'''
def load_12ECG_model(input_directory):

    kfold = 5

    filename = os.path.join(input_directory,config.best_w)

    # model
    model = []
    for fold in range(kfold):
        model.append(getattr(models, 'iresnest50_predict')())
    for fold in range(kfold):
        # model[fold].load_state_dict(torch.load(os.path.join("submit_model","best_weight_fold{}.pth".format(fold)), map_location='cpu')['state_dict'])
        model[fold].load_state_dict(torch.load(os.path.join(input_directory,"best_weight_fold{}.pth".format(fold)), map_location='cpu')['state_dict'])
        model[fold] = model[fold].to(device)
        model[fold].eval()

    return model
''''''
