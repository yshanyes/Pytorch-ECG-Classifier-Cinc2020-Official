# -*- coding: utf-8 -*-
'''
@time: 2019/10/1 10:20

@ author: ys
'''
import os


class Config:
    # for data_process.py
    #root = r'D:\ECG'
    #root = r'/home/yangshan/cinc2020_pytorch/Pytorch-ECG-Classifier-Cinc2020-0803/Post_data/'
    root = r'../input_directory'
    train_dir = root#os.path.join(root, '/')

    # aug_train_dir = os.path.join(r'round1', 'hf_round2_train')
    #trainA_dir = os.path.join(root, 'testA')
    test_dir = root#os.path.join('data', '/')

    train_label = os.path.join(root, 'hf_round1_label.txt')
    test_label = os.path.join('data', 'hf_round1_subA.txt')
    arrythmia = os.path.join(root, 'hf_round2_arrythmia.txt')

    train_data = os.path.join(r"./pth", 'round1_data_0.pth')
    train_data_cv = os.path.join(r"./pth", 'round1_data_{}.pth')

    train_aug_data = os.path.join(root, 'train_round2_aug.pth')
    train_all_data = os.path.join(root, 'train_round2_all.pth')


    train_round1_dir = os.path.join(r'round1', 'hf_round2_train')
    train_round1_train_label = os.path.join(r'round1', 'hf_round1_label.txt')
    train_round1_subA_label = os.path.join(r'round1', 'hf_round1_subA_label.txt')

    #2019/11/03
    train_aug_resample_data = os.path.join(root, 'train_round2_aug_resample.pth')
    #2019/11/04
    train_resample_data = os.path.join(root, 'train_round1_no_resample.pth')
    # for train

    #训练的模型名称
    model_name =  "iresnest50_pretrain" #  "resnest101" #  "resnest50" # "seresnet50" # "seresnext50_32x4d" # "semkresnet18" #"seresnet50" #"seresnet34" #"resnet34" #"mixnet_sm"#  #"resnet34"
    #在第几个epoch进行到下一个state,调整lr
    stage_epoch = [32,64,80]#[24,48,72,84] #[32,64,80]#
    #训练时的batch大小
    batch_size = 16 #32#
    #label的类别数
    num_classes = 27
    #最大训练多少个epoch
    max_epoch = 100 #100 #256 80#
    #目标的采样长度
    target_point_num = 500*10 #256*50  #256*60 #

    target_fs = 500

    #保存模型的文件夹
    ckpt = 'ckpt'
    #保存提交文件的文件夹
    sub_dir = 'submit'
    #初始的学习率
    lr = 1e-3
    #保存模型当前epoch的权重
    current_w = 'current_weight.pth'
    #保存最佳的权重
    best_w = 'best_weight.pth'
    # 学习率衰减 lr/=lr_decay
    lr_decay = 10

    kfold = 5
    #保存模型当前epoch的权重
    current_w_cv = 'current_weight_fold{}.pth'
    #保存最佳的权重
    best_w_cv = 'best_weight_fold{}.pth'
    #for test
    temp_dir=os.path.join(root,'temp')

    model = None

    round1_pretrain_weight = os.path.join(ckpt,'iresnest50','transform_best_weight.pth')

    round1_pretrain_weight_cv = os.path.join(ckpt,'iresnest50_cv','transform_best_weight_fold{}.pth')


config = Config()
