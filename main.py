# -*- coding: utf-8 -*-
'''
@time: 2019/10/1 10:20

@ author: ys
'''
import torch, time, os, shutil
import models, utils, pytorchtools
import numpy as np
import pandas as pd
from tensorboard_logger import Logger
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import ECGDataset
from config import config
from tqdm import tqdm
import radam
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED=41
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True


# 保存当前模型的权重，并且更新最佳的模型权重
def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, config.current_w)
    best_w = os.path.join(model_save_dir, config.best_w)
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)

# 保存当前模型的权重，并且更新最佳的模型权重
def save_ckpt_cv(state, is_best, model_save_dir,fold):
    current_w = os.path.join(model_save_dir, config.current_w_cv.format(fold))
    best_w = os.path.join(model_save_dir, config.best_w_cv.format(fold))
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)

def train_epoch(model, optimizer, criterion, train_dataloader, show_interval=10):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    acc_meter,f1_meter,f2_meter,g2_meter = 0,0,0,0
    cm_meter = 0
    for inputs, target in train_dataloader:
        inputs = inputs.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1

        #f1 = utils.calc_f1(target, torch.sigmoid(output))
        #f1_meter += f1
        # acc, f1, f2, g2 = utils.calc_metric(target, torch.sigmoid(output))
        acc, f1, f2, g2, cm = utils.calc_metric(target, torch.sigmoid(output))
        acc_meter+= acc
        f1_meter += f1
        f2_meter += f2
        g2_meter += g2
        cm_meter += cm

        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e acc:%.3f f1:%.3f f2:%.3f g2:%.3f cm:%.3f " % (it_count, loss.item(), acc,f1,f2,g2,cm))
    return loss_meter / it_count, acc_meter / it_count,f1_meter / it_count,f2_meter / it_count,g2_meter / it_count,cm_meter / it_count

def train_beat_epoch(model, optimizer, criterion, train_dataloader, show_interval=10):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    for inputs, beat, target in train_dataloader:
        inputs = inputs.to(device)
        beat = beat.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(inputs,beat)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        f1 = utils.calc_f1(target, torch.sigmoid(output))
        f1_meter += f1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e f1:%.3f" % (it_count, loss.item(), f1))
    return loss_meter / it_count, f1_meter / it_count

def val_epoch(model, criterion, val_dataloader, threshold=0.5):
    model.eval()
    f1_meter, loss_meter, it_count = 0, 0, 0
    acc_meter,f1_meter,f2_meter,g2_meter = 0,0,0,0
    cm_meter = 0
    with torch.no_grad():
        for inputs, target in val_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            output = torch.sigmoid(output)

            # f1 = utils.calc_f1(target, output, threshold)
            # f1_meter += f1
            # acc ,f1 ,f2 ,g2 = utils.calc_metric(target, output,threshold)
            acc ,f1 ,f2 ,g2, cm = utils.calc_metric(target, output,threshold)
            acc_meter+= acc
            f1_meter += f1
            f2_meter += f2
            g2_meter += g2
            cm_meter += cm

    return loss_meter / it_count, acc_meter / it_count,f1_meter / it_count,f2_meter / it_count,g2_meter / it_count,cm_meter / it_count

def val_beat_epoch(model, criterion, val_dataloader, threshold=0.5):
    model.eval()
    f1_meter, loss_meter, it_count = 0, 0, 0
    with torch.no_grad():
        for inputs, beat, target in val_dataloader:
            inputs = inputs.to(device)
            beat = beat.to(device)
            target = target.to(device)
            output = model(inputs,beat)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            output = torch.sigmoid(output)
            f1 = utils.calc_f1(target, output, threshold)
            f1_meter += f1
    return loss_meter / it_count, f1_meter / it_count

def train(args):
    # model
    model = getattr(models, config.model_name)()
    if args.ckpt and not args.resume:
        state = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(state['state_dict'])
        print('train with pretrained weight val_f1', state['f1'])
    model = model.to(device)
    # data
    train_dataset = ECGDataset(data_path=config.train_data, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=6)
    val_dataset = ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)

    print("train_datasize", len(train_dataset), "val_datasize", len(val_dataset))
    # optimizer and loss
    #optimizer = optim.Adam(model.parameters(), lr=config.lr)
    optimizer = radam.RAdam(model.parameters(), lr=config.lr, weight_decay=1e-4) #config.lr
    #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=False)
    w = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
    criterion = utils.WeightedMultilabel(w) ##   # utils.FocalLoss() #

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose=True, factor=0.1, patience=5, min_lr=1e-06, eps=1e-08)#CosineAnnealingLR  CosineAnnealingWithRestartsLR
    #scheduler = pytorchtools.CosineAnnealingWithRestartsLR(optimizer,T_max=30, T_mult = 1.2, eta_min=1e-6)

    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    # scheduler = pytorchtools.CosineAnnealingLR_with_Restart(optimizer, T_max=12, T_mult=1, model=model, out_dir='./snapshot',take_snapshot=True, eta_min=1e-9)

    # 模型保存文件夹
    model_save_dir = '%s/%s_%s' % (config.ckpt, config.model_name, time.strftime("%Y%m%d%H%M"))
    if args.ex: model_save_dir += args.ex
    best_f1 = -1
    best_cm = -1
    lr = config.lr
    start_epoch = 1
    stage = 1
    # 从上一个断点，继续训练
    if args.resume:
        if os.path.exists(args.ckpt):  # 这里是存放权重的目录
            model_save_dir = args.ckpt
            current_w = torch.load(os.path.join(args.ckpt, config.current_w))
            best_w = torch.load(os.path.join(model_save_dir, config.best_w))
            best_f1 = best_w['loss']
            start_epoch = current_w['epoch'] + 1
            lr = current_w['lr']
            stage = current_w['stage']
            model.load_state_dict(current_w['state_dict'])
            # 如果中断点恰好为转换stage的点
            if start_epoch - 1 in config.stage_epoch:
                stage += 1
                lr /= config.lr_decay
                utils.adjust_learning_rate(optimizer, lr)
                model.load_state_dict(best_w['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(start_epoch - 1))
    logger = Logger(logdir=model_save_dir, flush_secs=2)
    # =========>开始训练<=========
    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        train_loss, train_acc, train_f1, train_f2, train_g2,train_cm = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)
        val_loss, val_acc, val_f1, val_f2, val_g2, val_cm = val_epoch(model, criterion, val_dataloader)

        # train_loss, train_f1 = train_beat_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)
        # val_loss, val_f1 = val_beat_epoch(model, criterion, val_dataloader)

        print('#epoch:%02d, stage:%d, train_loss:%.3e, train_acc:%.3f, train_f1:%.3f, train_f2:%.3f, train_g2:%.3f,train_cm:%.3f,\n \
                val_loss:%0.3e, val_acc:%.3f, val_f1:%.3f, val_f2:%.3f, val_g2:%.3f, val_cm:%.3f,time:%s\n'
              % (epoch, stage, train_loss, train_acc,train_f1,train_f2,train_g2,train_cm, \
                val_loss, val_acc, val_f1, val_f2, val_g2, val_cm,utils.print_time_cost(since)))

        logger.log_value('train_loss', train_loss, step=epoch)
        logger.log_value('train_f1', train_f1, step=epoch)
        logger.log_value('val_loss', val_loss, step=epoch)
        logger.log_value('val_f1', val_f1, step=epoch)
        state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'f1': val_f1, 'lr': lr,
                 'stage': stage}

        save_ckpt(state, best_cm < val_cm, model_save_dir)
        best_cm = max(best_cm, val_cm)

        scheduler.step(val_cm)
        # scheduler.step()

        if val_cm < best_cm:
            epoch_cum += 1
        else:
            epoch_cum = 0

#         # if epoch in config.stage_epoch:
#         if epoch_cum == 5:
#             stage += 1
#             lr /= config.lr_decay
#             if lr < 1e-6:
#                 lr = 1e-6
#                 print("*" * 20, "step into stage%02d lr %.3ef" % (stage, lr))
#             best_w = os.path.join(model_save_dir, config.best_w)
#             model.load_state_dict(torch.load(best_w)['state_dict'])
#             print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
#             utils.adjust_learning_rate(optimizer, lr)

#         elif epoch_cum >= 12:
#             print("*" * 20, "step into stage%02d lr %.3ef" % (stage, lr))
#             break

        if epoch_cum >= 12:
            print("*" * 20, "step into stage%02d lr %.3ef" % (stage, lr))
            break

def train_cv(args):
    # model
    # 模型保存文件夹
    model_save_dir = '%s/%s_%s' % (config.ckpt, config.model_name+"_cv",time.strftime("%Y%m%d%H%M"))#'%s/%s_%s' % (config.ckpt, args.model_name+"_cv", time.strftime("%Y%m%d%H%M"))
    for fold in range(config.kfold):
        print("***************************fold : {}***********************".format(fold))
        model = getattr(models, config.model_name)()
        if args.ckpt and not args.resume:
            state = torch.load(args.ckpt, map_location='cpu')
            model.load_state_dict(state['state_dict'])
            print('train with pretrained weight val_f1', state['f1'])

        # num_ftrs = model.classifier.in_features
        # model.classifier = nn.Linear(num_ftrs, config.num_classes)

        #2019/11/11
        #save dense/fc weight for pretrain 55 classes
        # model = MyModel()
        # num_ftrs = model.classifier.out_features
        # model.fc = nn.Linear(55, config.num_classes)

        model = model.to(device)
        # data
        train_dataset = ECGDataset(data_path=config.train_data_cv.format(fold),train=True)

        train_dataloader = DataLoader(train_dataset,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    num_workers=6)

        val_dataset = ECGDataset(data_path=config.train_data_cv.format(fold),train=False)

        val_dataloader = DataLoader(val_dataset,
                                    batch_size=config.batch_size,
                                    num_workers=4)

        print("fold_{}_train_datasize".format(fold), len(train_dataset), "fold_{}_val_datasize".format(fold), len(val_dataset))
        # optimizer and loss
        optimizer = radam.RAdam(model.parameters(), lr=config.lr) #optim.Adam(model.parameters(), lr=config.lr)
        w = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
        criterion = utils.WeightedMultilabel(w) ## utils.FocalLoss() #

        if args.ex: model_save_dir += args.ex
        best_f1 = -1
        lr = config.lr
        start_epoch = 1
        stage = 1
        # 从上一个断点，继续训练
#         if args.resume:
#             if os.path.exists(args.ckpt):  # 这里是存放权重的目录
#                 model_save_dir = args.ckpt
#                 current_w = torch.load(os.path.join(args.ckpt, config.current_w))
#                 best_w = torch.load(os.path.join(model_save_dir, config.best_w))
#                 best_f1 = best_w['loss']
#                 start_epoch = current_w['epoch'] + 1
#                 lr = current_w['lr']
#                 stage = current_w['stage']
#                 model.load_state_dict(current_w['state_dict'])
#                 # 如果中断点恰好为转换stage的点
#                 if start_epoch - 1 in config.stage_epoch:
#                     stage += 1
#                     lr /= config.lr_decay
#                     utils.adjust_learning_rate(optimizer, lr)
#                     model.load_state_dict(best_w['state_dict'])
#                 print("=> loaded checkpoint (epoch {})".format(start_epoch - 1))
        logger = Logger(logdir=model_save_dir, flush_secs=2)
        # =========>开始训练<=========
        for epoch in range(start_epoch, config.max_epoch + 1):
            since = time.time()
            train_loss, train_acc, train_f1, train_f2, train_g2 = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)
            val_loss, val_acc, val_f1, val_f2, val_g2 = val_epoch(model, criterion, val_dataloader)

            # train_loss, train_f1 = train_beat_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)
            # val_loss, val_f1 = val_beat_epoch(model, criterion, val_dataloader)

            print('#epoch:%02d, stage:%d, train_loss:%.3e, train_acc:%.3f, train_f1:%.3f, train_f2:%.3f, train_g2:%.3f,\n \
                    val_loss:%0.3e, val_acc:%.3f, val_f1:%.3f, val_f2:%.3f, val_g2:%.3f, time:%s\n'
                  % (epoch, stage, train_loss, train_acc,train_f1,train_f2,train_g2, \
                    val_loss, val_acc, val_f1, val_f2, val_g2, utils.print_time_cost(since)))

            logger.log_value('fold{}_train_loss'.format(fold),  train_loss, step=epoch)
            logger.log_value('fold{}_train_f1'.format(fold), train_f1, step=epoch)
            logger.log_value('fold{}_val_loss'.format(fold),  val_loss, step=epoch)
            logger.log_value('fold{}_val_f1'.format(fold),  val_f1, step=epoch)
            state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'f1': val_f1, 'lr': lr,
                     'stage': stage}
            save_ckpt_cv(state, best_f1 < val_f1, model_save_dir,fold)
            best_f1 = max(best_f1, val_f1)

            if val_f1 < best_f1:
                epoch_cum += 1
            else:
                epoch_cum = 0

            # if epoch in config.stage_epoch:
            if epoch_cum == 5:
                stage += 1
                lr /= config.lr_decay
                if lr < 1e-6:
                    lr = 1e-6
                    print("*" * 20, "step into stage%02d lr %.3ef" % (stage, lr))
                best_w = os.path.join(model_save_dir, config.best_w_cv.format(fold))
                model.load_state_dict(torch.load(best_w)['state_dict'])
                print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
                utils.adjust_learning_rate(optimizer, lr)

            elif epoch_cum >= 12:
                print("*" * 20, "step into stage%02d lr %.3ef" % (stage, lr))
                break

            # if epoch in config.stage_epoch:
            #     stage += 1
            #     lr /= config.lr_decay
            #     best_w = os.path.join(model_save_dir, config.best_w_cv.format(fold))
            #     model.load_state_dict(torch.load(best_w)['state_dict'])
            #     print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            #     utils.adjust_learning_rate(optimizer, lr)

#用于测试加载模型
def val(args):
    list_threhold = [0.5]
    model = getattr(models, config.model_name)()
    if args.ckpt: model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'])
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    val_dataset = ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)
    for threshold in list_threhold:
        val_loss, val_f1 = val_epoch(model, criterion, val_dataloader, threshold)
        print('threshold %.2f val_loss:%0.3e val_f1:%.3f\n' % (threshold, val_loss, val_f1))

#提交结果使用
def test(args):
    from dataset import transform
    name2idx = {"AF":0,"I-AVB":1,"LBBB":2,"Normal":3,"PAC":4,"PVC":5,"RBBB":6,"STD":7,"STE":8}
    idx2name = {0:"AF",1:"I-AVB",2:"LBBB",3:"Normal",4:"PAC",5:"PVC",6:"RBBB",7:"STD",8:"STE"}

    # model
    model = getattr(models, config.model_name)()
    print(model)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'])
    model = model.to(device)
    model.eval()
    #sub_file = '%s/subB_%s.txt' % (config.sub_dir, time.strftime("%Y%m%d%H%M"))
    sub_file = 'result.txt'
    fout = open(sub_file, 'w', encoding='utf-8')
    with torch.no_grad():
        for line in tqdm(open(config.test_label, encoding='utf-8')):
            fout.write(line.strip('\n'))
            id = line.split('\t')[0]
            file_path = os.path.join(config.test_dir, id)
            df = pd.read_csv(file_path, sep=' ')
            df['III'] = df['II']-df['I']
            df['aVR'] = -(df['I']+df['II'])/2
            df['aVL'] = df['I']-df['II']/2
            df['aVF'] = df['II']-df['I']/2
            x = transform(df.values).unsqueeze(0).to(device)
            output = torch.sigmoid(model(x)).squeeze().cpu().numpy()
            ixs = [i for i, out in enumerate(output) if out > 0.5]
            for i in ixs:
                fout.write("\t" + idx2name[i])
            fout.write('\n')
    fout.close()

#提交结果使用
def test_cv(args):
    from dataset import transform
    from data_process import name2index,get_arrythmias,get_dict
    # arrythmias = get_arrythmias(config.arrythmia)
    # name2idx,idx2name = get_dict(arrythmias)
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    #utils.mkdirs(config.sub_dir)
    num_clases = 34
    kfold = 5
    # model
    model = []
    for fold in range(kfold):
        model.append(getattr(models, config.model_name)())
    for fold in range(kfold):
        model[fold].load_state_dict(torch.load(os.path.join(args.ckpt,"best_weight_fold{}.pth".format(fold)), map_location='cpu')['state_dict'])
        model[fold] = model[fold].to(device)
        model[fold].eval()

    #sub_file = '%s/subB_%s.txt' % (config.sub_dir, time.strftime("%Y%m%d%H%M"))
    sub_file = 'result.txt'
    fout = open(sub_file, 'w', encoding='utf-8')
    with torch.no_grad():
        for line in tqdm(open(config.test_label, encoding='utf-8')):
            fout.write(line.strip('\n'))
            id = line.split('\t')[0]
            file_path = os.path.join(config.test_dir, id)
            df = pd.read_csv(file_path, sep=' ')
            df['III'] = df['II']-df['I']
            df['aVR'] = -(df['I']+df['II'])/2
            df['aVL'] = df['I']-df['II']/2
            df['aVF'] = df['II']-df['I']/2
            x = transform(df.values).unsqueeze(0).to(device)
            output = 0#np.zeros(num_clases)
            for fold in range(kfold):
                output += torch.sigmoid(model[fold](x)).squeeze().cpu().numpy()
            output = output/5
            ixs = [i for i, out in enumerate(output) if out > 0.5]
            for i in ixs:
                fout.write("\t" + idx2name[i])
            fout.write('\n')
    fout.close()
#提交结果使用
def test_ensemble(args):
    from dataset import transform
    from data_process import name2index,get_arrythmias,get_dict
    # arrythmias = get_arrythmias(config.arrythmia)
    # name2idx,idx2name = get_dict(arrythmias)
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    #utils.mkdirs(config.sub_dir)
    num_clases = 34
    kfold = len(config.model_names)
    # model
    model = []
    for fold in range(kfold):
        model.append(getattr(models, config.model_names)())
    for fold in range(kfold):
        model[fold].load_state_dict(torch.load(os.path.join(args.ckpt,config.model_ckpts[fold],"best_weight.pth"), map_location='cpu')['state_dict'])
        model[fold] = model[fold].to(device)
        model[fold].eval()

    #sub_file = '%s/subB_%s.txt' % (config.sub_dir, time.strftime("%Y%m%d%H%M"))
    sub_file = './result.txt'
    fout = open(sub_file, 'w', encoding='utf-8')
    with torch.no_grad():
        for line in tqdm(open(config.test_label, encoding='utf-8')):
            fout.write(line.strip('\n'))
            id = line.split('\t')[0]
            file_path = os.path.join(config.test_dir, id)
            df = pd.read_csv(file_path, sep=' ')
            df['III'] = df['II']-df['I']
            df['aVR'] = -(df['I']+df['II'])/2
            df['aVL'] = df['I']-df['II']/2
            df['aVF'] = df['II']-df['I']/2
            x = transform(df.values).unsqueeze(0).to(device)
            output = 0#np.zeros(num_clases)
            for fold in range(kfold):
                output += torch.sigmoid(model[fold](x)).squeeze().cpu().numpy()
            output = output/kfold
            ixs = [i for i, out in enumerate(output) if out > 0.5]
            for i in ixs:
                fout.write("\t" + idx2name[i])
            fout.write('\n')
    fout.close()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>", help="train or infer")
    parser.add_argument("--ckpt", type=str, help="the path of model weight file")
    parser.add_argument("--ex", type=str, help="experience name")
    parser.add_argument("--resume", action='store_true', default=False)
    args = parser.parse_args()
    if (args.command == "train"):
        train(args)
    if (args.command == "train_cv"):
        train_cv(args)
    if (args.command == "test"):
        test(args)
    if (args.command == "test_cv"):
        test_cv(args)
    if (args.command == "val"):
        val(args)
