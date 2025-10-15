import math
import pickle
import time

import torchmetrics
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
import tqdm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import sklearn
import os, shutil


from LSTMVITCNN import MY_NETC

from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

from utils import compute_test



def train(target,path,prtrain):
    model_save_dir = 'moblie_vit_pretrain'
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    import argparse

    device = "cuda"



    data = pickle.load(open(os.path.join(path, 'patient%d_data.pkl' % target), 'rb'))

    label = pickle.load(open(os.path.join(path, 'patient%d_label.pkl' % target), 'rb'))

    data = np.array(data)
    print("all_data",data.shape)

    label = np.array(label)

    permutation = np.random.permutation(len(data))
    data = data[permutation]
    label=label[permutation]


    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3,random_state=1)
    print("x-train",x_train.shape)

    x_train = torch.tensor(x_train, dtype=torch.float).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    net = MY_NETC(image_size=32,  # 图像大小
                 patch_size=8,  # patch大小（分块的大小）
                 num_classes=2,  # imagenet数据集1000分类
                 embed_dim=256,
                 seq_len=0,  # position embedding的维度
                 hidden_dim=2048,
                 num_heads=256,
                 num_layers=4,
                 dropout=0.1,
                 classification=False).to('cuda')


    torch.cuda.empty_cache()
    if prtrain==1:
      checkpoint = torch.load(os.path.join(model_save_dir, 'best_w.pth'))
      net.load_state_dict(checkpoint, strict=False)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00001)
    #optimizer = torch.optim.AdamW(net.paramete`rs(), lr = 0.001)
    epochs_t = 100

    lr_schduler = CosineAnnealingLR(optimizer, T_max=epochs_t - 10, eta_min=0.09)  # default =0.07
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=lr_schduler)
    optimizer.zero_grad()
    optimizer.step()
    scheduler_warmup.step()

    # 参数设置
    batch_size = 128

    val_acc_list = []
    n_train_samples = x_train.shape[0]
    iter_per_epoch = n_train_samples // batch_size + 1
    best_acc,best_sen,best_spe,best_f1,best_auc = 0,0,0,0,0
    best_res = 'null'

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)
    print('train_______________________start')
    for epoch in range(epochs_t):
        f1 = 0
        t = 0
        predict = []
        x = []
        lables = []
        net.train()
        loss_sum = 0
        evaluation = []
        iter = 0
        sum1 = 0
        train_loss = 0
        with tqdm.tqdm(total=iter_per_epoch) as pbar:
            for X, y in train_iter:
                jj = 0
                sum1 = sum1 + 1
               # a = torch.tensor(128).to(device)
                output = net(X)
                # print(output)
                _, predicted = torch.max(output.data, 1)
                # print('y---------',y )
                evaluation.append((predicted == y).tolist())

                # print('f1-score----------------------------',f1_score)
                predicted = predicted.cpu().numpy()
                predict.extend(predicted)

                # output=output.cpu().numpy()
                x = y.cpu().numpy()
                lables.extend(x)
                # print('predict---------------',y)

                # print('predicted-----------------', predicted)
                optimizer.zero_grad()
                l = criterion(output, y)
                # l = comtrast_loss(output, x)
                l.backward()
                optimizer.step()
                loss_sum += l.item()
                # loss_sum =loss_sum.cpu().numpy()

                iter += 1
                pbar.set_description("Epoch %d, loss = %.2f" % (epoch, l.data))
                pbar.update(1)
        evaluation = [item for sublist in evaluation for item in sublist]
        predict, lables = np.array(predict), np.array(lables)

        train_loss = loss_sum / sum1

        # auc = roc_auc_score(predict,lable,multi_class='ovr')

        train_acc = sum(evaluation) / len(evaluation)

        current_lr = optimizer.param_groups[0]['lr']

        if not os.path.exists('./pre_train_save'):
            os.mkdir('./pre_train_save')


        acc, f1, specificity, sensitivity, auc = compute_test(predict, lables)
        print("Epoch:", epoch, "train_loss=", train_loss, " train_acc =", acc, "specificity=", specificity,
              "sensitivity=", sensitivity, "f1=", f1, "roc=", auc)
        # scheduler.step()
        scheduler_warmup.step()
        val_loss = 0
        evaluation = []
        pred_v = []
        true_v = []
        val_predict = []
        val_x = []
        val_lable = []
        i = 0

        net.eval()
        for X, y in test_iter:
            i = i + 1

            output = net(X)

            _, predicted = torch.max(output.data, 1)
            evaluation.append((predicted == y).tolist())
            l = criterion(output, y)
            # l = comtrast_loss(output, y)

            predicted = predicted.cpu().numpy()
            val_predict.extend(predicted)
            x = y.cpu().numpy()
            val_lable.extend(x)
            val_loss += l.item()
            pred_v.append(predicted.tolist())
            true_v.append(y.tolist())
        evaluation = [item for sublist in evaluation for item in sublist]
        pred_v = [item for sublist in pred_v for item in sublist]
        true_v = [item for sublist in true_v for item in sublist]
        val_predict = np.array(val_predict)
        val_loss = val_loss / i
        val_lable = np.array(val_lable)

        val_acc, val_f1, val_specificity, val_sensitivity, val_auc = compute_test(val_predict, val_lable)
        print("val_loss =", val_loss, "val_acc =", val_acc, 'val_f1', val_f1, 'val_sprcificity', val_specificity,
              'val_senstivity', val_sensitivity, "val_auc = ", val_auc)

        running_acc = sum(evaluation) / len(evaluation)
        val_acc_list.append(running_acc)

        #  finish_test_res.write("val_loss ="+str(np.round(val_loss,3))+ "val_acc ="+ str(np.round(val_acc,3))+"val_specificity="+str(np.round(val_specificity,3))+"f1="+str(np.round(f1,3)))

        #  finish_test_res.close()

        # state = {"state_dict": net.state_dict(), "epoch": epoch}
        # save_ckpt(state, best_acc < running_acc, model_save_dir, 'best_cls.pth')
        res = "bets results:" + \
              'val_epoch=_' + str(epoch) + \
              "val_loss" + str(val_loss) + \
              "_accuracy=_" + str(val_acc) + \
              "_sensitivity=_" + str(val_sensitivity) + \
              "_specificity=_" + str(val_specificity) + \
              '_F1=_' + str(val_f1) + \
              '_auc=_' + str(val_auc) + '\n'

        if val_f1 > best_f1:
            if not os.path.exists('./LTformer/%d' % target):
                os.makedirs('./LTformer/%d' % target)
            torch.save(net.state_dict(), './LTformer/%d/best_model.pth' % target)
            best_res = res
            best_f1 = val_f1
            best_acc,best_sen,best_spe,best_f1,best_auc= val_acc,val_sensitivity,val_specificity,val_f1,val_auc
    #finish_test_res = open("pre_train_save/with_pretrain_finish_test-res%d.txt" % (target), "w")  #覆盖写入
    finish_test_res = open("result/r.txt", "a")   #所有病人写入一个文件
    finish_test_res.write(str(target))
    finish_test_res.write(best_res)
    print("Highest acc:", max(val_acc_list))
    return target,float(best_acc),float(best_sen),float(best_spe),float(best_f1),float(best_auc)
    #主函数

path = 'D:/balancedata'

prtrain = 1

to_acc, to_sen, to_spe, to_f1, to_auc = [],[],[],[],[]
strat_time =time.time()

for target in range (1, 25):
    target, acc, sen, spe, f1, auc = train(target,path,prtrain)
    print(acc)
    to_acc.append(acc)
    to_sen.append(sen)
    to_spe.append(spe)
    to_f1.append(f1)
    to_auc.append(auc)


mean_acc, mean_sen, mean_spe, mean_f1, mean_auc = sum(to_acc)/len(to_acc),sum(to_sen)/len(to_sen),\
                                                  sum(to_spe)/len(to_spe),sum(to_f1)/len(to_f1),\
                                                  sum(to_auc)/len(to_auc)
print(to_acc)
men_res = "mean result:" + \
              "mean_accuracy=_" + str(mean_acc) + \
              "mean_sensitivity=_" + str(mean_sen) + \
              "mean_specificity=_" + str(mean_spe) + \
              'mean_F1=_' + str(mean_f1) + \
              'mean_auc=_' + str(mean_auc) + '\n'
finish_test_res = open("result/r.txt", "a")  # 所有病人写入一个文件
finish_test_res.write(men_res)
end_time = time.time()
str=" 总运行时间为"+str(end_time-strat_time)+"s"
print(str)