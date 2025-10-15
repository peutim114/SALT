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

from torch import optim



import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

from pretask.care_module import CARE
from pretask.transform import Transform
from utilss import compute_test

def save_ckpt(state, is_best, model_save_dir, message='best_w.pth'):
    current_w = os.path.join(model_save_dir, 'latest_w.pth')
    best_w = os.path.join(model_save_dir, message)
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)

def transform(x_, mode):
   # x_ = x.cpu().numpy()

    Trans = Transform()
    if mode == 'time_warp':
        pieces = random.randint(5,20)
        stretch = random.uniform(1.5,4)
        squeeze = random.uniform(0.25,0.67)
        x_ = Trans.time_warp(x_, 100, pieces, stretch, squeeze)
    elif mode == 'noise':
        factor = random.uniform(10,20)
        x_ = Trans.add_noise_with_SNR(x_,factor)
    elif mode == 'scale':
        x_ = Trans.scaled(x_,[0.3,3])
    elif mode == 'negate':
        x_ = Trans.negate(x_)
    elif mode == 'hor_flip':
        x_ = Trans.hor_filp(x_)
    elif mode == 'permute':
        pieces = random.randint(5,20)
        x_ = Trans.permute(x_,pieces)
    elif mode == 'cutout_resize':
        pieces = random.randint(5, 20)
        x_ = Trans.cutout_resize(x_, pieces)
    elif mode == 'cutout_zero':
        pieces = random.randint(5, 20)
        x_ = Trans.cutout_zero(x_, pieces)
    elif mode == 'crop_resize':
        size = random.uniform(0.25,0.75)
        x_ = Trans.crop_resize(x_, size)
    elif mode == 'move_avg':
        n = random.randint(3, 10)
        x_ = Trans.move_avg(x_,n, mode="same")
    #     to test
    elif mode == 'lowpass':
        order = random.randint(3, 10)
        cutoff = random.uniform(5,20)
        x_ = Trans.lowpass_filter(x_, order, [cutoff])
    elif mode == 'highpass':
        order = random.randint(3, 10)
        cutoff = random.uniform(5, 10)
        x_ = Trans.highpass_filter(x_, order, [cutoff])
    elif mode == 'bandpass':
        order = random.randint(3, 10)
        cutoff_l = random.uniform(1, 5)
        cutoff_h = random.uniform(20, 40)
        cutoff = [cutoff_l, cutoff_h]
        x_ = Trans.bandpass_filter(x_, order, cutoff)

    # else:
    #     print("Error")

    x_ = x_.copy()
    x_ = x_[:,None,:]
    return x_



model_save_dir = 'visiontransformer_model_pretrain'
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)
import argparse
target = 21
device = "cuda"

path = 'D:/新建文件夹/balancedata'
#path = 'H:/balancedata'
data = pickle.load(open(os.path.join(path, 'patient%d_data.pkl'%target),'rb'))


strat_time=time.time()

label = pickle.load(open(os.path.join(path, 'patient%d_label.pkl'%target ),'rb'))

data=np.array(data)

label=np.array(label)
# x_test = data[620:788]
# y_test = label[620:788]
# x_train = data[323:620]
# y_train = label[323:620]

# data = np.expand_dims(data, 2)
#
trans = []
for i in range(data.shape[0]):
    t1 = transform(data[i], 'permute')
    trans[1].append(t1)
for i in range(data.shape[0]):
    t2 = transform(data[i], 'noise')
    trans[2].append(t2)
print('trans',np.array(trans).shape)
data = trans.reshape(-1, 4, 32, 32)
#打乱数据
permutation = np.random.permutation(len(data))
data = data[permutation]
label=label[permutation]
roc = roc_auc_score(label,label)
#数据分割 按照训练测试0.3的比例划分
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3,random_state=3)
#数据传入cuda
x_train = torch.tensor(x_train, dtype=torch.float).to(device)
x_test = torch.tensor(x_test, dtype=torch.float).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)




net = CARE(param_momentum = 0.99,total_iters=400).to(device)






criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.Adam(net.parameters(),lr=0.001,weight_decay=0)
epochs_t =100
lr_schduler = CosineAnnealingLR(optimizer, T_max=epochs_t - 10, eta_min=0.09)#default =0.07
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=lr_schduler)
optimizer.zero_grad()
optimizer.step()
scheduler_warmup.step()


#参数设置
batch_size = 256

val_acc_list = []
n_train_samples = x_train.shape[0]
iter_per_epoch = n_train_samples // batch_size + 1
best_f1 = -1
best_res = 'null'

train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)
print('train_______________________start')
for epoch in range(epochs_t):
    f1=0
    t=0
    predict = []
    x = []
    lables = []
    net.train()
    loss_sum = 0
    evaluation = []
    iter = 0
    sum1=0
    train_loss = 0
    with tqdm.tqdm(total=iter_per_epoch) as pbar:
        for X, y in train_iter:

            sum1 =sum1+1
            l,con_l, con2_l, fea_l= net(X)
            print('output',output.shape)
            _, predicted = torch.max(output.data, 1)


            #print('y---------',y )
            evaluation.append((predicted == y).tolist())
            f1_score=torchmetrics.functional.f1_score(predicted,y)

            #print('f1-score----------------------------',f1_score)
            predicted=predicted.cpu().numpy()
            predict.extend(predicted)

            #output=output.cpu().numpy()
            x=y.cpu().numpy()
            lables.extend(x)
           # print('predict---------------',y)

           # print('predicted-----------------', predicted)
            optimizer.zero_grad()
           # l = criterion(output, y)
            #l = comtrast_loss(output, x)

            l.backward()
            optimizer.step()
            loss_sum += l.item()
            #loss_sum =loss_sum.cpu().numpy()

            iter += 1
            pbar.set_description("Epoch %d, loss = %.2f" % (epoch, l.data))
            pbar.update(1)
    evaluation = [item for sublist in evaluation for item in sublist]
    predict,lables= np.array(predict),np.array(lables)



    train_loss = loss_sum / sum1

    # auc = roc_auc_score(predict,lable,multi_class='ovr')

    train_acc = sum(evaluation) / len(evaluation)

    current_lr = optimizer.param_groups[0]['lr']

    if not os.path.exists('./pre_train_save'):
        os.makedirs('./pre_train_save')

    print(f1)

    if epoch>=3:
        acc, f1, specificity, sensitivity, auc = compute_test(predict, lables)
        print("Epoch:", epoch,"train_loss=",train_loss, " train_acc =", acc, "specificity=", specificity,"sensitivity=",sensitivity,"f1=",f1,"roc=",auc)
    # scheduler.step()
    scheduler_warmup.step()
    val_loss = 0
    evaluation = []
    pred_v = []
    true_v = []
    val_predict = []
    val_x = []
    val_lable = []
    i=0


    net.eval()
    for X, y in test_iter:
        i=i+1
        output = net(X)

        _, predicted = torch.max(output.data, 1)

        evaluation.append((predicted == y).tolist())
        l = criterion(output, y)
           # l = comtrast_loss(output, y)



        predicted=predicted.cpu().numpy()
        val_predict.extend(predicted)
        x=y.cpu().numpy()
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

    val_acc, val_f1, val_specificity, val_sensitivity,val_auc = compute_test(val_predict, val_lable)
    print("val_loss =", val_loss, "val_acc =", val_acc, 'val_f1', val_f1, 'val_sprcificity', val_specificity,
            'val_senstivity', val_sensitivity, "val_auc = ", val_auc)

    running_acc = sum(evaluation) / len(evaluation)
    val_acc_list.append(running_acc)



    res = "bets results:" + \
          'val_epoch=_' + str(epoch) + \
          "val_loss"    + str(val_loss)+\
          "_accuracy=_" + str(val_acc) + \
          "_sensitivity=_" + str(val_sensitivity) + \
          "_specificity=_" + str(val_specificity) + \
          '_F1=_' + str(val_f1) +\
          '_auc=_'+str(val_auc)+'\n'

    if  val_f1 > best_f1:
        if not os.path.exists('save_model/%d'%target):
            os.makedirs('save_model/%d'%target)
        torch.save(net.state_dict(), 'save_model/%d/best_model.pth'%target)
        best_res = res
        best_f1 = val_f1
print(best_res)
finish_test_res = open("result/%d.txt" % (target), "a")
end_time=time.time()
finish_test_res.write(best_res)
str=" 总运行时间为"+str(end_time-strat_time)+"s"
finish_test_res.write(str)
print("Highest acc:", max(val_acc_list))
