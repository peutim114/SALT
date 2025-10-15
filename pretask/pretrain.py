import pickle
import torch
import torch.nn as nn
import numpy as np
import tqdm
import random
import time
import os, shutil

from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
import torch.nn.functional as F
from transform import Transform
import argparse
from pretask.care_module import CARE

#from lstmattention import ViT_LSTM, MY_NET

import multiprocessing
multiprocessing.set_start_method('spawn')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-permute', '--transform_function_1', type=str)
parser.add_argument('-noise', '--transform_function_2', type=str)
arg = parser.parse_args()
torch.set_default_tensor_type(torch.FloatTensor)
#device
target=24
device = "cuda"
strat_time=time.time()
#模型保存路径
model_save_dir = 'moblie_vit_pretrain'
os.makedirs(model_save_dir, exist_ok=True)
log_file = "%s_%s_%s.log" % (arg.transform_function_1, arg.transform_function_2, time.strftime("%m%d%H%M"))

log_templete = {"acc": None,
                    "cm": None,
                    "f1": None,
                "per F1":None,
                "epoch":None,
                    }
#加载数据
path = 'D:/新建文件夹/balancedata'
data = pickle.load(open(os.path.join(path, 'all_data.pkl'),'rb'))
label = pickle.load(open(os.path.join(path, 'all_label.pkl' ),'rb'))
data=np.array(data)
label=np.array(label)

#打乱数据
permutation = np.random.permutation(len(data))
data = data[permutation]
label = label[permutation]
#数据分割 按照训练测试0.3的比例划分
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.01)
#数据传入cuda
x_train = torch.tensor(x_train, dtype=torch.float).to(device)
x_test = torch.tensor(x_test, dtype=torch.float).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

#保存最好的和最后一轮模型
def save_ckpt(state, is_best, model_save_dir, message='best_w.pth'):
    current_w = os.path.join(model_save_dir, 'latest_w.pth')
    best_w = os.path.join(model_save_dir, message)
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)

def comtrast_loss(x, criterion):
    LARGE_NUM = 1e9
    temperature = 0.1
    x = F.normalize(x, dim=-1)

    num = int(x.shape[0] / 2)
    hidden1, hidden2 = torch.split(x, num)

    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = torch.arange(0 , num).to('cuda')
    masks = F.one_hot(torch.arange(0, num), num).to('cuda')


    logits_aa = torch.matmul(hidden1, hidden1_large.T) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(hidden2, hidden2_large.T) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(hidden1, hidden2_large.T) / temperature
    logits_ba = torch.matmul(hidden2, hidden1_large.T) / temperature
    # print(labels)
    #
    # print(torch.cat([logits_ab, logits_aa], 1).shape)

    loss_a = criterion(torch.cat([logits_ab, logits_aa], 1),
        labels)
    loss_b = criterion(torch.cat([logits_ba, logits_bb], 1),
        labels)
    loss = torch.mean(loss_a + loss_b)
    return loss, labels, logits_ab
def transform(x, mode):
    x_ = x.cpu().numpy()


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

criterion = nn.CrossEntropyLoss().to(device)

net = CARE(param_momentum = 0.99,total_iters=400).to(device)
img = torch.randn(1, 3, 256, 256)


batch_size = 128

optimizer = torch.optim.SGD(net.parameters(), lr=0.001 * (batch_size / 64), momentum=0.9, weight_decay=0.00001)

epochs = 50
lr_schduler = CosineAnnealingLR(optimizer, T_max=epochs - 10, eta_min=0.05)#default =0.07
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=lr_schduler)
optimizer.zero_grad()
optimizer.step()
scheduler_warmup.step()


train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)

val_acc_list = []
n_train_samples = x_train.shape[0]
iter_per_epoch = n_train_samples // batch_size + 1
best_acc = -1
err = []
best_err = 1
margin = 1
print('prtrain---------------------start')

for epoch in range(epochs):
    net.train()
    loss_sum = 0
    evaluation = []
    iter = 0
    with tqdm.tqdm(total=iter_per_epoch) as pbar:
        error_counter = 0

        for X, y in train_iter:
            trans = []
            trans1 = []
            trans2 = []
            for i in range(X.shape[0]):
                t1 = transform(X[i], arg.transform_function_1)
                trans1.append(t1)
            for i in range(X.shape[0]):
                t2 = transform(X[i], arg.transform_function_2)
                trans2.append(t2)
            trans.append(trans1)
            trans.append(trans2)
          #  print('trans',np.array(trans).shape)


           # trans = np.concatenate(trans)
            trans = torch.tensor(trans, dtype=torch.float, device="cuda")
            #print(trans.shape)
            l, con_loss, con2_loss  = net(trans)
            #print('output----------',output)
            optimizer.zero_grad()

           # l, lab_con, log_con = comtrast_loss(output, criterion)
            # _, log_p = torch.max(log_con.data,1)
            # evaluation.append((log_p == lab_con).tolist())
            l.backward()
            optimizer.step()
            loss_sum += l
            iter += 1
            pbar.set_description("Epoch %d, loss = %.2f" % (epoch, l.data))
            pbar.update(1)
        err = l.data
    evaluation = [item for sublist in evaluation for item in sublist]

    if not os.path.exists('save_model'):
        os.makedirs('save_model')
    torch.save(net.state_dict(), 'save_model/best_model.pth' )


    current_lr = optimizer.param_groups[0]['lr']
    #print("Epoch:", epoch,"lr:", current_lr, "error:", error, " train_loss =", loss_sum.data)
    print("Epoch:", epoch, "lr:", current_lr,  " train_loss =", loss_sum.data)
    scheduler_warmup.step()
    state = {"state_dict": net.state_dict(), "epoch": epoch}

end_time =time.time()
str=" 总运行时间为"+str(end_time-strat_time)+"s"
print(str)