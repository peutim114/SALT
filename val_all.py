import pickle

import sklearn
import torch
import torch.nn as nn
import numpy as np

import os

import torchmetrics
from sklearn.model_selection import train_test_split

from LSTMVITCNN import MY_NETC
from lstmattention import MY_NET
from utils import compute_test



def val(target,path):
    model_save_dir = 'Transformer_model/best_w.pth'

    import argparse
    device = "cuda"


    path = 'D:/新建文件夹/balancedata'
    test_data = pickle.load(open(os.path.join(path, 'patient%d_data.pkl'%target),'rb'))
    # cv_split = pickle.load(open(os.path.join(path, 'cv_split_5_fold_chb%d.pkl' % (target)), 'rb'))
    # data = cv_split[str(3)]
    # test_data ,test_label= data['val'], data['val_label'],

    test_label = pickle.load(open(os.path.join(path, 'patient%d_label.pkl'%target),'rb'))
    #print(test_label)
    test_data=np.array(test_data)
    test_label=np.array(test_label)
    test_data=test_data.reshape(-1,16,256)

    #打乱数据
    permutation = np.random.permutation(len(test_data))
    test_data = test_data[permutation]
    test_label=test_label[permutation]
    #数据传入cuda
    # x_train, test_data, y_train, test_label = train_test_split(test_data, test_label, test_size=0.8)
    test_data = torch.tensor(test_data, dtype=torch.float).to(device)
    test_label = torch.tensor(test_label, dtype=torch.long).to(device)
    #参数设置
    batch_size = 128
    test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)


    #导入模型


    #net=convnext_tiny(2).to('cuda')
    #net = resnet18(classification=True).to('cuda')
    #net = nn.DataParallel(net)
    #checkpoint = torch.load('save_model/last_model.pth')
    net = MY_NETC(image_size=32,  # 图像大小
                  patch_size=8,  # patch大小（分块的大小）
                  num_classes=2,  # imagenet数据集1000分类
                  embed_dim=256,
                  seq_len=0,  # position embedding的维度
                  hidden_dim=2048,
                  num_heads=256,
                  num_layers=4,
                  dropout=0.1,
                  classification=True).to('cuda')
    # net = MY_NET(image_size=32,  # 图像大小
    #              patch_size=8,  # patch大小（分块的大小）
    #              num_classes=2,  # imagenet数据集1000分类
    #              embed_dim=256,
    #              seq_len=0,  # position embedding的维度
    #              hidden_dim=2048,
    #              num_heads=256,
    #              num_layers=4,
    #              dropout=0.1,
    #              classification=True).to('cuda')
    # net= convnext_base(2).to('cuda
    checkpoint = torch.load('./vit/%d/best_model.pth'%target)
    #checkpoint = torch.load('H:/code/code-1wzw/save_model/1/last_model.pth')
    net.load_state_dict(checkpoint,strict=False)
    #net.load_state_dict(checkpoint['state_dict'], strict=False)
    criterion = nn.CrossEntropyLoss().to(device)
    evaluation = []
    val_loss = 0
    val_predict = []
    val_x = []
    val_lable = []
    #开始测试
    torch.cuda.empty_cache()
    with torch.no_grad():
        net.eval()
        i=0
        for X, y in test_iter:
            output = net(X)
            i+=1
            _, predicted = torch.max(output.data, 1)
            evaluation.append((predicted == y).tolist())
          # print(predicted)
            l = criterion(output, y)
            print(l)
            val_loss += l.item()
            #print("y==",y)
            #print("predicted",predicted)
            predicted = predicted.cpu().numpy()
            val_predict.extend(predicted)
            x = y.cpu().numpy()
            val_lable.extend(x)
        val_predict = np.array(val_predict)

        val_lable = np.array(val_lable)
        val_loss = val_loss/i

    accc=sklearn.metrics.accuracy_score(val_lable, val_predict,  normalize=True, sample_weight=None)

    val_acc, val_f1, val_specificity, val_sensitivity, roc = compute_test(val_predict,val_lable)
    print('val_oredict',val_predict)
    print('val_lable',val_lable)
    print("val_loss =", val_loss, "val_acc =", val_acc, 'val_f1', val_f1, 'val_sprcificity', val_specificity,
     'val_senstivity', val_sensitivity,'roc =',roc)
    print("accc= ",accc)
    res = "bets results:" + str(target)+\
          "val_loss" + str(val_loss) + \
          "_accuracy=_" + str(val_acc) + \
          "_sensitivity=_" + str(val_sensitivity) + \
          "_specificity=_" + str(val_specificity) + \
          '_F1=_' + str(val_f1) + \
          '_auc=_' + str(roc) + '\n'
    if not os.path.exists('val_result' ):
                os.makedirs('val_result')
    finish_test_res = open("val_result/vit跨病人平均.txt", "a")  # 所有病人写入一个文件
    finish_test_res.write(res)
    return target, float(val_acc), float(val_sensitivity), float(val_specificity), float(val_f1), float(roc)
path = 'D:/新建文件夹/balancedata'
prtrain = 1
to_acc, to_sen, to_spe, to_f1, to_auc = [],[],[],[],[]

for target in range (22, 25):
    target, acc, sen, spe, f1, auc = val(target,path)
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

finish_test_res = open("val_result/vit跨病人平均.txt", "a")  # 所有病人写入一个文件
finish_test_res.write(men_res)