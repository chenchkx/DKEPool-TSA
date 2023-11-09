#coding=utf-8

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from scipy.io import loadmat
from utils_ps import separate_data
from utils_ps import load_psdata
# from models.graphcnn import GraphCNN
# from sopool_attn.graphcnn import GraphCNN
from gcpool.graphcnn import GraphCNN

criterion = nn.CrossEntropyLoss()




def performance(predicted, expected):
    predicted = np.array(predicted.cpu() , dtype='uint8').squeeze(1)
    res =(predicted ^expected)#亦或使得判断正确的为0,判断错误的为1
    r = np.bincount(res)
    tp_list = ((predicted)&(expected))
    fp_list = (predicted&(~expected))
    tp_list=tp_list.tolist()
    fp_list=fp_list.tolist()
    tp=tp_list.count(1)
    fp=fp_list.count(1)
    tn = r[0]-tp
    fn = (len(res) - r[0])-fp
    tnr = tn/(tn + fp)
    tpr = tp/(tp + fn)
    F1=(2*tp)/(2*tp+fn+fp)
    acc=(tp+tn)/(tp+tn+fp+fn)
    # recall = tp / (tp + fn)
    return F1, acc,tnr,tpr

def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        #compute loss
        loss = criterion(output, labels)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    
    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    return acc_test, pred



def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="CASE39",
                        help='name of dataset (default: CASE39)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='number of hidden units (default: 32)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--fac_dim', type=int, default=1,
                        help='dimensionality of the factorized matrices')
    parser.add_argument('--rep_dim', type=int, default=1000,
                        help='dimensionality of the representations')
    parser.add_argument('--filename', type = str, default = "10flod.txt",
                        help='output file')
    parser.add_argument('--fo_type', type=int, default=0,
                        help='first order type, 0: mean operation, 1: summation operation')
    parser.add_argument('--so_type', type=int, default=0,
                        help='first order type, 0: cov operation, 1: bilinear operation')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(10) #设置CPU占用核数

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    num_classes = 2

    train_graphs, test_graphs, train_label, test_label = load_psdata(args.dataset, args.fold_idx)

    model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes,
                     args.fac_dim, args.rep_dim,args.final_dropout,  args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type,
                     args.fo_type, args.so_type,device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    path_ = str('state/')+str(args.dataset) + '_' + str(args.fold_idx) +'_' + str(args.batch_size) + '_'+ str(args.hidden_dim) + '_'+  str(args.lr) + '_.pth'

    max_acc = 0.0
    max_F1 = 0.0
    max_tnr = 0.0
    max_tpr = 0.0
    max_state_dict = model.state_dict()
    for epoch in range(1, args.epochs + 1):

        avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
        acc_test, pred_test = test(args, model, device, train_graphs, test_graphs, epoch)
        # print(model.state_dict())
        F1, acc, tnr, tpr = performance(pred_test, test_label)
        if acc > max_acc:
            max_acc = acc
            max_F1 = F1
            max_tnr = tnr
            max_tpr = tpr
            max_state_dict = model.state_dict()

        print("performance F1, acc, tnr, tpr : %f %f %f %f" % (max_acc, max_F1, max_tnr, max_tpr))

        print("")

        print(model.eps)

    torch.save(max_state_dict, path_)
    # model.load_state_dict(torch.load(path_))
    with open(str('result/')+str(args.dataset)+'_' + 'F1' + '_'+ str(args.batch_size) + '_'+ str(args.hidden_dim) + '_'+  str(args.epochs)+'_' + str(args.lr)+ '_results.txt', 'a+') as f:
        f.write(str(max_F1) + '\n')
    with open(str('result/')+str(args.dataset)+'_' + 'ACC' + '_'+ str(args.batch_size) + '_'+ str(args.hidden_dim) + '_'+ str(args.epochs)+'_' + str(args.lr)+ '_results.txt', 'a+') as f:
        f.write(str(max_acc) + '\n')
    with open(str('result/')+str(args.dataset)+'_' + 'TNR' + '_'+ str(args.batch_size) + '_'+ str(args.hidden_dim) + '_'+ str(args.epochs)+'_' + str(args.lr)+ '_results.txt', 'a+') as f:
        f.write(str(max_tnr) + '\n')
    with open(str('result/')+str(args.dataset)+'_' + 'TPR' + '_'+ str(args.batch_size) + '_'+ str(args.hidden_dim) + '_'+ str(args.epochs)+'_' + str(args.lr)+ '_results.txt', 'a+') as f:
        f.write(str(max_tpr) + '\n')

if __name__ == '__main__':
    main()
