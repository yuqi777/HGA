from ast import arg
from asyncio.log import logger
import imp
from traceback import print_tb
import torch
import torch.nn as nn
import torch.nn.functional as F
from parse import parse_args
from pprint import pformat
from thop import profile

import matplotlib.pyplot as plt
import random
from data_loader.build import build_loader
from utils import *
from models import MLP, GNN
import warnings
import numpy as np
import torch.optim as optim
from time import time
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.metrics import recall_score,precision_score

args_config = parse_args()
# logger = getLogger(*args_config)
csvwriter = CSVWriter(args_config.csv_path)

yy_train_loss = []
yy_val_loss = []
yy_train_acc = []
yy_val_acc = []
yy_test_loss=[]
yy_test_acc = []

xx = []

def train(epoch, gnn, gnn_optimizer, train_loader):
    total, correct, train_acc = 0, 0, 0
    losses = AverageMeter()

    gnn.train()

    progressbar = tqdm(enumerate(train_loader), desc=f'epoch:[{epoch}]', total=len(train_loader))
    for batch_idx, batch in progressbar:            
            gnn_optimizer.zero_grad()     
            
            adj, mask, emb, labels = batch 
        
            outputs = gnn(emb.cuda(), adj.cuda(), mask.cuda()) 
            #Calculate 
            # flops, params = profile(gnn, inputs=(emb.cuda(), adj.cuda(), mask.cuda() ))
            # print('FLOPs = ' + str(flops/1000**3) + 'G')
            # print('Params = ' + str(params/1000**2) + 'M')   
            bce_loss_batch = gnn.bce_loss(outputs, labels.float().cuda()) 
            l2_loss_batch = gnn._l2_loss() 
            loss_batch = (l2_loss_batch + bce_loss_batch) 
            losses.update(loss_batch.item())

            gnn_optimizer.zero_grad()
            loss_batch.backward()        
            gnn_optimizer.step()

            preds = gnn.predict(outputs)
            preds = torch.argmax(preds, 1)

            correct += preds.eq(torch.argmax(labels.cuda(), 1)).sum().item()
            total += len(labels)
            train_acc = correct*1.0 / total

            progressbar.set_postfix({
            'batch_loss': "{:.4f}".format(loss_batch.item()), 
            'avg_loss': "{:.4f}".format(losses.avg),
            'avg_acc': "{:.4f}".format(train_acc),
        })
    return losses.avg, train_acc
    
def test(epoch, gnn, test_loader):
    total, correct, acc = 0, 0, 0
    losses = AverageMeter()
    progressbar = tqdm(enumerate(test_loader), desc='test', total=len(test_loader))
    with torch.no_grad():
        for _,batch in progressbar:            
            adj, mask, emb, labels = batch   
            outputs = gnn(emb.cuda(), adj.cuda(), mask.cuda())    
            bce_loss_batch = gnn.bce_loss(outputs, labels.float().cuda()) 
            l2_loss_batch = gnn._l2_loss() 
            loss_batch = (l2_loss_batch + bce_loss_batch) 
            losses.update(loss_batch.item())    
            

            preds = gnn.predict(outputs)
            preds = torch.argmax(preds, 1)

            correct += preds.eq(torch.argmax(labels.cuda(), 1)).sum().item()
            total += len(labels)
            test_acc = correct*1.0 / total
            # print(labels,preds)
            recall=recall_score(torch.argmax(labels.cuda(), 1).cpu().numpy(),preds.detach().cpu().numpy(),zero_division=1)
            precision=precision_score(torch.argmax(labels.cuda(), 1).cpu().numpy(),preds.detach().cpu().numpy(),zero_division=1)
     
            progressbar.set_postfix({
            'batch_loss': "{:.4f}".format(loss_batch.item()), 
            'avg_loss': "{:.4f}".format(losses.avg),
            'avg_acc': "{:.4f}".format(test_acc),
            'recall': "{:.4f}".format(recall),
            'precision': "{:.4f}".format(precision)
        })
    return losses.avg, test_acc ,recall,precision




def main():
    SEED = 77

    random.seed(SEED)
    torch.manual_seed(SEED)
    allrecall = []
    allprecision=[]

    """initialize args and dataset"""
    
    """set gpu id"""
    args_config.gpu = args_config.gpu and torch.cuda.is_available()
    device = torch.device("cuda" if args_config.gpu else "cpu")

    if args_config.gpu:
        print('Using GPU')
        torch.cuda.manual_seed(SEED)
        torch.cuda.set_device(args_config.gpu_id)
    
    train_adj, train_embed, train_y, val_adj, val_embed, val_y, test_adj, test_embed, test_y = load_data(args_config.dataset)
    
    # preprocessing
    train_adj, train_mask = preprocess_adj(train_adj)
    train_embed = preprocess_features(train_embed)
    val_adj, val_mask = preprocess_adj(val_adj)
    val_embed = preprocess_features(val_embed)
    test_adj, test_mask = preprocess_adj(test_adj)
    test_embed = preprocess_features(test_embed)
    print("Data Loaded")

    print("Build Data Loader")
    train_loader, test_loader, val_loader = build_loader(args_config=args_config, train_adj=train_adj, train_mask=train_mask, train_emb=train_embed, train_y=train_y,
                                             test_adj=test_adj, test_mask=test_mask, test_emb=test_embed, test_y=test_y,
                                             val_adj=val_adj, val_mask=val_mask, val_emb=val_embed, val_y=val_y)
    
    gnn = GNN(args_config.input_dim, train_y.shape[1]).cuda() 
    gnn_optimizer = torch.optim.Adam(gnn.parameters(), args_config.learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(gnn_optimizer, step_size=40, gamma=0.2)

    total_params = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
    logger.info(f"model total trainable parameters: {total_params:,}")    
    logger.info(f"current experiment parameters are:")
    logger.info(pformat(args_config)+"\n")
    best_acc = 0.0
    best_acc1 = 0.0
    best_epoch = 0
    brep=0
    logger.info('==>Start training...')
    

    for epoch in range(0, args_config.epochs):
        xx.append(epoch)
        lr_list = [param_groups['lr'] for param_groups in gnn_optimizer.param_groups]
        logger.info(f"current LR: {lr_list}")

        train_loss, train_acc = train(epoch, gnn, gnn_optimizer, train_loader)
        val_loss, val_acc ,_,_= test(epoch, gnn, val_loader)
        test_loss, test_acc,test_rec,test_pre = test(epoch, gnn, test_loader)
        
        


        lr_scheduler.step()
        allrecall.append(test_rec)
        allprecision.append(test_pre)
        kwargs = {'epoch':epoch,'train_loss': train_loss, 'train_acc': train_acc, 'test_loss':test_loss, 'test_acc':test_acc}

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            best_epoch = epoch
        
        logger.info(f"Epoch: {epoch} - train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - test_loss:{test_loss:.4f} - test_acc: {val_acc:.4f}, Best acc is {best_acc:.4f} achieved at epoch {best_epoch}\n")
        csvwriter.write(**kwargs)
        if val_acc > best_acc1:
            best_acc1 = val_acc
            es = 0
            torch.save(gnn.state_dict(), 'model_weight.pt')
        else:
            es += 1
            print("Counter {} of 20".format(es))

            if es > args_config.early_stopping:
                print("Early stopping with best_acc: ", best_acc, "and val_acc for this epoch: ", val_acc, "...")
                brep=epoch
                break
        yy_train_loss.append(train_loss)      
        yy_train_acc.append(train_acc)
        
        yy_val_loss.append(val_loss)      
        yy_val_acc.append(val_acc)
        
        yy_test_loss.append(test_loss)      
        yy_test_acc.append(test_acc)
 
    logger.info("==> Training finished!!!")
    a=0
    b=0
    for i in range(len(allrecall)):
        a = a + allrecall[i]
        b = b + allprecision[i]
    a = a / len(allrecall)
    b = b / len(allrecall)
    # print(allrecall)
    print('precision is {:.4f}'.format(b))
    print('recall is {:.4f}'.format(a))
    print('F-SCORE is {:.4f}'.format((2*a*b)/(a+b)))
    if len(allrecall)!=args_config.epochs:
        xx.pop()
    plt.plot(xx, yy_train_loss, "r", label="train_loss")

    plt.legend()
    plt.savefig("loss.png")
    plt.close()
    plt.plot(xx, yy_train_acc, "y", label="train_acc")
    plt.plot(xx, yy_val_acc, "g", label="valid_acc")
    plt.plot(xx, yy_test_acc, "b", label="test_acc")
    plt.legend()
    plt.savefig("acc.png")
    plt.close()

    



if __name__ == '__main__':
    main()
    