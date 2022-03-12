'''
  title={PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer},
  author={Yu, Zitong and Shen, Yuming and Shi, Jingang and Zhao, Hengshuang and Torr, Philip and Zhao, Guoying},
  conference={CVPR 2022}
'''
from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import scipy.io as sio

from model import ViT_ST_ST_Compact3_TDC_gra_sharp

 
from Loadtemporal_data import VIPL_train, VIPL_test, Normaliztion, ToTensor, RandomHorizontalFlip 

from TorchLossComputer import TorchLossComputer


import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb


# feature  -->   [ batch, patch*patch, channel, T]
# torch.Size([1, 64, 64, 60, 60])
def FeatureMap2Heatmap( x, Score1, Score2, Score3):
    ## initial images 
    ## initial images 
    org_img = x[0,:,32,:,:].cpu()  
    org_img = org_img.data.numpy()*128+127.5
    org_img = org_img.transpose((1, 2, 0))
    #org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(args.log+'/'+args.log + '_x_visual.jpg', org_img)
 

    # [B, head, 640, 640]
    org_img = Score1[0, 1].cpu().data.numpy()*3000
    org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(args.log+'/'+'Score1_head1.jpg', org_img)
    
    
    org_img = Score2[0, 1].cpu().data.numpy()*3000
    org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(args.log+'/'+'Score2_head1.jpg', org_img)
    
    
    org_img = Score3[0, 1].cpu().data.numpy()*3000
    org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(args.log+'/'+'Score3_head1.jpg', org_img)






class Neg_Pearson(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson,self).__init__()
        return
    def forward(self, preds, labels):       # all variable operation
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                # x
            sum_y = torch.sum(labels[i])               # y
            sum_xy = torch.sum(preds[i]*labels[i])        # xy
            sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            
            loss += 1 - pearson
            
        loss = loss/preds.shape[0]
        return loss



class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


# main function
def train_test():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    #os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log+'/'+ args.log+'_log.txt', 'w')


          
    print("cross-validastion fold 1:")
    log_file.write("cross-validastion fold 1:")
    log_file.write("\n")
    log_file.flush()
    

    # Dataset root
    VIPL_root_list = args.input_data + '/VIPL_frames/'
    VIPL_train_list = 'VIPL_fold1_train.txt'
    #VIPL_test_list = 'VIPL_fold1_test.txt'
    
    

    print('train from scratch!\n')
    log_file.write('train from scratch!\n')
    log_file.flush()
    
    

    model = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(160,128,128), patches=(4,4,4), dim=96, ff_dim=144, num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
     
    model = model.cuda()

    lr = args.lr
    optimizer1 = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
    #optimizer1 = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=args.step_size, gamma=args.gamma)
        
    criterion_reg = nn.MSELoss()
    criterion_L1loss = nn.L1Loss()
    criterion_class = nn.CrossEntropyLoss()
    criterion_Pearson = Neg_Pearson()    
    
    echo_batches = args.echo_batches
    
    # a --> Pearson loss; b --> frequency loss
    a_start = 0.1
    b_start = 1.0
    exp_a = 0.5
    exp_b = 5.0

    
    # train
    for epoch in range(args.epochs):  
        scheduler1.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        loss_rPPG_avg = AvgrageMeter()
        loss_peak_avg = AvgrageMeter()
        loss_kl_avg_test = AvgrageMeter()
        loss_hr_mae = AvgrageMeter()
        
        model.train()
        
        #pdb.set_trace()

        VIPL_trainDL = VIPL_train(VIPL_train_list, VIPL_root_list, transform=transforms.Compose([Normaliztion(),RandomHorizontalFlip(), ToTensor()]))
        dataloader_train = DataLoader(VIPL_trainDL, batch_size=args.batchsize, shuffle=True, num_workers=4)    # batchsize = 4

        for i, sample_batched in enumerate(dataloader_train):
            # get the inputs
            inputs, ecg = sample_batched['video_x'].cuda(), sample_batched['ecg'].cuda()  
            clip_average_HR, frame_rate = sample_batched['clip_average_HR'].cuda(), sample_batched['frame_rate'].cuda()

            optimizer1.zero_grad()
            
            gra_sharp = 2.0
            #gra_sharp = np.sqrt(24.0)
            
            rPPG, Score1, Score2, Score3 = model(inputs, gra_sharp)
            
            rPPG = (rPPG-torch.mean(rPPG)) /torch.std(rPPG)	 	# normalize2
            loss_rPPG = criterion_Pearson(rPPG, ecg)
            
            clip_average_HR = (clip_average_HR-40)    # [40, 180]
            
            fre_loss = 0.0
            kl_loss = 0.0
            train_mae = 0.0
            for bb in range(inputs.shape[0]):
                loss_distribution_kl, fre_loss_temp, train_mae_temp = TorchLossComputer.cross_entropy_power_spectrum_DLDL_softmax2(rPPG[bb], clip_average_HR[bb], frame_rate[bb], std=1.0)  # std=1.1
                fre_loss = fre_loss + fre_loss_temp
                kl_loss = kl_loss + loss_distribution_kl
                train_mae = train_mae + train_mae_temp
            fre_loss = fre_loss/inputs.shape[0]
            kl_loss = kl_loss/inputs.shape[0]
            train_mae = train_mae/inputs.shape[0]
            
            
            if epoch >25:
                a = 0.05
                b = 5.0
            else:
                # exp descend
                a = a_start*math.pow(exp_a, epoch/25.0)
                # exp ascend
                b = b_start*math.pow(exp_b, epoch/25.0)
            
            a = 0.1
            #b = 1.0
            
            loss =  a*loss_rPPG + b*(fre_loss+kl_loss)
            #loss =  0.1*loss_rPPG + fre_loss
            
            
            loss.backward()
            optimizer1.step()
            
            
            n = inputs.size(0)
            loss_rPPG_avg.update(loss_rPPG.data, n)
            loss_peak_avg.update(fre_loss.data, n)
            loss_kl_avg_test.update(kl_loss.data, n)
            loss_hr_mae.update(train_mae, n)
            

            if i % echo_batches == echo_batches-1:    # print every 50 mini-batches

                # visulization
                visual = FeatureMap2Heatmap(inputs, Score1, Score2, Score3)
                
                # log written
                log_file.write('epoch:%d, batch:%3d, lr=%f, sharp=%.3f, a=%.3f, NegPearson= %.4f, b=%.3f, kl= %.4f, fre_CEloss= %.4f, hr_mae= %.4f' % (epoch + 1, i + 1, lr, gra_sharp, a, loss_rPPG_avg.avg, b, loss_kl_avg_test.avg, loss_peak_avg.avg, loss_hr_mae.avg))
                log_file.write("\n")
                log_file.flush()

                
                # show the ecg images
                results_rPPG = []
                y1 = 2*rPPG[0].cpu().data.numpy()
                y2 = ecg[0].cpu().data.numpy()  # +1 all positive
                results_rPPG.append(y1)
                results_rPPG.append(y2)
                sio.savemat( args.log+'/'+'rPPG.mat' , {'results_rPPG': results_rPPG})
                
        
        
        log_file.write("\n")        
        log_file.write('epoch:%d, batch:%3d, lr=%f, sharp=%.3f, a=%.3f, NegPearson= %.4f, b=%.3f, kl= %.4f, fre_CEloss= %.4f, hr_mae= %.4f' % (epoch + 1, i + 1, lr, gra_sharp, a, loss_rPPG_avg.avg, b, loss_kl_avg_test.avg, loss_peak_avg.avg, loss_hr_mae.avg))
        log_file.write("\n")
        log_file.write("\n")
        log_file.flush()        
                
                                                                       
                 
              
        ## save model with corresponding epoch with the lowest val reg loss in one fold
        torch.save(model.state_dict(), args.log+'/'+'Physformer'+'_%d_%d.pkl' % (index,epoch))
        

    print('Finished Training')
    log_file.close()
    
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=2, help='the gpu id used for predict')
    parser.add_argument('--input_data', type=str, default="/scratch/project_2003204/") # /scratch/project_2003204/VIPL_frames_Matlab/
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  #default=0.0001
    parser.add_argument('--batchsize', type=int, default=4, help='batchsize')
    parser.add_argument('--step_size', type=int, default=50, help='stepsize of optim.lr_scheduler.StepLR, how many epochs lr decays once')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=100, help='how many batches display once')  # 100
    parser.add_argument('--epochs', type=int, default=25, help='total training epochs')
    parser.add_argument('--log', type=str, default="Physformer_TDC07_sharp2_hid96_head4_layer12_VIPL", help='log and save model name')

    args = parser.parse_args()
    train_test()
