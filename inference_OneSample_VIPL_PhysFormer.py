from __future__ import print_function, division
import matplotlib.pyplot as plt
import argparse,os
import torch
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import ViT_ST_ST_Compact3_TDC_gra_sharp

from Loadtemporal_data_test import VIPL, Normaliztion, ToTensor

import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb
import scipy.io as sio

from TorchLossComputer import TorchLossComputer
 

def FeatureMap2Heatmap( x, Score1, Score2, Score3):
    ## initial images 
    ## initial images 
    org_img = x[0,:,32,:,:].cpu()  
    org_img = org_img.data.numpy()*128+127.5
    org_img = org_img.transpose((1, 2, 0))
    #org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(args.log+'/'+args.log + 'visual.jpg', org_img)
 

    # [B, head, 640, 640]
    org_img = Score1[0, 1].cpu().data.numpy()*4000
    org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(args.log+'/'+'Score1_head1.jpg', org_img)
    
    
    org_img = Score2[0, 1].cpu().data.numpy()*4000
    org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(args.log+'/'+'Score2_head1.jpg', org_img)
    
    
    org_img = Score3[0, 1].cpu().data.numpy()*4000
    org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(args.log+'/'+'Score3_head1.jpg', org_img)




# main function
def train_test():

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log+'/'+ args.log+'_log.txt', 'w')
    
    
    
    VIPL_root_list = args.input_data + '/VIPL_frames/'
    VIPL_test_list = 'VIPL_fold1_test1.txt'
    
        
    dict_list = []
    dict_list.append('Physformer_VIPL_fold1.pkl')
    

    print('evaluation~ forward!\n')
    
    
    model = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(160,128,128), patches=(4,4,4), dim=96, ff_dim=144, num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
    
    gra_sharp = 2.0
    
    
    model = model.cuda()
    model.load_state_dict(torch.load(dict_list[0]))
   
    model.eval()
  
  
    VIPL_testDL = VIPL(VIPL_test_list, VIPL_root_list, transform=transforms.Compose([Normaliztion(), ToTensor()]))
    dataloader_test = DataLoader(VIPL_testDL, batch_size=1, shuffle=False, num_workers=2)
  
    
    results_rPPG = []
    results_HR_pred = []
    with torch.no_grad():
        for i, sample_batched in enumerate(dataloader_test):
            
            inputs = sample_batched['video_x'].cuda()
            clip_average_HR, frame_rate = sample_batched['clip_average_HR_peaks'].cuda(), sample_batched['framerate'].cuda()
            
            rPPG_long = torch.randn(1).cuda()
     
            #HR = 0.0
            for clip in range(inputs.shape[1]):
                rPPG, Score1, Score2, Score3 = model(inputs[:,clip,:,:,:,:], gra_sharp)
                
                
                rPPG = rPPG[0, 30:30+160]
                
                #HR_predicted = TorchLossComputer.cross_entropy_power_spectrum_forward_pred(rPPG, frame_rate)+40
                #HR += HR_predicted
                
                rPPG_long = torch.cat((rPPG_long, rPPG),dim=0)
            
            #HR = HR/inputs.shape[1]
    
            log_file.write('\n sample number :%d \n' % (i+1))
            log_file.write('\n')
            log_file.flush()
            
            visual = FeatureMap2Heatmap(inputs[:,inputs.shape[1]-1,:,:,:,:], Score1, Score2, Score3)
  
            
            ## save the results as .mat 
            results_rPPG.append(rPPG_long[1:].cpu().data.numpy())
            #results_HR_pred.append([HR.cpu().data.numpy(), clip_average_HR[0].cpu().data.numpy()])
  
    
    # visual and save 
    #visual = FeatureMaP2Heatmap(x_visual, x_visual3232, x_visual1616)
    sio.savemat( args.log+'/'+args.log+ '.mat' , {'outputs_rPPG_concat': results_rPPG})   
    #sio.savemat( args.log+'/'+args.log+ '_HR.mat' , {'outputs_HR': results_HR_pred})     

       
 

    print('Finished val')
    log_file.close()
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=2, help='the gpu id used for predict')
    parser.add_argument('--input_data', type=str, default="/scratch/project_2003204/") # /scratch/project_2003204/VIPL_frames_Matlab/
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  #default=0.0001
    parser.add_argument('--step_size', type=int, default=12, help='stepsize of optim.lr_scheduler.StepLR, how many epochs lr decays once')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')
    parser.add_argument('--epochs', type=int, default=10, help='total training epochs')
    parser.add_argument('--log', type=str, default="Inference_Physformer_TDC07_sharp2_hid96_head4_layer12_VIPL", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')

    args = parser.parse_args()
    train_test()
