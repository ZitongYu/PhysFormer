from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math



clip_frames = 160   



class Normaliztion (object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        video_x, clip_average_HR, ecg_label, frame_rate = sample['video_x'],sample['clip_average_HR'], sample['ecg'],sample['frame_rate']
        new_video_x = (video_x - 127.5)/128
        return {'video_x': new_video_x, 'clip_average_HR':clip_average_HR, 'ecg':ecg_label, 'frame_rate':frame_rate}




class RandomHorizontalFlip (object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        video_x, clip_average_HR, ecg_label, frame_rate = sample['video_x'],sample['clip_average_HR'], sample['ecg'],sample['frame_rate']

        h, w = video_x.shape[1], video_x.shape[2]
        new_video_x = np.zeros((clip_frames, h, w, 3))


        p = random.random()
        if p < 0.5:
            #print('Flip')
            for i in range(clip_frames):
                # video 
                image = video_x[i, :, :, :]
                image = cv2.flip(image, 1)
                new_video_x[i, :, :, :] = image
                
                
                
            return {'video_x': new_video_x, 'clip_average_HR':clip_average_HR, 'ecg':ecg_label, 'frame_rate':frame_rate}
        else:
            #print('no Flip')
            return {'video_x': video_x, 'clip_average_HR':clip_average_HR, 'ecg':ecg_label, 'frame_rate':frame_rate}



class ToTensor (object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        video_x, clip_average_HR, ecg_label, frame_rate = sample['video_x'],sample['clip_average_HR'], sample['ecg'],sample['frame_rate']

        # swap color axis because
        # numpy image: (batch_size) x depth x H x W x C
        # torch image: (batch_size) x C x depth X H X W
        video_x = video_x.transpose((3, 0, 1, 2))
        video_x = np.array(video_x)
        
        clip_average_HR = np.array(clip_average_HR)
        
        frame_rate = np.array(frame_rate)
        
        
        return {'video_x': torch.from_numpy(video_x.astype(np.float)).float(), 'clip_average_HR': torch.from_numpy(clip_average_HR.astype(np.float)).float(), 'ecg': torch.from_numpy(ecg_label.astype(np.float)).float(), 'frame_rate': torch.from_numpy(frame_rate.astype(np.float)).float()}



# train
class VIPL_train (Dataset):

    def __init__(self, info_list, root_dir, transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):
        #print(self.landmarks_frame.iloc[idx, 0])
        video_path = os.path.join(self.root_dir, str(self.landmarks_frame.iloc[idx, 0]))
        
        start_frame = self.landmarks_frame.iloc[idx, 1]
        
        frame_rate  = self.landmarks_frame.iloc[idx, 2]
        clip_average_HR  = self.landmarks_frame.iloc[idx, 3]
        
        
        p = random.random()
        if p < 0.5 and start_frame==61:    # sampling aug     p < 0.5
            ecg_label = self.landmarks_frame.iloc[idx, 5:5+160].values 
            video_x, clip_average_HR, ecg_label_new = self.get_single_video_x_aug(video_path, start_frame, clip_average_HR, ecg_label)
            
        else: 
            video_x = self.get_single_video_x(video_path, start_frame)
            ecg_label_new = self.landmarks_frame.iloc[idx, 5:5+160].values 
 
        
        sample = {'video_x': video_x, 'frame_rate':frame_rate, 'ecg':ecg_label_new, 'clip_average_HR':clip_average_HR}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_video_x(self, video_path, start_frame):
        video_jpgs_path = video_path

        video_x = np.zeros((clip_frames, 128, 128, 3))
        
        size_crop = np.random.randint(16)
        
        log_file = open('temp.txt', 'w')
        
        image_id = start_frame
        for i in range(clip_frames):
            s = "%05d" % image_id
            image_name = 'image_' + s + '.png'

            # face video 
            image_path = os.path.join(video_jpgs_path, image_name)
            
            
            log_file.write(image_path)
            log_file.write("\n")
            log_file.flush()
                
            
            tmp_image = cv2.imread(image_path)
            #cv2.imwrite('test111.jpg', tmp_image)
            
            if tmp_image is None:    # It seems some frames missing 
                tmp_image = cv2.imread(self.root_dir+'p30/v1/source2/image_00737.png')
            
            
            tmp_image = cv2.resize(tmp_image, (128+size_crop, 128+size_crop), interpolation=cv2.INTER_CUBIC)[(size_crop//2):(128+size_crop//2), (size_crop//2):(128+size_crop//2), :]
            
            video_x[i, :, :, :] = tmp_image  
                        
            image_id += 1
   
        return video_x
    
   
    def get_single_video_x_aug(self, video_path, start_frame, clip_average_HR, ecg_label):
        video_jpgs_path = video_path

        video_x = np.zeros((clip_frames, 128, 128, 3))
        ecg_label_new = np.zeros(clip_frames)
        
        
        size_crop = np.random.randint(16)
        
        if clip_average_HR>88:  #  halve
            clip_average_HR = clip_average_HR/2
            for tt in range(clip_frames):
                if tt%2 == 0:
                    image_id = start_frame + tt//2
                    s = "%05d" % image_id
                    image_name = 'image_' + s + '.png'
                    image_path = os.path.join(video_jpgs_path, image_name)
                    tmp_image = cv2.imread(image_path)
                    if tmp_image is None:    # It seems some frames missing 
                        tmp_image = cv2.imread(self.root_dir+'p30/v1/source2/image_00737.png')
                    ecg_label_new[tt] = ecg_label[tt//2]
                    
                else:
                    image_id1 = start_frame + tt//2
                    image_id2 = image_id1+1
                    s = "%05d" % image_id1
                    image_name = 'image_' + s + '.png'
                    image_path = os.path.join(video_jpgs_path, image_name)
                    tmp_image1 = cv2.imread(image_path)
                    s = "%05d" % image_id2
                    image_name = 'image_' + s + '.png'
                    image_path = os.path.join(video_jpgs_path, image_name)
                    tmp_image2 = cv2.imread(image_path)
                    if tmp_image1 is None:    # It seems some frames missing 
                        tmp_image1 = cv2.imread(self.root_dir+'p30/v1/source2/image_00737.png')
                    if tmp_image2 is None:    # It seems some frames missing 
                        tmp_image2 = cv2.imread(self.root_dir+'p30/v1/source2/image_00737.png')
                    
                    tmp_image = tmp_image1//2+tmp_image2//2    # mean linear interpolation
                    ecg_label_new[tt] = ecg_label[tt//2]/2+ecg_label[tt//2+1]/2
                
                
                tmp_image = cv2.resize(tmp_image, (128+size_crop, 128+size_crop), interpolation=cv2.INTER_CUBIC)[(size_crop//2):(128+size_crop//2), (size_crop//2):(128+size_crop//2), :]
                
                image_x_aug = tmp_image
                video_x[tt, :, :, :] = image_x_aug  
            
                    
        else:     # double
            clip_average_HR = clip_average_HR*2
            for tt in range(clip_frames):
                image_id = start_frame + tt*2
                s = "%05d" % image_id
                image_name = 'image_' + s + '.png'
                image_path = os.path.join(video_jpgs_path, image_name)
                tmp_image = cv2.imread(image_path)
                if tmp_image is None:    # It seems some frames missing 
                    tmp_image = cv2.imread(self.root_dir+'p30/v1/source2/image_00737.png')
                
                tmp_image = cv2.resize(tmp_image, (128+size_crop, 128+size_crop), interpolation=cv2.INTER_CUBIC)[(size_crop//2):(128+size_crop//2), (size_crop//2):(128+size_crop//2), :]
                
                image_x_aug = tmp_image
                video_x[tt, :, :, :] = image_x_aug  
                
                # approximation
                if tt<80:
                    ecg_label_new[tt] = ecg_label[tt*2]
                else:
                    ecg_label_new[tt] = ecg_label_new[tt-80]
            
        
   
        return video_x, clip_average_HR, ecg_label_new
    
    


