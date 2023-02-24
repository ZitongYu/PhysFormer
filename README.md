# PhysFormer
Main code of **CVPR2022 paper "PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer"**    [[.pdf]](https://arxiv.org/pdf/2111.12082.pdf)   

![image](https://github.com/ZitongYu/PhysFormer/blob/main/framework.png)  

module load pytorch/1.9

pip install --user imgaug

Training on VIPL-HR:
-----

```
python train_Physformer_160_VIPL.py
```

Testing on One sample on VIPL-HR:
----
1. Download the test data [[Google Drive]](https://drive.google.com/file/d/1n1TpMQfU-OkZdJglEJyFp-vGo9JXbgsT/view?usp=sharing)   
2. Run the model inference code (with trained checkpoint 'Physformer_VIPL_fold1.pkl' [[Google Drive]](https://drive.google.com/file/d/1jBSbM88fA-beaoVi8ILFyL0SvVVMA9c9/view?usp=sharing)) to get the predicted rPPG signal clips:
```
python inference_OneSample_VIPL_PhysFormer.py
```
3. Calculate the HR error with the file 'Inference_HRevaluation.m' using Matlab (You can also easily use python script to implement it). 


Citation
------- 
If you find it is useful in your research, please cite:  
         
         @inproceedings{yu2021physformer,
            title={PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer},
            author={Yu, Zitong and Shen, Yuming and Shi, Jingang and Zhao, Hengshuang and Torr, Philip and Zhao, Guoying},
            booktitle={CVPR},
            year={2022}
          }
          
          @article{yu2023physformer++,
           title={PhysFormer++: Facial Video-based Physiological Measurement with SlowFast Temporal Difference Transformer},
           author={Yu, Zitong and Shen, Yuming and Shi, Jingang and Zhao, Hengshuang and Cui, Yawen and Zhang, Jiehua and Torr, Philip and Zhao, Guoying},
           journal={International Journal of Computer Vision (IJCV)},
           pages={1--24},
           year={2023}
         }

If you use the VIPL-HR datset, please cite:  
         
         @article{niu2019rhythmnet,
           title={Rhythmnet: End-to-end heart rate estimation from face via spatial-temporal representation},
           author={Niu, Xuesong and Shan, Shiguang and Han, Hu and Chen, Xilin},
           journal={IEEE Transactions on Image Processing},
           year={2019}
         }
