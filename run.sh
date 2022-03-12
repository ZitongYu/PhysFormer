module load pytorch/1.9
pip install --user imgaug
pip install --user thop

srun -N 1 -n 1 -c 2 -t 36:00:00 --gres=gpu:a100:1 -p gpusmall --account=project_2001654 python3 ~/code_PhysFormer_CVPR_github/train_Physformer_160_VIPL.py

#srun -N 1 -n 1 -c 2 -t 00:15:00 --gres=gpu:a100:1 -p gputest --account=project_2001654 python3 ~/code_PhysFormer_CVPR_github/inference_OneSample_VIPL_PhysFormer.py 




