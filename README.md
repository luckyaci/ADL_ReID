# ADL_ReID

requirements：
torch==1.3.1
torchvision==0.4.2
tensorboard
future
fire
tqdm

#train
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='gpu_num' python train.py train --trainset_name market --save_dir='save_dir'

#test
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='gpu_num' python test.py train --trainset_name market --save_dir='save_dir'
