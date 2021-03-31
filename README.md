# ADL_ReID
We will standardize our code and add a detailed description before the paper is published !

## requirements
torch==1.3.1

torchvision==0.4.2


## Datasets
**Market-1501, DukeMTMC-reID, and MSMT17 should be organized as follows:**
<pre>
.
+-- data
|   +-- market
|       +-- bounding_box_train
|       +-- query
|       +-- bounding_box_test
|   +-- duke
|       +-- bounding_box_train
|       +-- query
|       +-- bounding_box_test
|   +-- msmt17
|       +-- train
|       +-- test
|       +-- list_train.txt
|       +-- list_val.txt
|       +-- list_query.txt
|       +-- list_gallery.txt
+ -- other files in this repo
</pre>
## train
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='gpu_num' python train.py train --trainset_name market --save_dir='save_dir'

## test
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='gpu_num' python test.py train --trainset_name market --save_dir='save_dir'
