# SURFMNet-pytorch
A pytorch implementation of: "Unsupervised Deep Learning for Structured Shape Matching" [[link](http://openaccess.thecvf.com/content_ICCV_2019/papers/Roufosse_Unsupervised_Deep_Learning_for_Structured_Shape_Matching_ICCV_2019_paper.pdf)]

## Installation
This implementation runs on python >= 3.7, use pip to install dependencies:
```
pip3 install -r requirements.txt
```

## Download data
Download data using `download_data.sh` script.
```
bash download_data.sh
```

## Usage
Use the `train.py` script to train the FMNet network.
```
usage: train.py [-h] [--lr LR] [--b1 B1] [--b2 B2] [-bs BATCH_SIZE] [--n-epochs N_EPOCHS] [--feat-dim FEAT_DIM] [--dim-basis DIM_BASIS] [-nv N_VERTICES] [-nb NUM_BLOCKS] [--wb WB]
                [--wo WO] [--wl WL] [--wd WD] [--sub-wd SUB_WD] [-d DATAROOT] [--save-dir SAVE_DIR] [--n-cpu N_CPU] [--no-cuda] [--checkpoint-interval CHECKPOINT_INTERVAL]
                [--log-interval LOG_INTERVAL]

Lunch the training of SURFMNet model.

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               adam: learning rate
  --b1 B1               adam: decay of first order momentum of gradient
  --b2 B2               adam: decay of first order momentum of gradient
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        size of the batches
  --n-epochs N_EPOCHS   number of epochs of training
  --feat-dim FEAT_DIM   Input feature dimension
  --dim-basis DIM_BASIS
                        number of eigenvectors used for representation. The first 500 are precomputed and stored in input
  -nv N_VERTICES, --n-vertices N_VERTICES
                        Number of vertices used per shape
  -nb NUM_BLOCKS, --num-blocks NUM_BLOCKS
                        number of resnet blocks
  --wb WB               Bijectivity penalty weight
  --wo WO               Orthogonality penalty weight
  --wl WL               Laplacian commutativity penalty weight
  --wd WD               Descriptor preservation via commutativity penalty weight
  --sub-wd SUB_WD       Percentage of subsampled vertices used to compute descriptor preservation commutativity penalty
  -d DATAROOT, --dataroot DATAROOT
                        root directory of the dataset
  --save-dir SAVE_DIR   root directory of the dataset
  --n-cpu N_CPU         number of cpu threads to use during batch generation
  --no-cuda             Disable GPU computation
  --checkpoint-interval CHECKPOINT_INTERVAL
                        interval between model checkpoints
  --log-interval LOG_INTERVAL
                        interval between logging train information
```

### Example
```
python3 train.py -d ./data/faust/train_mini -bs 2 -n-epochs 2
```
