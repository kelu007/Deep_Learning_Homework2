# Image Classification Based on ResNet-18

## Checkpoint Path

Link: https://pan.baidu.com/s/1UHAwEMBER2hSdcBVbm9X7Q 

Key: i2zw

## Data Augmentation

`python data_augment.py`

output: `./img`

## Train

`python train.py -net resnet18 -gpu -method none/cutout/cutmix/mixup `

output: 
`./checkpoint`
`./runs`

## Test

`python test.py -net resnet18 -weights ${ckpt_path_you_want_to_test}`
