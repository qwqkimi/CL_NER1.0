# CL_NER1.0
刑法名词实体识别

## requirements 
* tensorflow 1.15.0
* keras contribute 2.0.8 (https://github.com/keras-team/keras-contrib)
* h5py 
* pickle

## usage
‘CL_NER/data/annotation.zip’ 应解压到 'CL_NER/code' 中

### config
在config.py中可以修改train_data路径和test_data路径等参数 数据集默认存放路径为‘annotation/output/train_data/'

### run
train 
  python3 train.py

test 
  python3 val.py
  
### bug
TypeError: Tensors in list passed to 'values' of 'ConcatV2' Op have types [bool, float32] that don't

see https://blog.csdn.net/u011740601/article/details/103800575
