# UKANCNet：UKANCNet is proposed, an enhanced UKAN variant for windmill blade defect identification, featuring: (1) Replace the decoder with fine to coarse feature integration to minimize information loss. (2) Apply Attention mechanism in suppressing background noise while highlighting defects. (3) Optimize loss to enable hyperparameter auto-tuning to reduce computational costs. 

## 环境配置：
* Python3.8
* Pytorch1.10.1
* * 最好使用GPU训练
* 详细环境配置见`requirements.txt`

背景名称background mask 颜色[0, 0, 0], 对应标签  0
缺陷名称crack mask 颜色[128, 0, 0], 对应标签 1
缺陷名称broke mask 颜色[0, 128, 0], 对应标签2
缺陷名称wear mask 颜色[128, 128, 0], 对应标签 3
叶片名称blade mask 颜色[0, 0, 128], 对应标签 4

     ## 文件结构：
```
  ├── src: 搭建UKANCNet模型代码
  ├── train_utils: 训练、验证以及多GPU训练相关模块
  ├── my_dataset.py: 自定义dataset用于读取数据集
  ├── train.py: 以单GPU为例进行训练
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  └── compute_mean_std.py: 统计数据集各通道的均值和标准差
```

## 训练方法
* 确保提前准备好数据集
* 若要使用单GPU或者CPU训练，直接使用train.py训练脚本

## 注意事项
* 在使用训练脚本时，注意要将`--data-path`设置为自己存放数据文件夹所在的**根目录**，修改`--num-classes`、`--data-path`和`--weights`即可，其他代码尽量不要改动

