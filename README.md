# Deep-learning
记录了学习图像语义分割过程中编写的模型.

文件及文件夹说明：
- notes:存放笔记文件夹  
- models:存放模型文件夹
- data:存放数据文件夹
- experiments:存放其他额外实验文件夹
- test:存放测试文件夹
- utils:存放通用工具文件夹
- logs:存放tensorboard运行图像和代码运行时的输出信息
- FCN-dataset1-generator.py:使用FCN模型训练dataset1数据集,并且使用generator逐步导入数据.
- FCN-dataset1-keras.py: 最初版本的FCN,使用keras
- FCN-VOC21012-keras.py: 使用FCN训练VOC2012数据集,但是效果不太好,是因为预处理部分没有做好,有待改进
- models.py:存放模型文件
- Unet_ResNet-dataset1.py:使用Unet结合Resnet训练dataset1,效果并不好.
- Unet-dataset1.py: 单纯使用Unet对dataset1进行分割,效果较好.

运行示例:
```python
python3 -u Unet-dataset1.py  > ./logs/log.txt 2&1 &
```
- -u表示不启用缓存,实时打印信息到日志文件,如果不加-u,日志文件不会实时刷新  
- \>./logs/log.txt 表示重定向到日志文件  
- 2>&1 表示将标准错误转变成标准输出,将错误信息也输出到日志文件中(0-> stdin, 1->stdout, 2->stderr)  
