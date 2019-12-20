# Interesting-semantic-segmentation
## Description
A beginner in semantic segmentation, this repository records the models I wrote during learning image semantic segmentation, some experiments, and various possible ideas.  
***
**Directories**  
**data**: Store temporary data folders, including model weights files and object files  
**logs**: Store tensorboard information folder, used to observe loss and acc change information during training  
**logs/log.txt**: Store log when use `run.sh` to training  
**models**: Store model files  
**test**: Test models or datasets  
**utils**: Some common utils and preprocessing utils  
**run**.sh: one-click run script, the usage is shown below  

# Use environment
**System:**  
+ Ubuntu 14.04  
  
**Dependencies:**  
+ python=3.6
+ tensorflow-gpu==1.4.1(cuda=8.0)
+ keras==2.0.8

# Models
The following models are currently implemented:
- FCN
- Unet
- PSPnet
- SegNet  
  
To do moels:  
deeplabv3+,DenseNet

# Usage:
**Train:**  
Use script:  
```sh
./run FCN-dataset1-keras.py
```  
The program will run in the background, the output information and errors will be saved in `log.txt` in logs directory.  
or you can **use python**
```python
python FCN-dataset1-keras.py
```
Model weights and history object will be saved in the **data directory**.

**Evaluation:**
```python
python evaluation.py
```
You need to modify the model weights and history object path in `evaluation.py`, the loss, acc, val-loss, val-acc, mIou and predict picture from test data will be saved in data directory. 
# Dataset
dataset1: https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view?usp=sharing  
PASCAL VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  
