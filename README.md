# Kaggle_DigitRecognizer
----
This is a simple CNN network built by Tensorflow 2. It is built to recognize 10 different numbers(0-9).                
                                

## Requirements
----
Tensorflow >=2.0               
Pandas                
Sklearn       

## Data
----
Download the MNIST dataset. It contains train.csv and test.csv. In the network, the data is converted into 28*28 as input.        

## Network 
---
It is a simple CNN network.                 
_________________________________________________________________           
Layer (type)                 Output Shape              Param #            
=================================================================           
conv2d (Conv2D)              multiple                  320       
_________________________________________________________________           
max_pooling2d (MaxPooling2D) multiple                  0         
_________________________________________________________________            
conv2d_1 (Conv2D)            multiple                  18496                    
_________________________________________________________________              
max_pooling2d_1 (MaxPooling2 multiple                  0         
_________________________________________________________________            
conv2d_2 (Conv2D)            multiple                  73856     
_________________________________________________________________
flatten (Flatten)            multiple                  0                       
_________________________________________________________________            
dense (Dense)                multiple                  73792     
_________________________________________________________________              
dense_1 (Dense)              multiple                  650       
_________________________________________________________________              
dropout (Dropout)            multiple                  0         
=================================================================
Total params: 167,114
Trainable params: 167,114
Non-trainable params: 0
_________________________________________________________________      

## Train 
'python mnist.py -train /PATH/TO/CSV -test /PATH/TO/CSV'              
Example:
'python mnist.py -train Dataset/train.csv -test Dataset/test.csv -lr 0.001 -output result.csv -e 15'            
The training process can be viewed using tensorboard
'tensorboard --logdir=logs/'
![](https://github.com/Yunying-Chen/Kaggle_DigitRecognizer/blob/master/IMG/acc.png ''acc.png'')
![](https://github.com/Yunying-Chen/Kaggle_DigitRecognizer/blob/master/IMG/loss.png ''loss.png'')

## Eval
With Kaggle's raw dataset, it can reach 0.99285 score. 
