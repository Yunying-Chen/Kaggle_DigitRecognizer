# Kaggle_DigitRecognizer
This is a simple CNN network built by Tensorflow 2. It is built to recognize 10 different numbers(0-9).                
                                

## Requirements
Tensorflow >=2.0               
Pandas                
Sklearn       

## Data
Download the MNIST dataset. It contains train.csv and test.csv. In the network, the data is converted into 28*28 as input.        

## Network 
It is a simple CNN network.                 
![Image](https://github.com/Yunying-Chen/Kaggle_DigitRecognizer/blob/master/IMG/network.png)

## Train 
```
python mnist.py -train /PATH/TO/CSV -test /PATH/TO/CSV
```                        
Example:                 
```
python mnist.py -train Dataset/train.csv -test Dataset/test.csv -lr 0.001 -output result.csv -e 15
```



The training process can be viewed using tensorboard            
```
tensorboard --logdir=logs/
```                         
![Image](https://github.com/Yunying-Chen/Kaggle_DigitRecognizer/blob/master/IMG/acc.png)
![Image](https://github.com/Yunying-Chen/Kaggle_DigitRecognizer/blob/master/IMG/loss.png)

## Eval
With Kaggle's raw dataset, it can reach 0.99285 score. 
![Image](https://github.com/Yunying-Chen/Kaggle_DigitRecognizer/blob/master/IMG/score.png)
