import csv
import numpy as np
import argparse
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split

def Load_TrainData(path):
    train = pd.read_csv(path)
    train_label = train.iloc[:,0].to_numpy()
    train_data =  train.iloc[:,1:].to_numpy()
    train_data = train_data.reshape(train_data.shape[0],28,28,1)
    train_data = train_data/255
    return train_data,train_label

def Load_TestData(path):
    test_data = pd.read_csv(path)
    test_data = test_data.to_numpy()
    test_data = test_data.reshape(test_data.shape[0],28,28,1)
    test_data = test_data/255
    return test_data

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32,kernel_size=[3,3],activation=tf.nn.relu)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2,2])
        self.conv2 = tf.keras.layers.Conv2D(filters=64,kernel_size=[3,3],activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2,2])
        self.conv3 = tf.keras.layers.Conv2D(filters=128,kernel_size=[3,3],activation=tf.nn.relu)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=64,activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10,activation=tf.nn.softmax)
        self.dropout = tf.keras.layers.Dropout(0.5)
    def call(self,inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dropout(x)
        output = self.dense2(x)
        return output

parser = argparse.ArgumentParser()
parser.add_argument("-train", "--train_paths", type=str,
                    required=True,
                    help="The path of training data file.")
parser.add_argument("-test", "--test_paths", type=str,
                    help="The path of testing data file.")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                    help="Learning rate.")
parser.add_argument("-output", "--output_path", type=str, default="submission.csv",
                    help="Learning rate.")
parser.add_argument("-e", "--epochs", type=int, default=10,
                    help="Num of epochs to train.")
args = parser.parse_args()

train_data,train_label = Load_TrainData(args.train_paths)
x_train,x_val,y_train,y_val = train_test_split(train_data,train_label,test_size=0.1,random_state=2)
test_data = Load_TestData(args.test_paths)

model = Model()
model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'] )

callbacks = [
    keras.callbacks.TensorBoard(log_dir="logs/log",
                                histogram_freq=1)
]
model.fit(x_train,y_train,epochs=15,batch_size=64, callbacks=callbacks,validation_data=(x_val,y_val))
y_pred = model.predict(test_data)
y_pred = np.argmax(y_pred,axis=1)
results = pd.Series(y_pred,name="Label")
submission = pd.concat([pd.Series(range(1,test_data.shape[0]),name = "ImageId"),results],axis = 1)
submission.to_csv(args.output_path,index=False)