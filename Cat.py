# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 09:09:27 2018

@author: 56576
"""

import os
from PIL import Image
import numpy as np
from keras.utils import np_utils #用于将Label向量化
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.optimizers import SGD,RMSprop,Adam
from keras.layers import Conv2D,MaxPool2D
from keras.models import load_model
import PicPreProcess as ppp


train_path = ppp.train_path
test_path = ppp.test_path
#test_path='D:/Cat/NewCat/'
file_dict = ppp.file_path
input_width = ppp.input_width
input_height = ppp.input_height
model_path = "D:/Cat1.1/model9.h5"

#ppp.Init()


#----------------训练集-----------------------
#将目录下的照片转化为一个列表
images1 = os.listdir(train_path)

#返回一个numpy数组类型的图片
def GetImages1(filename):
    
    #打开路径下的文件 并转化为RGB
    img = Image.open(train_path+filename).convert('RGB')
    return np.array(img)


#x_train和y_train都是列表 分别保存了训练集的输入和输出
x_train = []
y_train = []

#将转化为数组的图片加入训练的列表
for i in images1:
    x_train.append(GetImages1(i))

x_train = np.array(x_train)


#split('_')是将文件名以'_'分割 这样可以提取预处理完毕后的数据的标签(即分割后的数组的第一个部分)
for filename in images1:
    y_train.append(int(filename.split('_')[0]))

#将标签也加入数组
y_train = np.array(y_train)

    
# ---------------测试集-----------------------
#处理过程同上
images2 = os.listdir(test_path)
x_test = []
y_test = []

def GetImages2(filename):
    img = Image.open(test_path+filename).convert('RGB')
    return np.array(img)

for i in images2:
    x_test.append(GetImages2(i))

x_test = np.array(x_test)
for filename in images2:
    y_test.append(int(filename.split('_')[0]))
y_test = np.array(y_test) 

#将训练和测试的标签都由标签数字转化为向量 来进行交叉熵的优化
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


#将输入转化为浮点数 并位于0-1来提高准确率
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255



#-------------------神经网络------------------------
#卷积层1 深度为32 输入为预处理时生成的100x100的RGB图片
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(input_width, input_height, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


#model = load_model(file_dict+'model0.h5')
#for i in range (2,10):
#    model.fit(x_train,y_train,batch_size=15,epochs=1)
#    print (i)
#    model.save(file_dict+'model'+str(i)+'.h5')
#    print ('\n')

del model

model = load_model(model_path)

loss,acc = model.evaluate(x_test,y_test,batch_size=1)
print ("loss:",loss)
print ("acc:",acc)