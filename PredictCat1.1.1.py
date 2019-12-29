# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 19:38:37 2018

@author: 56576
"""

import os
from PIL import Image
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.optimizers import SGD,RMSprop,Adam
from keras.layers import Conv2D,MaxPool2D
from keras.models import load_model
import PicPreProcess
from PIL import Image
import matplotlib.pyplot as plt

TestDir = 'D:/Cat1.1/GY/'
TempDir = 'D:/Cat1.1/GYTemp/'
ModelDir = 'D:/Cat1.1/model0.h5'
input_width = PicPreProcess.input_width
input_height = PicPreProcess.input_height


#将读入的图片转化为200x200的图片 并写入Temp文件夹
def PrePic(picname):
    
    img = Image.open(TestDir+picname)
    new_img = img.resize((input_width,input_height),Image.BILINEAR)
    
    #没有文件 创造文件
    if not os.path.exists(TempDir):
        print ("Creating Temp")
        os.mkdir(TempDir)
    new_img.save(os.path.join(TempDir,os.path.basename(picname)))


def GetImages(filename):
    
    img = Image.open(TempDir+filename).convert('RGB')
    return np.array(img)


def PredictCat(picname):
    
    PrePic(picname)
    x_test = []
    
    x_test.append(GetImages(picname))
    
    x_test = np.array(x_test)
    x_test= x_test.astype('float32')
    x_test /= 255
    
    keras.backend.clear_session()

    #卷积层1 深度为32 输入为预处理时生成的200x200的RGB图片
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(input_width,input_height,3)))
    #model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'))
    
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    
    model = load_model(ModelDir)
    classes = model.predict_classes(x_test)
    return classes


def Identify(num):
    
    if (num==1):
        return "布偶猫"
    if (num==2):
        return "孟买猫"
    if (num==3):
        return "暹罗猫"
    if (num==4):
        return "英短"
    if (num==5):
        return "橘猫"
    else:
        print (num)
         
print ("模型路径:",ModelDir)
for name in os.listdir(TestDir):
    
 #   img = Image.open(TestDir+name)
    
    pre = PredictCat(name)
    pre = Identify(pre)
    print ('--------------------')
    
    
    #显示图片
    print ("图片:",name)
    img=Image.open(TestDir+name)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    #img.show() 
    print ("模型判断为:",pre)