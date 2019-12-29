# -*- coding: utf-8 -*-
"""
Spyder Editor

Created By Chen
2018 5 11

This is a temporary script file.
"""

from PIL import Image
import numpy as np
import os

#将训练的图片进行预处理

#所有待处理的图片路径
in_path = 'D:/Cat/NewCat/'

#总文件路径
file_path = "D:/Cat1.1/"
#训练集路径
train_path="D:/Cat1.1/TrainSet/"
#测试集路径
#test_path='D:/Cat/NewCat/'
test_path="D:/Cat1.1/TestSet/"
#测试集与训练集分割的大小
split_num = 0.99
#图片分辨率
input_width = 100
input_height = 100

#统一图片类型并按顺序命名
def renamesJPG(filepath,kind):
    images = os.listdir(filepath)
    
    #原名字改为 种类_序号.jpg
    i=1
    for name in images:
        os.rename(filepath+name,filepath+kind+'_'+str(i)+'.jpg')
        i=i+1
        
        
#返回一个numpy数组类型的图片
def GetImages(filename):
    
    #打开路径下的文件 并转化为RGB
    img = Image.open(train_path+filename).convert('RGB')
    return np.array(img)



#预处理图片 将大小调整为width*height的RGB图片 并随机地以spilit_num生成训练和测试集
#生成时会对图片进行随机旋转来增强模型的健壮性
def PicPreprocess(width,height,in_path,train_path,test_path):
    img  = Image.open(in_path)
    
    is_train_exists = os.path.exists(train_path)
    is_test_exists = os.path.exists(test_path)
    
    if not is_train_exists:
        print ("Creating Train Set")
        os.mkdir(train_path)
    if not is_test_exists:
        print ("Creating Test Set")
        os.mkdir(test_path)
        
    
    #将图片转化为目标尺寸的RGB图片 默认为100*100
    new_img = img.resize((width,height),Image.BILINEAR).convert('RGB')

    t = []
    
    #temp是不含有路径信息和图片格式的照片名       
    temp = os.path.basename(in_path)
    temp = temp.split('.')[0]
        
    #将照片进行随机的旋转并写入训练集和测试集
    for i in range(0,4):
        choice = np.random.random()
        #retote将照片进行逆时针旋转
        t.append(new_img.rotate(i*90))                        
        
        if (choice <= split_num):
            #若随机数小于split_num 则写入训练集
            t[i].save(train_path+str(temp)+'_'+str(i)+'.jpg')
        else:
            t[i].save(test_path+str(temp)+'_'+str(i)+'.jpg')
                       
  
def Init():
    for name in os.listdir(in_path):
        picpath = os.path.join(in_path,os.path.basename(name))
        PicPreprocess(width = input_width,
                      height = input_height,
                      in_path = picpath,
                      train_path = train_path,
                      test_path = test_path)           