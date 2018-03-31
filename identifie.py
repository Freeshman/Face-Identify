#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 20:24:18 2017

@author: hu-tom
"""
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import photo_data_generator
import cv2
from PIL import Image
camera_id=0
istrained=True
model="save/model.ckpt"
checkpoint_dir="save"
def weight_variable(shape,name):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name=name)
def bias_variable(shape,name):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=name)
def conv2d(x,W):
    #strides[0]=stirdes[3]=1
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#conv1 layer
def compute_accuracy(v_xs,v_ys):
	global prediction
	y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:0.5})
	correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:0.5})
	return result

#x_data = np.linspace(-1,1,300)[:,np.newaxis]
#noise  = np.random.normal(0,0.05,x_data.shape)
#y_data = np.square(x_data)-0.5+noise

xs = tf.placeholder(tf.float32,[None,784])#28x28
ys = tf.placeholder(tf.float32,[None,2])
keep_prob=tf.placeholder(tf.float32)
x_image=tf.reshape(xs,[-1,28,28,1])


#print(x_image.shape)#[n_samples,28,28,1]

#conv1 layer
W_conv1=weight_variable([3,3,1,32],name='wc1')#patch 5x5,in size 1,out size 32     
b_conv1=bias_variable([32],name='bc1')
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#tf.nn.relu for nonlinear output size28x28x32 
h_pool1=max_pool_2x2(h_conv1)#output size14x14x32

#conv2 layer
W_conv2=weight_variable([3,3,32,64],name='wc2')#patch 5x5,in size 32,out size 64     
b_conv2=bias_variable([64],name='bc2')
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)#tf.nn.relu for nonlinear output size28x28x32 
h_pool2=max_pool_2x2(h_conv2)#output size7x7x64

#conv3 layer
W_conv3=weight_variable([5,5,64,128],name='wc3')#patch 5x5,in size 32,out size 64     
b_conv3=bias_variable([128],name='bc3')
h_conv3=tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)#tf.nn.relu for nonlinear output size28x28x32 
h_pool3=max_pool_2x2(h_conv3)#output size7x7x128
#print(h_pool3)

#conv4 layer
W_conv4=weight_variable([5,5,128,256],name='wc4')#patch 5x5,in size 32,out size 64     
b_conv4=bias_variable([256],name='bc4')
h_conv4=tf.nn.relu(conv2d(h_pool3,W_conv4)+b_conv4)#tf.nn.relu for nonlinear output size28x28x32 
h_pool4=max_pool_2x2(h_conv4)#output size7x7x128
#print(h_pool4)

##conv5 layer
#W_conv5=weight_variable([5,5,256,512])#patch 5x5,in size 32,out size 64     
#b_conv5=bias_variable([512])
#h_conv5=tf.nn.relu(conv2d(h_pool4,W_conv5)+b_conv5)#tf.nn.relu for nonlinear output size28x28x32 
#h_pool5=max_pool_2x2(h_conv5)#output size7x7x128
#print(h_pool5)

#func1 layer
W_f1=weight_variable([2*2*256,1024],name='wf1')
b_f1=bias_variable([1024],name='bf1')
#[n_samples,7,7,64]->>[n_samples,7*7*64]
h_pool4_flat=tf.reshape(h_pool4,[-1,2*2*256])
h_f1=tf.nn.relu(tf.matmul(h_pool4_flat,W_f1)+b_f1)
h_f1_drop=tf.nn.dropout(h_f1,keep_prob)

#func2 layer
W_f2=weight_variable([1024,2],name='wf2')
b_f2=bias_variable([2],name='bf2')

prediction=tf.nn.softmax(tf.matmul(h_f1_drop,W_f2)+b_f2)

Cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(tf.clip_by_value(prediction,1e-8,1.0)),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(Cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(Cross_entropy)
#cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
#train_step = tf.train.AdamOptimizer(1e-5).minimize(cost_func)

init = tf.global_variables_initializer()
# Create a saver.
saver = tf.train.Saver()
sess = tf.Session()


sess.run(init)
if istrained:
    saver=tf.train.import_meta_graph('save/model.ckpt.meta')
    saver.restore(sess,model)      
mnist=photo_data_generator.dataresize("./tr_data")
for i in range(100):
    batch_xs,bathc_ys = mnist[0][:][:]
    sess.run(train_step,feed_dict={xs:batch_xs,ys:bathc_ys,keep_prob:0.5})
    if i % 50 ==0:
        accu=compute_accuracy(mnist[2][0],mnist[2][1])
        print(accu)
        if(accu>=0.98):
            break
        '''
        for j in range(5):
            im_test=mnist[0][0][j]
            im_test=im_test.reshape(1,784)
            print('pre=',prediction.eval(feed_dict={xs:im_test,keep_prob:0.5}, session=sess))
            print('anser=',mnist[0][1][j])
        '''
        #print(sess.run(y_pre))
saver_path = saver.save(sess,model)  # 将模型保存到save/model.ckpt文件
print("Model saved in file:", saver_path)
cap=cv2.VideoCapture(camera_id)
classfier=cv2.CascadeClassifier(r'/PATH/TO/YOUR/OpenCV-tmp/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
if cap.isOpened():
    print('open the cap')
else:
    print('open faild!')
while cap.isOpened():        
    ok,frame=cap.read()
    if not ok:
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    faceRects=classfier.detectMultiScale(gray,1.2, 3, cv2.CASCADE_SCALE_IMAGE,(32,32))
    if len(faceRects)>0:
        for faceRect in faceRects:
            x,y,w,h=faceRect           
            #img_name = './%d.jpg'%(num)  
            img = frame[y : y + h , x : x + w ]
            img = Image.fromarray(img)
            img =img.convert('L').resize((28,28))
            width,hight=img.size
            img = np.asarray(img,dtype='float64')/256.   
            img = img.reshape(1, hight*width)     
            pre = prediction.eval(feed_dict={xs:img,keep_prob:0.5},session=sess)
            if pre[0][1]>=0.7:  
                color=(0,255,0)
                cv2.rectangle(frame, (x, y), (x + w , y + h ), color, thickness = 2)
                cv2.putText(frame,'Me',(x, y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)                                  
            else:  
                color=(255,0,0)
                cv2.rectangle(frame, (x , y ), (x + w , y + h ), color, thickness = 2)  
                cv2.putText(frame,'Others',(x, y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)  
            #cv2.imwrite(img_name, image)  
            #cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),color,2)
         #   num += 1  
        #if num > pic_num:   #如果超过指定最大保存数量退出循环  
         #   break  
    cv2.imshow('identify',frame)
    c=cv2.waitKey(1)
    if c&0xff==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
#sess.close()
