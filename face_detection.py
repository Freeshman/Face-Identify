#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 19:38:42 2018

@author: hu-tom
"""

import cv2
win_name='face'
camera_id=1
#def usbvideo(win_name,camera_id):
pic_num=500
num=0
cv2.namedWindow(win_name)
cap=cv2.VideoCapture(camera_id)
classfier=cv2.CascadeClassifier(r'/PATH/TO/YOUR/OpenCV-tmp/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
color=(0,255,0)
if cap.isOpened():
    print('open the cap')
else:
    print('open faild!')
count=0
while cap.isOpened():
    count=count+1        
    ok,frame=cap.read()
    if not ok:
        break
    c=cv2.waitKey(1)
    if c&0xff==ord('c') or count>130:
        count=0
        print("detecte face...")
        gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        faceRects=classfier.detectMultiScale(gray,1.2, 3, cv2.CASCADE_SCALE_IMAGE,(32,32))
        if len(faceRects)>0:
            for faceRect in faceRects:
                x,y,w,h=faceRect           
                img_name = './new/n%d.jpg'%(num)  
                image = gray[y - 10: y + h + 10, x - 10: x + w + 10]  
                cv2.imwrite(img_name, image)  
                cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),color,2)
                num += 1  
            if num > pic_num:   #如果超过指定最大保存数量退出循环  
                break  
    elif c&0xff==ord('q'):
        break
    cv2.imshow(win_name,frame)
    
cap.release()
cv2.destroyAllWindows()
