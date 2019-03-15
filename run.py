# -*- coding: utf-8 -*-

import os
import cv2
import time
import schedule
from datetime import datetime

path = 'model/ssd_inception_v2_coco_2018_01_28/freeze'
net = cv2.dnn.readNetFromTensorflow(os.path.join(path, 'frozen_inference_graph.pb'), os.path.join(path, 'ssd_inception_v2.pbtxt'))

def test_job():
    frame = cv2.imread('data/images/20190122-103653.jpg')
    rows, cols = frame.shape[0], frame.shape[1]
    net.setInput(cv2.dnn.blobFromImage(frame, size=(600, 480), swapRB=True, crop=False))
    out = net.forward()
    for detection in out[0,0,:,:]:
        score = float(detection[2])
        print(score)
        if score > 0.3:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
    cv2.imshow('', frame)
    cv2.waitKey(0)


def job():
    video_cap = cv2.VideoCapture(1)
    ret, frame = video_cap.read()
    if not ret:
        print('[INFO] 拍摄图像失败')
    else:
        net.setInput(cv2.dnn.blobFromImage(frame, size=(600, 480), swapRB=True, crop=False))
        out = net.forward()
        out = net.forward()
        count = 0
        rows, cols = frame.shape[0], frame.shape[1]
        for detection in out[0,0,:,:]:
            score = float(detection[2])
            print(score)
            if score > 0.3:
                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
                count += 1

        file_name = 'data/{}.jpg'.format(datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
        cv2.imwrite(file_name, frame)
        with open('data/count.csv', 'wt') as f:
            f.write('{},{}'.format(file_name, count))
        print('[INFO] 拍摄图像保存成功 {}'.format(file_name))


if __name__ == '__main__':
    schedule.every(1).minutes.do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)
    
    # 本地测试模型加载
    # test_job()