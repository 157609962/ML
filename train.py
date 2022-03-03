import cv2
import numpy as np
from train import get_hog, svmPredict
import time

hog = get_hog()
model_path = 'svm_model.xml'
model = cv2.ml.SVM_load(model_path)
object = ['wet garbage', 'recyclable garbage', 'harmful garbage', 'other garbage']

def imag_recog():
    imag = cv2.imread('D:/yolo/yolov5-master/data/images/22.jpg')
    imag_or = imag.copy()
    imag = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    imag = cv2.resize(imag, (100, 100))
    imag_hog = hog.compute(imag)
    prediction = svmPredict(model, imag_hog[None,:])
    print('The classification of the garbage is \"'+object[int(prediction)]+' \"')
    cv2.putText(imag_or, object[int(prediction)], (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('result', imag_or)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def vedio_recog():
    cap = cv2.VideoCapture(0)
    while 1:
        ret, frame = cap.read()
        imag = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imag = cv2.resize(imag, (100, 100))
        imag_hog = hog.compute(imag)
        prediction = svmPredict(model, imag_hog[None,:])
        cv2.putText(frame, object[int(prediction)], (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('result', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    t0 = time.time()
    imag_recog()
    # vedio_recog()
    t1 = time.time()
    print('The time of processing is ', t1 - t0)
