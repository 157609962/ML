import os
import cv2
import numpy as np
from time import sleep
from alive_progress import alive_bar
from PIL import Image

all_imag = []
all_label = []
object = ['wet garbage', 'recyclable garbage', 'harmful garbage', 'other garbage']

def getimagandlabel(main_dir): # 输入图片路径，获取图片的标签
    listdir = os.listdir(main_dir)
    if listdir is not None:
        for path in listdir:
            if path[0] == '.':
                continue
            filelist = os.listdir(os.path.join(main_dir, path))
            for file in filelist:
                all_label.append(np.int32(path))  # 通过文件夹名称进行图片的分类放置
                path_inte = os.path.join(main_dir, path, file)  # 完整子文件路径
                imag = resize(path_inte)
                # imag = cv2.Canny(imag, 100, 100)
                imag = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
                # imag = imag[2, :]
                imag = cv2.resize(imag, (100, 100))
                all_imag.append(imag)

def get_hog():  # 创建hog特征（在后面调用该函数提取图片的hog特征并训练）
    winSize = (20, 20)
    blockSize = (8, 8)
    blockStride = (4, 4)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)
    return hog

def svmInit(C=12.5, gamma=0.50625):  # 创建svm模型（模型初始化）
    model = cv2.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(C)
    model.setKernel(cv2.ml.SVM_RBF)
    model.setType(cv2.ml.SVM_C_SVC)
    return model  # 输出建立的svm分类器模型

def svmTrain(model, samples, responses):  # 训练svm模型
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    model.save("svm_model.xml")  # 保存训练后的svm模型数据，用于垃圾分类图片检测
    print('The date of model have been saved...')

def svmPredict(model, samples):  # 输入垃圾图片，并用训练好的模型进行识别
    return model.predict(samples)[1].ravel()

def svmEvaluate(model, test_imag, test_label):  # 模型准确性的评估
    predictions = svmPredict(model, test_imag)
    accuracy = (test_label == predictions).mean()
    print('Percentage Accuracy: %.2f %%' % (accuracy * 100))  # 输出测试的准确率

def resize(path_imag):
    im = Image.open(path_imag)
    rec = 100  # 设置的图片大小
    w, h = im.size
    if w > h:
        a = w
        w = rec
        h = int(h * rec / a)
    else:
        a = h
        h = rec
        w = int(w * rec / a)
    # print(w, h)
    im = im.resize((w, h))
    # im.show()
    # blank1 = np.ones((100, 100))
    blank_white = Image.new('RGB', (rec, rec), (255, 255, 255))
    # blank_black = Image.new('RGB', (rec, rec), (0, 0, 0))
    box = (int((rec - w) / 2), int((rec - h) / 2), int(rec - (rec - w) / 2), int(rec - (rec - h) / 2))
    blank_white.paste(im, box=box)
    # blank_black.paste(im, box=box)
    # blank.show()
    im = np.array(im).astype(np.uint8)
    im = np.array(blank_white).astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return im

if __name__ == '__main__':
    main_dir = 'D:/Opencv-python/garbage_classification_hog_svm/garbage_images'
    getimagandlabel(main_dir)
    hog = get_hog()
    print('Calculating HoG descriptor for every image ... ')
    hog_descriptors = []
    with alive_bar(len(all_imag)) as bar:  # declare your expected total
        for imag in all_imag:
            hog_descriptors.append(hog.compute(imag))
            bar()  # call after consuming one item
            sleep(0.001)
    hog_descriptors = np.squeeze(hog_descriptors)
    print('Spliting data into training (90%) and test set (10%)')
    train_n=int(0.9*len(hog_descriptors))
    hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
    train_label, test_label = np.split(all_label, [train_n])
    print('Training SVM model ...')
    model = svmInit()  # 创建svm分类器
    svmTrain(model, hog_descriptors_train, train_label)  # 训练svm分类器
    print('Evaluating model ... ')
    svmEvaluate(model, hog_descriptors_test, test_label)
