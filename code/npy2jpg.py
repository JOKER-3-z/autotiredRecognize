import cv2
import numpy as np
import pandas as pd
import os
from shutil import move
import random

origin_pth = os.path.split(os.path.realpath(__file__))[0]
os.chdir(origin_pth)
os.rename('../xfdata/自动驾驶疲劳检测挑战赛公开数据-更新', '../xfdata/sleepy_npy')

#----------------------
# generate jpg folder
#----------------------

def mkdir(path):
    floder = os.path.exists(path)
    if not floder:
        os.makedirs(path)
    else:
        pass

folder = [
    '../xfdata/sleepy',
    '../xfdata/sleepy/train',
    '../xfdata/sleepy/valid',
    '../xfdata/sleepy/test',
    '../xfdata/sleepy/pred',
    '../xfdata/sleepy/train/sleepy',
    '../xfdata/sleepy/train/non-sleepy',
    '../xfdata/sleepy/valid/sleepy',
    '../xfdata/sleepy/valid/non-sleepy'
]
for i in range(len(folder)):
    mkdir(folder[i])

train_path = np.load('../xfdata/sleepy_npy/train.npy')
test_path = np.load('../xfdata/sleepy_npy/test.npy')
train_label = pd.read_csv('../xfdata/sleepy_npy/train_label.csv', header=None)
train_label = train_label[0]

#-----------------
# generate jpg
#-----------------
def imgenhance(imgdata):
    img = imgdata.astype('uint8')
    return img

for i in range(len(train_label)):
    if train_label[i] == 'sleepy':
        os.chdir('../xfdata/sleepy/train/sleepy')
        cv2.imwrite('{}.jpg'.format(i), imgenhance(train_path[i,]))
    else:
        os.chdir('../xfdata/sleepy/train/non-sleepy')
        cv2.imwrite('{}.jpg'.format(i), imgenhance(train_path[i,]))
    os.chdir(origin_pth)

os.chdir('../xfdata/sleepy/test')
for i in range(test_path.shape[0]):
    cv2.imwrite('{}.jpg'.format(i), imgenhance(test_path[i,]))
os.chdir(origin_pth)
#----------
# count
#----------

classes = ['sleepy', 'non-sleepy']
sample_number = []
for i in range(2):
    train_dir = '../xfdata/sleepy/train/{}'.format(classes[i])
    file = os.listdir(train_dir)
    sample_number.append(len(file))
print('jpg number:', sample_number)

#----------------------
# sample valid set
#----------------------

def moveFile(train_Dir, val_Dir, rate=0.1):
    pathDir = os.listdir(train_Dir)
    filenumber = len(pathDir)
    sample = random.sample(pathDir, int(filenumber * rate))
    for name in sample:
       move(os.path.join(train_Dir, name) ,os.path.join(val_Dir, name) )
    return

for i in range(2):
    train_dir = '../xfdata/sleepy/train/{}'.format(classes[i])
    val_dir = '../xfdata/sleepy/valid/{}'.format(classes[i])
    moveFile(train_dir, val_dir)
sample_number = []
for i in range(2):
    train_dir = '../xfdata/sleepy/train/{}'.format(classes[i])
    file = os.listdir(train_dir)
    sample_number.append(len(file))
print('original training set:', sample_number)
sample_number = []
for i in range(2):
    train_dir = '../xfdata/sleepy/valid/{}'.format(classes[i])
    file = os.listdir(train_dir)
    sample_number.append(len(file))
print('original val set:', sample_number)

#----------------------------------------------
# random generate balanced sample (trainning)
#----------------------------------------------

for i in range(2):
    train_dir = '../xfdata/sleepy/train/{}'.format(classes[i])
    pathDir = os.listdir(train_dir)
    filenumber = len(pathDir)
    os.chdir(train_dir)
    if filenumber != 5500:
        for j in range(5500-filenumber):
            index = np.random.randint(filenumber)
            image = cv2.imread(pathDir[index])
            filename = 'synthetic{}.jpg'.format(j)
            cv2.imwrite(filename, image)      
    os.chdir(origin_pth)

sample_number = []
for i in range(2):
    train_dir = '../xfdata/sleepy/train/{}'.format(classes[i])
    file = os.listdir(train_dir)
    sample_number.append(len(file))
print('balanced training set:', sample_number)

#-----------------------------------------------
# random generate balanced sample (valid)
#-----------------------------------------------

for i in range(2):
    train_dir = '../xfdata/sleepy/valid/{}'.format(classes[i])
    pathDir = os.listdir(train_dir)
    filenumber = len(pathDir)
    os.chdir(train_dir)
    if filenumber != 550:
        for j in range(550-filenumber):
            index = np.random.randint(filenumber)
            image = cv2.imread(pathDir[index])
            filename = 'synthetic{}.jpg'.format(j)
            cv2.imwrite(filename, image)      
    os.chdir(origin_pth)

sample_number = []
for i in range(2):
    train_dir = '../xfdata/sleepy/train/{}'.format(classes[i])
    file = os.listdir(train_dir)
    sample_number.append(len(file))
print('balanced val set:', sample_number)

