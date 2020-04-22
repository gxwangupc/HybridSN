import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, BatchNormalization
from keras.layers import Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

from operator import truediv

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import time
import spectral
import argparse
import warnings
warnings.filterwarnings("ignore")

class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('-GPU', type=int, default=7, help='which gpu device to use')
        self.parser.add_argument('-DATASET', default='Indian', choices=['Salinas', 'Indian', 'PaviaU'],
                                 help='which data set for experiment')
        self.parser.add_argument('-CHANNEL', type=int, default=30, help='input channels')
        self.parser.add_argument('-WINDOW_SIZE', type=int, default=25, help='size of training/testing patches')
        self.parser.add_argument('-N_CLS', type=int, default=16, choices=[16, 16, 9],
                                 help='how many class in the training data')
        self.parser.add_argument('-EPOCH', type=int, default=100, help='how many epochs to train for')
        self.parser.add_argument('-BATCH_SIZE', type=int, default=256, help='training batch size')
        self.parser.add_argument('-BN_FLAG', type=bool, default=False, help='do batch normalization or not')
        self.parser.add_argument('-DR_RATE', type=float, default=.4, help='dropout rate')
        self.parser.add_argument('-LR', type=float, default=1e-3, help='learning rate')
        self.parser.add_argument('-DECAY', type=float, default=1e-06, help='')
        self.parser.add_argument('-use_PCA', type=bool, default=True,
                                 help='use PCA or not')
        self.parser.add_argument('-TEST_RATIO', type=float, default=.7, help='ratio of testing data in the full set')

        self.parser.add_argument('-DATA_DIR', default='./dataset/', help='directory to load data')
        self.parser.add_argument('-RESULT', default='./result/', help='directory to save results')

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()
        return self.opt

opt = Config().parse()
opt.CHANNEL = 30 if opt.DATASET == 'Indian' else 15
opt.N_CLS = 9 if opt.DATASET == 'PaviaU' else 16

if not os.path.exists(opt.RESULT):
    os.makedirs(opt.RESULT)

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.GPU)
print("using gpu {}".format(opt.GPU))

"""
Data preparation.
"""
def load(dataset):
    if dataset == 'Indian':
        data = loadmat(os.path.join(opt.DATA_DIR, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        label = loadmat(os.path.join(opt.DATA_DIR, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif dataset == 'Salinas':
        data = loadmat(os.path.join(opt.DATA_DIR, 'Salinas_corrected.mat'))['salinas_corrected']
        label = loadmat(os.path.join(opt.DATA_DIR, 'Salinas_gt.mat'))['salinas_gt']
    elif dataset == 'PaviaU':
        data = loadmat(os.path.join(opt.DATA_DIR, 'PaviaU.mat'))['paviaU']
        label = loadmat(os.path.join(opt.DATA_DIR, 'PaviaU_gt.mat'))['paviaU_gt']
    else:
        raise NotImplementedError

    return data, label

def apply_pca(data):
    data_pca = np.reshape(data, (-1, data.shape[-1]))
    pca = PCA(n_components=opt.CHANNEL, whiten=True)
    data_pca = pca.fit_transform(data_pca)
    data_pca = np.reshape(data_pca, (data.shape[0], data.shape[1], opt.CHANNEL))

    return data_pca, pca

def pad_zeros(input, margin=4):
    output = np.zeros((input.shape[0] + 2 * margin, input.shape[1] + 2 * margin, input.shape[2]))
    row_offset = margin
    col_offset = margin
    output[row_offset:input.shape[0] + row_offset, col_offset:input.shape[1] + col_offset, :] = input
    return output

def create_patches(data, label, rm_zero_labels = True):
    margin = int((opt.WINDOW_SIZE - 1) / 2)
    data_padded = pad_zeros(data, margin=margin)
    # split patches
    data_patches = np.zeros((data.shape[0] * data.shape[1], opt.WINDOW_SIZE, opt.WINDOW_SIZE, data.shape[-1]))
    label_patches = np.zeros((data.shape[0] * data.shape[1]))
    patch_index = 0
    for row in range(margin, data_padded.shape[0] - margin):
        for col in range(margin, data_padded.shape[1] - margin):
            patch = data_padded[row - margin:row + margin + 1, col - margin:col + margin + 1]
            data_patches[patch_index, :, :, :] = patch
            label_patches[patch_index] = label[row - margin, col - margin]
            patch_index = patch_index + 1
    # remove zero labels
    if rm_zero_labels:
        data_patches = data_patches[label_patches > 0, :, :, :]
        data_patches = data_patches.reshape(data_patches.shape[0], -1)
        label_patches = label_patches[label_patches > 0]
        label_patches -= 1

    return data_patches, label_patches

def split(data, label):
    data_tr, data_te, label_tr, label_te = train_test_split(data, label, test_size=opt.TEST_RATIO, random_state=345,
                                                        stratify=label)
    return data_tr, data_te, label_tr, label_te


data, label = load(opt.DATASET)
if opt.use_PCA:
    data, pca = apply_pca(data)
else:
    pass
data_patches, label_patches = create_patches(data, label)
data_tr, data_te, label_tr, label_te = split(data_patches, label_patches)

data_tr = data_tr.reshape(-1, opt.WINDOW_SIZE, opt.WINDOW_SIZE, opt.CHANNEL, 1)
label_tr = np_utils.to_categorical(label_tr)
data_te = data_te.reshape(-1, opt.WINDOW_SIZE, opt.WINDOW_SIZE, opt.CHANNEL, 1)
label_te = np_utils.to_categorical(label_te)



"""
Model,
"""
# Input
input = Input((opt.WINDOW_SIZE, opt.WINDOW_SIZE, opt.CHANNEL, 1))

# Convolutional layers
conv3d_1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(input)
conv3d_2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv3d_1)
conv3d_3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv3d_2)

reshape = Reshape((conv3d_3._keras_shape[1], conv3d_3._keras_shape[2], conv3d_3._keras_shape[3]*conv3d_3._keras_shape[4]))(conv3d_3)
conv2d_1 = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(reshape)

flatten = Flatten()(conv2d_1)

# Fully connected layers
dense_1 = Dense(units=256, activation='relu')(flatten)
dropout_1 = Dropout(opt.DR_RATE)(dense_1)
dense_2 = Dense(units=128, activation='relu')(dropout_1)
dropout_2 = Dropout(opt.DR_RATE)(dense_2)
output = Dense(units=opt.N_CLS, activation='softmax')(dropout_2)

# Defining the model with input layer and output layer
model = Model(inputs=input, outputs=output)
model.summary()

# Compiling the model
adam = Adam(lr=opt.LR, decay=opt.DECAY)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


"""
Training.
"""
saved_model = os.path.join(opt.RESULT, "best-model.hdf5") # checkpoint
checkpoint = ModelCheckpoint(saved_model, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = model.fit(x=data_tr, y=label_tr, batch_size=opt.BATCH_SIZE, epochs=opt.EPOCH, callbacks=callbacks_list)


"""
Testing.
"""
model.load_weights(saved_model) # load best weights
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

pred_te = model.predict(data_te)
pred_te = np.argmax(pred_te, axis=1)

classification = classification_report(np.argmax(label_te, axis=1), pred_te)
print(classification)

def aa_and_each_acc(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(data_te, label_te):
    start = time.time()
    pred_te = model.predict(data_te)
    pred_te = np.argmax(pred_te, axis=1)
    end = time.time()
    print(end - start)
    if opt.DATASET == 'Indian':
        targets = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif opt.DATASET == 'Salinas':
        targets = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
    elif opt.DATASET == 'PaviaU':
        targets = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
    else:
        raise NotImplementedError

    classification = classification_report(np.argmax(label_te, axis=1), pred_te, target_names=targets)
    oa = accuracy_score(np.argmax(label_te, axis=1), pred_te)
    confusion = confusion_matrix(np.argmax(label_te, axis=1), pred_te)
    each_acc, aa = aa_and_each_acc(confusion)
    kappa = cohen_kappa_score(np.argmax(label_te, axis=1), pred_te)
    score = model.evaluate(data_te, label_te, batch_size=32)
    Test_Loss = score[0] * 100
    Test_accuracy = score[1] * 100

    return classification, confusion, Test_Loss, Test_accuracy, oa * 100, each_acc * 100, aa * 100, kappa * 100

classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(data_te,label_te)
classification = str(classification)
confusion = str(confusion)
report = os.path.join(opt.RESULT, "classification_report.txt")

with open(report, 'w') as f:
    f.write('{} Test loss (%)'.format(Test_loss))
    f.write('\n')
    f.write('{} Test accuracy (%)'.format(Test_accuracy))
    f.write('\n')
    f.write('\n')
    f.write('{} Kappa accuracy (%)'.format(kappa))
    f.write('\n')
    f.write('{} Overall accuracy (%)'.format(oa))
    f.write('\n')
    f.write('{} Average accuracy (%)'.format(aa))
    f.write('\n')
    f.write('\n')
    f.write('{}'.format(classification))
    f.write('\n')
    f.write('{}'.format(confusion))


"""
Generate classification map.
"""
def patch(data, height_index, width_index):
    height_slice = slice(height_index, height_index + opt.WINDOW_SIZE)
    width_slice = slice(width_index, width_index + opt.WINDOW_SIZE)
    data_patch = data[height_slice, width_slice, :]

    return data_patch

# Load the original image
raw_data, raw_label = load(opt.DATASET)
if opt.use_PCA:
    raw_data, pca = apply_pca(raw_data)
else:
    pass
margin = int((opt.WINDOW_SIZE - 1) / 2)
raw_data_padded = pad_zeros(raw_data, margin=margin)

# Calculate the predicted image
pred_map = np.zeros((raw_label.shape[0], raw_label.shape[1]))
for row in range(raw_label.shape[0]):
    for col in range(raw_label.shape[1]):
        target = int(raw_label[row, col])
        if target == 0:
            continue
        else:
            img_patch = patch(raw_data_padded, row, col)
            data_te_img = img_patch.reshape(1,img_patch.shape[0],img_patch.shape[1], img_patch.shape[2], 1).astype('float32')
            prediction = model.predict(data_te_img)
            prediction = np.argmax(prediction, axis=1)
            pred_map[row][col] = prediction+1

spectral.save_rgb(os.path.join(opt.RESULT, str(opt.DATASET)+"_predictions.jpg"), pred_map.astype(int), colors=spectral.spy_colors)
spectral.save_rgb(os.path.join(opt.RESULT, str(opt.DATASET) + "_groundtruth.jpg"), raw_label, colors=spectral.spy_colors)
