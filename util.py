import itertools
import argparse
import numpy as np
import pandas as pd
import sklearn
import random
import sklearn.metrics
import time
import os
import sys
import json
import gc
import json
import pickle
import hashlib
import tempfile
import keras.optimizers
import cv2
import math
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
import ast
import h5py
from keras.utils import np_utils
from itertools import tee
import keras.backend as K
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import PIL.Image

JPG_ROOT_PATH = '/host/4TB_hybridvs_data/YOLO-RES-720P/jpg'
CSV_ROOT_PATH = '/host/4TB_hybridvs_data/YOLO-RES-720P/out'
RES_ROOT_PATH = '/host/4TB_hybridvs_data/YOLO-RES-720P/exp'
FRAMES_PER_SEC = 10

computed_metrics = ['accuracy', 'mean_squared_error']

def read_coco_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

class ClockLog():
    def __init__(self, secs):
        self.interval = secs
        self.last_log = 0
    def log(self, msg):
        now = time.time()
        if now - self.last_log > self.interval:
            print (msg)
            self.last_log = now

# In case we want more callbacks
def get_callbacks(model_fname, patience=2):
    return [ModelCheckpoint(model_fname)]
    return [EarlyStopping(monitor='loss',     patience=patience, min_delta=0.00001),
            EarlyStopping(monitor='val_loss', patience=patience + 2, min_delta=0.0001),
            ModelCheckpoint(model_fname, save_best_only=True)]

def get_loss():
    return 'categorical_crossentropy'

def get_optimizer(lr_mult):
#     return keras.optimizers.RMSprop(lr=lr_mult)# / (5 * nb_layers))
    return keras.optimizers.Adam(lr=lr_mult)# / (5 * nb_layers))


def generate_conv_net_base_old(
        input_shape, nb_classes,
        nb_dense, nb_filters, nb_layers, lr_mult=0.001,
        kernel_size=(3, 3), stride=(1, 1)):
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape,
                            subsample=stride,
                            activation='relu'))
#     model.add(Convolution2D(nb_filters, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    nb_filter_multiplier = 2
    
    for i in range(nb_layers - 1):
        model.add(Convolution2D(nb_filters * nb_filter_multiplier,
                                3, 3, border_mode='same', activation='relu'))
#         model.add(Convolution2D(nb_filters * nb_filter_multiplier,
#                                 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        nb_filter_multiplier += 1

    model.add(Flatten())
    model.add(Dense(nb_dense, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    loss = get_loss()
    model.compile(loss=loss,
                  optimizer=get_optimizer(lr_mult),
                  metrics=computed_metrics)
    return model

def generate_conv_net_base(
        input_shape, nb_classes,
        nb_dense, nb_filters, nb_layers, lr_mult=0.001,
        kernel_size=(3, 3), stride=(1, 1)):
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape,
                            subsample=stride))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    nb_filter_multiplier = 2
    
    for i in range(nb_layers - 1):
        model.add(Convolution2D(nb_filters * nb_filter_multiplier,
                                3, 3, border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.25))
        nb_filter_multiplier += 1

    model.add(Flatten())
    model.add(Dense(nb_dense, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    loss = get_loss()
    model.compile(loss=loss,
                  optimizer=get_optimizer(lr_mult),
                  metrics=computed_metrics)
    return model

def generate_conv_net_base_regression(
        input_shape, nb_dense, nb_filters, nb_layers, lr_mult=0.001,
        kernel_size=(3, 3), stride=(1, 1)):
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='same',
                            input_shape=input_shape,
                            subsample=stride))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    nb_filter_multiplier = 2
    
    for i in range(nb_layers - 1):
        model.add(Convolution2D(nb_filters * nb_filter_multiplier,
                                3, 3, border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.25))
        nb_filter_multiplier += 1

    model.add(Flatten())
    model.add(Dense(nb_dense, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    loss = 'mean_squared_error'
    model.compile(loss=loss,
                  optimizer=get_optimizer(lr_mult),
                  metrics=computed_metrics)
    return model

def train_model(model, X_train, Y_train, batch_size=32, nb_epoch=10, patience=2, save_path=None):
    print ('training samples: %d/%d' % (
            np.count_nonzero(Y_train.argmax(axis=-1)), X_train.shape[0]))
    if save_path is None:
        temp_fname = tempfile.mkstemp(suffix='.hdf5', dir='/tmp/')[1]
    else:
        temp_fname = save_path + '.hdf5'
    temp_fname = str(temp_fname)
#     validation_split = 0.33333333
#     if len(Y_train) * validation_split > 50000.0:
#         validation_split = 50000.0 / float(len(Y_train))
#         print validation_split

#     begin_train = time.time()
    model.fit(X_train, Y_train,
                batch_size=batch_size,
                nb_epoch=nb_epoch,
                # validation_split=validation_split,
                # validation_data=(X_test, Y_test),
                shuffle=True,
                class_weight='auto',
                callbacks=get_callbacks(temp_fname, patience=patience))
#     train_time = time.time() - begin_train

    model.load_weights(temp_fname)

def test_model(model, X_test, Y_test, batch_size=128):
    probs = model.predict(X_test, batch_size=128, verbose=0)
    pos_samples = np.count_nonzero(Y_test.argmax(axis=-1))
    total_test_num = probs.shape[0]
    print ('test num: %d, pos num: %d' % (total_test_num, pos_samples))
    combined_res = np.column_stack((probs[:,1], Y_test[:,1]))
#     combined_res = combined_res[combined_res[:,0].argsort()]
#     pos_idxs = np.where(combined_res[:,1] == 1)[0].tolist()
#     accuracy = []
#     FN = 0
#     for i in range(total_test_num):
#         FN += combined_res[i,1]
#         TP = pos_samples - FN
#         FP = total_test_num - i - 1 - TP
#         TN = i + 1 - FN
#         FN_error = float(FN) / (FN + TP)
#         FP_error = float(FP) / (TP + FP + FN)
#         recall = float(TP) / (TP + FN)
#         precision = float(TP) / (TP + FP)
#         accuracy.append((TP, FP, TN, FN))
    return combined_res

def noscope_way(cfg):
    no_time_range = cfg['generic'] # models not tuned for different time ranges
    random_pos_ratio = cfg['input_cfg'][3] == -1 # pos ratio not controlled to 0.2
    no_crop = cfg['input_cfg'][4] == '-1,-1,-1,-1'
    fixed_input_size = cfg['resol'][0] == 50 and cfg['resol'][1] == 50
    return no_time_range and random_pos_ratio and no_crop and fixed_input_size

# def cal_accuracy(total_num, pos_idxs, error_list=[0.01, 0.05, 0.10, 0.2, 0.3, 0.5]):
#     ret = {}
#     for error in error_list:
#         filter_low = pos_idxs[int(len(pos_idxs) * error) + 1]
#         filter_high = total_num - pos_idxs[int(len(pos_idxs) * (1 - error))]
#         ret[error] = [float(filter_low) / total_num, float(filter_high + filter_low) / total_num]
#     return ret

# def cal_accuracy_many(pos_idxs_lst, error_list=[0.01, 0.05, 0.10, 0.2, 0.3, 0.5]):
#     ret = {e : [0, 0] for e in error_list}
#     for (total_num, pos_idxs) in pos_idxs_lst:
#         for error in error_list:
#             ret[error][0] += pos_idxs[min(int(len(pos_idxs) * error) + 1, len(pos_idxs) - 1)]
#             ret[error][1] += total_num - pos_idxs[int(len(pos_idxs) * (1 - error))]
#     total_num = sum(x[0] for x in pos_idxs_lst)
#     return {e: [float(ret[e][0]) / total_num, float(ret[e][0] + ret[e][1]) / total_num] for e in error_list}

# def cal_score_filter(scores, error_list=[0.01, 0.05, 0.1, 0.2]):
#     sorted_scores = scores[scores[:,0].argsort()]
#     pos_idxs = np.where(sorted_scores[:,1] == 1)[0].tolist()
#     total_num = scores.shape[0]
#     ret = {}
#     for error in error_list:
#         filter_low = pos_idxs[int(len(pos_idxs) * error) + 1]
#         filter_high = total_num - pos_idxs[int(len(pos_idxs) * (1 - error))]
#         ret[error] = [float(filter_low) / total_num, float(filter_high + filter_low) / total_num]
#     return ret

# def cal_score_filter_many(scores_lst, error_list=[0.01, 0.05, 0.1, 0.2]):
#     ret = {e : [0, 0] for e in error_list}
#     total_num = 0
#     for scores in scores_lst:
#         sorted_scores = scores[scores[:,0].argsort()]
#         pos_idxs = np.where(sorted_scores[:,1] == 1)[0].tolist()
#         if len(pos_idxs) == 0: continue
#         num = scores.shape[0]
#         total_num += num
#         for error in error_list:
#             filter_l = int(len(pos_idxs) * error)
#             filter_l = max(0, min(filter_l, len(pos_idxs) - 1))
#             ret[error][0] += pos_idxs[filter_l]
#             filter_h = num - pos_idxs[int(len(pos_idxs) * (1 - error))]
#             filter_h = max(0, min(filter_h, len(pos_idxs) - 1))
#             ret[error][1] += pos_idxs[filter_h]
#     return {e: [float(ret[e][0]) / total_num, float(ret[e][0] + ret[e][1]) / total_num] for e in error_list}

def cal_scores(scores, thre=0.5):
    precision, recall, threshold = sklearn.metrics.precision_recall_curve(
        scores[:,1], scores[:,0])
    auc = sklearn.metrics.roc_auc_score(scores[:,1], scores[:,0])
    true_labels = scores[:,1]
    predicted_labels = scores[:,0] > thre
    confusion = sklearn.metrics.confusion_matrix(true_labels, predicted_labels)
    # Minor smoothing to prevent division by 0 errors
    TN = float(confusion[0][0]) + 1
    FN = float(confusion[1][0]) + 1
    TP = float(confusion[1][1]) + 1
    FP = float(confusion[0][1]) + 1
    metrics = {'recall': TP / (TP + FN),
               'specificity': TN / (FP + TN),
               'precision': TP / (TP + FP),
               'npv':  TN / (TN + FN),
               'fpr': FP / (FP + TN),
               'fdr': FP / (FP + TP),
               'fnr': FN / (FN + TP),
               'accuracy': (TP + TN) / (TP + FP + TN + FN),
               'f1': (2 * TP) / (2 * TP + FP + FN),
               'auc': auc,
               'precision_recall': [precision, recall, threshold]}
    return metrics

def cal_score_filter(scores, error_list=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5]):
    precision, recall, threshold = sklearn.metrics.precision_recall_curve(
        scores[:,1], scores[:,0])
    pred_scores = scores[:,0].tolist()
    ff = {}
    temp = {threshold[i]: (precision[i], recall[i]) for i in range(len(threshold))}
    for e in error_list:
        pos_thre = min([x for x in temp.keys() if temp[x][0] > (1 - e)] + [1])
        neg_thre = max([x for x in temp.keys() if temp[x][1] > (1 - e)] + [0])
        filter_pos = float(len([x for x in pred_scores if x > pos_thre])) / scores.shape[0]
        filter_neg = float(len([x for x in pred_scores if x < neg_thre])) / scores.shape[0]
        ff[e] = (filter_pos, filter_neg)
    return ff

def cal_score_filter_many(scores_lst, error_list=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5]):
    ff_lst = [cal_score_filter(s, error_list=error_list) for s in scores_lst]
    ret = {}
    for e in error_list:
        filter_neg_lst = [ff[e][1] for ff in ff_lst]
        filter_pos_lst = [ff[e][0] for ff in ff_lst]
        ret[e] = (np.mean(filter_pos_lst), np.mean(filter_neg_lst))
    return ret

def get_bin_acc(cfg_str_lst, gt_crop=False, manual_crop=None):
    def update_scores(scores, testids, obj_ids):
        for i in range(scores.shape[0]):
            if testids[i] in obj_ids:
                scores[i, 1] = 1
            else:
                scores[i, 1] = 0
    tmp_cfg = json.loads(cfg_str_lst[0])
    video = tmp_cfg['video']
    video_clip_num = tmp_cfg['input_cfg'][2]
    obj = tmp_cfg['OBJECT']
    bin_dir = os.path.join(RES_ROOT_PATH, tmp_cfg['video'], 'bin_acc')
    if not os.path.exists(bin_dir):
        os.mkdir(bin_dir)
    save_fname = os.path.join(bin_dir, video)
    if gt_crop: save_fname += '_crop'
    if manual_crop is not None: save_fname += ('_' + manual_crop)
    if os.path.exists(save_fname):
        with open(save_fname, 'rb') as f:
            stored_acc = pickle.load(f)
    else:
        print ('first time run...')
        stored_acc = {}
    temp = [cfg_str for cfg_str in cfg_str_lst if cfg_str in stored_acc]
    if len(temp) == len(cfg_str_lst):
        print ('no need to update accuracy')
        return {cfg_str: stored_acc[cfg_str] for cfg_str in cfg_str_lst}
    print ('updating cfg accuracy:', len(cfg_str_lst) - len(temp))
    sub_videos = [video + '-' + str(p + 1) + '_10FPS' for p in range(video_clip_num)]
    img_sub_dirs = [os.path.join(JPG_ROOT_PATH, v) for v in sub_videos]
    img_sub_nums = [len(os.listdir(d)) for d in img_sub_dirs]
    csv_sub_files = [os.path.join(CSV_ROOT_PATH, d + '.csv') for d in sub_videos]
    total_frame_num = sum(img_sub_nums)
    SMOOTH_WINDOW = tmp_cfg['smooth_window'] # for smooth the yolo output
    testid_dir = os.path.join(RES_ROOT_PATH, tmp_cfg['video'], 'testids')
    testids_wnd = []
    for i in range(video_clip_num):
        id_fname = os.path.join(testid_dir, video + '-' + obj + '-' + str(i))
        with open(id_fname, 'rb') as f:
            ids = pickle.load(f)
            testids_wnd.append(ids)
    testids_concat = np.concatenate(testids_wnd, axis=0)
    gt_obj_ids = {}
    for cfg_str in cfg_str_lst:
        if cfg_str in stored_acc:
            continue
        cfg = json.loads(cfg_str)
        if gt_crop:
            crop = cfg['input_cfg'][-1]
        elif manual_crop is not None:
            crop = manual_crop
        else:
            crop = '-1,-1,-1,-1'
        if crop not in gt_obj_ids:
            obj_ids, _ = get_csv_samples_many(
                csv_sub_files, img_sub_nums, obj, total_frame_num,
                WINDOW=SMOOTH_WINDOW, crop=str2ilst(crop))
            gt_obj_ids[crop] = obj_ids
        obj_ids = gt_obj_ids[crop]
        scores = read_scores(cfg_str)
        cur_acc = {}
        update_scores(scores, testids_concat, obj_ids)
        cur_acc = cal_scores(scores)
        cur_acc['filter-combine'] = cal_score_filter(scores)
        if cfg['generic']:
            cur_acc['filter'] = cur_acc['filter-combine']
        else:
            wnd_sz = scores.shape[0] / video_clip_num
            scores_lst = [scores[i*wnd_sz:(i+1)*wnd_sz,:] for i in range(video_clip_num)]
            cur_acc['filter'] = cal_score_filter_many(scores_lst)
        
#         else:
#             scores_lst = []
#             for (wnd_id, score) in enumerate(scores):
#                 update_scores(score, testids_wnd[wnd_id], obj_ids)
#                 scores_lst.append(score)
#             cur_acc['filter'] = cal_score_filter_many(scores_lst)
#             cat_scores = np.concatenate(scores_lst, axis=0)
#             cur_acc['accuracy'] = cal_scores(cat_scores)
        stored_acc[cfg_str] = cur_acc
    with open(save_fname, 'wb') as f:
        pickle.dump(stored_acc, f)
    return {cfg_str: stored_acc[cfg_str] for cfg_str in cfg_str_lst}

def read_scores(cfg_str):
    cfg = json.loads(cfg_str)
    cfg_md5 = hashlib.md5(cfg_str).hexdigest()
    cfg_fname = os.path.join(RES_ROOT_PATH, cfg['video'], 'scores', cfg_md5)
    return np.load(cfg_fname)

def cfg2mean(cfg):
    tag = ','.join([str(cfg['resol'][0]), str(cfg['resol'][1]), cfg['input_cfg'][-1]])
    return tag

def save_mean(cfg, mean):
    cfg_tag = cfg2mean(cfg)
    mean_dir = os.path.join(RES_ROOT_PATH, cfg['video'], 'mean')
    if not os.path.exists(mean_dir):
        os.mkdir(mean_dir)
    mean_fname = os.path.join(mean_dir, cfg_tag)
    np.save(mean_fname, mean)

def read_mean(cfg):
    cfg_tag = cfg2mean(cfg)
    mean_fname = os.path.join(RES_ROOT_PATH, cfg['video'], 'mean', cfg_tag) + '.npy'
    return np.load(mean_fname)
    
# def read_scores(cfg_str, concat=True):
#     cfg = json.loads(cfg_str)
#     cfg_md5 = hashlib.md5(cfg_str).hexdigest()
#     cfg_fname = os.path.join(RES_ROOT_PATH, 'scores', cfg_md5)
#     if cfg['generic']:
#         ret = np.load(cfg_fname)
#     else:
#         wnd_num = cfg['input_cfg'][2]
#         ret = []
#         for i in range(wnd_num):
#             ret.append(np.load(cfg_fname + '_' + str(i)))
#         if concat:
#             ret = np.concatenate(ret, axis=0)
#     return ret

def filter_crop(df, crop):
    df = df[(df['xmin']+df['xmax'])/2 >= crop[1]]
    df = df[(df['xmin']+df['xmax'])/2 <= crop[3]]
    df = df[(df['ymin']+df['ymax'])/2 >= crop[0]]
    df = df[(df['ymin']+df['ymax'])/2 <= crop[2]]
    return df

def rep_obj(OBJ):
    if OBJ == 'deer':
        return ['carnivore', 'ungulate', 'placental', 'bovid', 'even-toed ungulate', 'ruminant',
               'canine', 'mammal', 'sheep', 'bovine', 'antelope', 'cattle', 'deer']
    elif OBJ == 'eagle':
        return []
    else:
        return [OBJ]

def get_csv_samples(csv_fname, OBJ, limit, begin=0, WINDOW=3, thre=40, crop=None):
    df = pd.read_csv(csv_fname)
    df = df[df['object_name'].isin(rep_obj(OBJ))]
    df = df[df['confidence']>thre]
    if not crop is None and crop[0] > 0:
        df = filter_crop(df, crop)
    groups = df.set_index('frame')
    counts = map(lambda i: i in groups.index, range(begin, limit))
    counts = np.array(counts)
    idx = np.argwhere(counts > 0)[:,0]
    
    smoothed_counts = np.convolve(np.ones(WINDOW), np.ravel(counts), mode='same') > WINDOW * 0.6
    print ("pos samples before/after smooth: %d/%d" % (np.sum(counts), np.sum(smoothed_counts)))
    smoothed_counts = smoothed_counts.reshape(len(counts), 1)
    counts = smoothed_counts
    idx = np.argwhere(counts > 0)[:,0]
    return idx, df[df['frame'].isin(idx)]

def get_csv_samples_many(csv_fnames, img_nums, OBJ, limit, begin=0, WINDOW=3, thre=40, crop=None):
    assert len(csv_fnames) == len(img_nums), 'csv and img sub shall be same!'
    print ('reading csv files from ' + csv_fnames[0] + '...')
    dfs = []
    for idx, csv_fname in enumerate(csv_fnames):
        df = pd.read_csv(csv_fname)
        df = df[df['object_name'].isin(rep_obj(OBJ))]
        df = df[df['confidence']>thre]
        if not crop is None and crop[0] > 0:
            df = filter_crop(df, crop)
        df['frame'] = df['frame'] + sum(img_nums[:idx])
        dfs.append(df)
    df = pd.concat(dfs, axis = 0, ignore_index = True)
    groups = df.set_index('frame')
    counts = map(lambda i: i in groups.index, range(begin, limit))
    counts = np.array(counts)
    
    smoothed_counts = np.convolve(np.ones(WINDOW), np.ravel(counts), mode='same') > WINDOW * 0.6
    print ("pos samples before/after smooth: %d/%d" % (np.sum(counts), np.sum(smoothed_counts)))
    smoothed_counts = smoothed_counts.reshape(len(counts), 1)
    idx = np.argwhere(smoothed_counts > 0)[:,0]
    return idx, df[df['frame'].isin(idx)]

def get_csv_samples_counting(csv_fnames, img_nums, OBJ, limit, begin=0, thre=40, crop=(-1, -1, -1, -1)):
    assert len(csv_fnames) == len(img_nums), 'csv and img sub shall be same!'
    dfs = []
    for idx, csv_fname in enumerate(csv_fnames):
        df = pd.read_csv(csv_fname)
        df = df[df['object_name'].isin([OBJ])]
        df = df[df['confidence']>thre]
        df['frame'] = df['frame'] + sum(img_nums[:idx])
        dfs.append(df)
    df = pd.concat(dfs, axis = 0, ignore_index = True)
    groups = df.set_index('frame')
    counts = map(lambda i: 0, range(begin, limit))
    counts = [0] * (limit - begin)
    for idx in list(groups.index):
        if idx < limit:
            counts[idx - begin] += 1
#     counts = [x if (x < nb_classes - 1) else nb_classes - 1 for x in counts]
    return counts

def get_model_fname(cur_cfg):
    models_dir = os.path.join(RES_ROOT_PATH, cur_cfg['video'], 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    cfg_str = json.dumps(cur_cfg)
    cfg_md5 = hashlib.md5(cfg_str).hexdigest()
    model_fname = os.path.join(models_dir, cur_cfg['video'] + '-' + cfg_md5)
    return model_fname

def get_labels(obj_ids, frameset):
    ret = [t in obj_ids for t in frameset]
    print ('reading label sum: %d, pos: %d' % (len(ret), sum(ret)))
    return np_utils.to_categorical(ret, 2)

def get_big_box(matches):
    min_x = min(m[0] for m in matches)
    min_y = min(m[1] for m in matches)
    max_x = max(m[2] for m in matches)
    max_y = max(m[3] for m in matches)
    return (min_x, min_y, max_x, max_y)

def contain_rect(br, sr):
    return br[0] <= sr[0] and br[1] <= sr[1] and br[2] >= sr[2] and br[3] >= sr[3]

def convert(crop):
    return (crop[1], crop[0], crop[1] + crop[3], crop[0] + crop[2])

def get_frames_from_images(frameset, img_dirs, img_nums, fgmask_dict=None,
                           resol=(-1, -1), crop=(-1, -1, -1, -1), dtype='float32'):
    def get_image_path(idx):
        for i in range(len(img_dirs)):
            if sum(img_nums[:i + 1]) > idx:
                return os.path.join(img_dirs[i], str(idx - sum(img_nums[:i]) + 1).zfill(7) + '.jpg')
    assert len(img_dirs) == len(img_nums), 'image subdir numbers not consistent!'
#     print ('reading images... %d from %s...') % (len(frameset), img_dirs[0])
    if resol[0] > 0:
        frames = np.zeros(tuple([len(frameset)] + list(resol) + [3]), dtype=dtype )
    else:
        test_frame = cv2.imread(os.path.join(img_dirs[0], '0000001.jpg'))
        frames = np.zeros(tuple([len(frameset)] + list(test_frame.shape)), dtype=dtype )
    for i in range(len(frameset)):
        img_path = get_image_path(frameset[i])
        frame = cv2.imread(img_path)
        if fgmask_dict is not None:
            boxes = fgmask_dict[frameset[i]]
            boxes = [convert(b) for b in boxes]
            if crop[0] > 0:
                boxes = [b for b in boxes if contain_rect(crop, b)]
            if len(boxes) > 0:
                crop = get_big_box(boxes)
        if crop[0] > 0:
            frame = frame[crop[0]:crop[2],crop[1]:crop[3],:]
        if resol[0] > 0:
            frame = cv2.resize(frame, (resol[1], resol[0]), interpolation=cv2.INTER_NEAREST)
        frames[i, :] = frame

    if dtype == 'float32':
        frames /= 255.0

    return frames

def get_frames_single(frameset, img_dir, resol=(-1, -1), crop=(-1, -1, -1, -1), dtype='float32'):
    def get_image_path(idx):
        return os.path.join(img_dir, str(idx + 1).zfill(7) + '.jpg')
    print ('reading images... %d') % (len(frameset))
    if resol[0] > 0:
        frames = np.zeros(tuple([len(frameset)] + list(resol) + [3]), dtype=dtype )
    else:
        test_frame = cv2.imread(os.path.join(img_dir, '0000001.jpg'))
        frames = np.zeros(tuple([len(frameset)] + list(test_frame.shape)), dtype=dtype )
    for i in range(len(frameset)):
        img_path = get_image_path(frameset[i])
        frame = cv2.imread(img_path)
        if crop[0] > 0:
            frame = frame[crop[0]:crop[2],crop[1]:crop[3],:]
        if resol[0] > 0:
            frame = cv2.resize(frame, (resol[1], resol[0]), interpolation=cv2.INTER_NEAREST)
        frames[i, :] = frame

    if dtype == 'float32':
        frames /= 255.0

    return frames
def resize_images(frames, resol, dtype='float32'):
    new_frames = np.zeros(tuple([frames.shape[0]] + list(resol) + [3]), dtype=dtype)
    for i in range(frames.shape[0]):
        one_frame = cv2.resize(frames[i,:,:,:], (resol[1], resol[0]), interpolation=cv2.INTER_NEAREST)
        new_frames[i, :] = one_frame
    return new_frames

def str2ilst(s):
    if s is not None:
        return [int(x) for x in s.split(',')]
    else:
        return None

cam_spd = {}
def get_NN_speed(hw, param):
    global cam_spd
    if hw not in cam_spd:
        cam_spd[hw] = {}
        fname = '/host/hybridvs_data/models/odroid_speed.txt'
        if hw == 'rpi':
            fname = '/host/hybridvs_data/models/rpi_speed.txt'
        for line in open(fname).readlines():
            cam_spd[hw][line.strip().split()[0]] = float(line.strip().split()[1])
    if param.startswith('50,25'):
        return cam_spd[hw][param.replace('50,25', '50,50')] * 2
    else:
        return cam_spd[hw][param]
    
def get_all_scores(cfg_str, frameset):
    cfg = json.loads(cfg_str)
    video = cfg['video']
    video_clip_num = cfg['input_cfg'][2]
    crop = str2ilst(cfg['input_cfg'][-1])
    resol = cfg['resol']
    out_dir = os.path.join(RES_ROOT_PATH, video, 'full_scores')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    score_fname = os.path.join(out_dir, hashlib.md5(cfg_str).hexdigest())
#     print score_fname
    if os.path.exists(score_fname):
        with open(score_fname, 'rb') as f:
            score_dict = pickle.load(f)
    else:
        score_dict = {}
    if frameset is None:
        return score_dict
    run_frameset = set(frameset) - set(score_dict.keys())
    if len(run_frameset) == 0:
        return {idx: score_dict[idx] for idx in frameset}
    sub_videos = [video + '-' + str(p + 1) + '_10FPS' for p in range(video_clip_num)]
    img_sub_dirs = [os.path.join(JPG_ROOT_PATH, v) for v in sub_videos]
    img_sub_nums = [len(os.listdir(d)) for d in img_sub_dirs]
    csv_sub_files = [os.path.join(CSV_ROOT_PATH, d + '.csv') for d in sub_videos]
    total_frame_num = sum(img_sub_nums)
    mean = read_mean(cfg)
    print ('run cfg on new frames:', len(run_frameset))
    if cfg['generic']:
        model_fname = get_model_fname(cfg)
        tag = model_fname.split('/')[-1]
        model_fname += '.hdf5'
        model = keras.models.load_model(str(model_fname))
    for i in range(video_clip_num):
        if not cfg['generic']:
            K.clear_session()
            model_fname = get_model_fname(cfg)
            tag = model_fname.split('/')[-1]
            model_fname += ('_' + str(i) + '.hdf5')
            model = keras.models.load_model(str(model_fname))
        cur_frameset = [x for x in run_frameset if
                        x >= sum(img_sub_nums[:i]) and x < sum(img_sub_nums[:i+1])]
        if len(cur_frameset) == 0: continue
        cur_frames = get_frames_from_images(
            cur_frameset, img_sub_dirs, img_sub_nums, resol=resol, crop=crop)
        cur_frames -= mean
        probs = model.predict(cur_frames, batch_size=128, verbose=0)
        for i in range(len(cur_frameset)):
            score_dict[cur_frameset[i]] = probs[i,1]
    
    with open(score_fname, 'wb') as f:
        pickle.dump(score_dict, f)
    
    return {idx: score_dict[idx] for idx in frameset}
    

# def has_full_scores(cfg_str):
#     cfg = json.loads(cfg_str)
#     video = cfg['video']
#     tag = cfg_md5 = hashlib.md5(cfg_str).hexdigest()
#     out_fname = os.path.join(RES_ROOT_PATH, video, 'full_scores', tag)
#     return os.path.exists(out_fname)

# def read_full_scores(cfg_str, concat=True):
#     cfg = json.loads(cfg_str)
#     cfg_md5 = hashlib.md5(cfg_str).hexdigest()
#     wnd_num = cfg['input_cfg'][2]
#     ret = []
#     for i in range(wnd_num):
#         score_fname = os.path.join(
#             RES_ROOT_PATH, cfg['video'], 'full_scores', str(i + 1) + '-' + cfg_md5)
#         ret.append(np.load(score_fname))
#     if concat:
#         ret = np.concatenate(ret, axis=0)
#     return ret

# def run_all_frames(cfg_str):
#     cfg = json.loads(cfg_str)
#     video = cfg['video']
#     video_clip_num = cfg['input_cfg'][2]
#     out_dir = os.path.join(RES_ROOT_PATH, 'full_scores')
#     if not os.path.exists(out_dir):
#         os.mkdir(out_dir)
#     print 'run all frames: ', cfg_str
#     if cfg['generic']:
#         model_fname = get_model_fname(cfg)
#         tag = model_fname.split('/')[-1]
#         model_fname += '.hdf5'
#         model = keras.models.load_model(model_fname)
#     crop = str2ilst(cfg['input_cfg'][-1])
#     resol = cfg['resol']
#     for i in range(video_clip_num):
#         if not cfg['generic']:
#             K.clear_session()
#             model_fname = get_model_fname(cfg)
#             tag = model_fname.split('/')[-1]
#             model_fname += ('_' + str(i) + '.hdf5')
#             model = keras.models.load_model(model_fname)
#         out_fname = os.path.join(out_dir, video + '-' + str(i + 1) + '-' + tag)
#         if os.path.exists(out_fname): continue
#         img_dir = os.path.join(JPG_ROOT_PATH, video + '-' + str(i + 1))
#         img_num = len(os.listdir(img_dir))
#         frameset = range(img_num)
#         csv_fname = os.path.join(CSV_ROOT_PATH, video + '-' + str(i + 1) + '_608.csv')
#         X_test = get_frames_single(frameset, img_dir, resol=resol, crop=crop)
#         obj_ids, _ = get_csv_samples(
#             csv_fname, cfg['OBJECT'], X_test.shape[0], WINDOW=3, thre=40)
#         Y_test = get_labels(obj_ids, frameset)
#         test_model(model, X_test, Y_test, batch_size=128, save_fname=out_fname)
