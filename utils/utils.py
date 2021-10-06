import math
import sys
from operator import itemgetter

import cv2
import matplotlib.pyplot as plt
import numpy as np


#---------------------------------#
#   Picture pre-processing
#   Gauss normalizing
#---------------------------------#
def pre_process(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y
    
#---------------------------------#
#   l2 normalization
#---------------------------------#
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output
    
#---------------------------------#
#   Calculate 128-dim feature vector
#---------------------------------#
def calc_128_vec(model,img):
    face_img = pre_process(img)
    pre = model.predict(face_img)
    pre = l2_normalize(np.concatenate(pre))
    pre = np.reshape(pre,[128])
    return pre

#---------------------------------#
#   Calculated face distance
#---------------------------------#
def face_cosdistance(type, face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    elif type == 'cosine':
        res = (np.inner(face_encodings, face_to_compare) / (
                (np.linalg.norm(np.array(face_encodings).T, axis=0).reshape(-1, 1)) * ((np.linalg.norm(np.array(face_to_compare).T, axis=0).reshape(-1, 1)).T)))[0][0]
    elif type == 'pearson':
        X = np.vstack([np.array(face_encodings), np.array(face_to_compare)])
        res = np.corrcoef(X)[0][1]
    elif type == 'tanimoto':
        pq = np.dot(np.array(face_encodings), np.array(face_to_compare))
        p_square = np.linalg.norm(np.array(face_encodings))
        q_square = np.linalg.norm(np.array(face_to_compare))
        res = pq / (p_square ** 2 + q_square ** 2 - pq)
    return res.tolist()


