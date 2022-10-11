## rgb cipherimage retrieval - server
import sys
sys.path.append("..")
import numpy as np
from Encryption_algorithm.JPEG.jacdecColorHuffman import jacdecColor
from Encryption_algorithm.JPEG.jdcdecColorHuffman import jdcdecColor
from Encryption_algorithm.JPEG.Quantization import *
from Encryption_algorithm.JPEG.invzigzag import invzigzag
from Encryption_algorithm.JPEG.zigzag import zigzag
import matplotlib.pyplot as plt
#from scipy.interpolate import make_interp_spline
import tqdm

# 64 histograms for each cipherimage
DC_bin_interval = [i for i in range(-1056, 1057, 64)]
AC_bin_interval = [-1024] + [i for i in range(-200, 200, 50)] + [1024]
DC_histgram_dimension = 33
AC_histgram_dimension = 9

def extract_feature(dc, ac, size, type, QF, N = 8):
    
    _, acarr = jacdecColor(ac, type)
    _, dcarr = jdcdecColor(dc, type)
    acarr = np.array(acarr)
    dcarr = np.array(dcarr)
    
    if type == 'Y':
        row, col = size
        row = int(16*np.ceil(row/16))
        col = int(16*np.ceil(col/16))
    else:
        row, col = size
        row = int(8*np.ceil(row/16))
        col = int(8*np.ceil(col/16))
        
    Eob = np.where(acarr==999)
    Eob = Eob[0]
    count = 0
    kk = 0
    ind1 = 0
    allblock8 = np.zeros([8, 8, int(row*col/(8*8))])
    allblock8_number = 0
    for m in range(0, row, N):
        for n in range(0, col, N):
            ac = acarr[ind1: Eob[count]]
            ind1 = Eob[count] + 1
            count = count + 1
            acc = np.append(dcarr[kk], ac)
            az = np.zeros(64-acc.shape[0])
            acc = np.append(acc, az)
            temp = invzigzag(acc, 8, 8)
            temp = iQuantization(temp, QF, type)
            allblock8[:, :, allblock8_number] = temp
            kk = kk + 1
            allblock8_number = allblock8_number + 1
    
    allcoe = np.zeros([allblock8.shape[2], 64])
    for j in range(0, allblock8.shape[2]):
        temp = allblock8[:,:,j]
        allcoe[j, :] = zigzag(temp)

    hist_img = np.zeros(AC_histgram_dimension*63+DC_histgram_dimension)
    for j in range(0, 64):
        if j == 0:
            hist_tmp = np.zeros([1, DC_histgram_dimension])
            tmp = allcoe[:, j]
            hist_t = np.histogram(tmp, bins=DC_bin_interval)
            hist_t = hist_t[0]
            hist_tmp[0, :] = hist_t
            hist_tmp = hist_tmp.T
            hist_norm = hist_tmp / np.sum(hist_tmp)
            hist_img[0:DC_histgram_dimension] = hist_norm[:, 0]
        else:
            hist_tmp = np.zeros([1, AC_histgram_dimension])
            tmp = allcoe[:, j]
            hist_t = np.histogram(tmp, bins=AC_bin_interval)
            hist_t = hist_t[0]
            hist_tmp[0, :] = hist_t
            hist_tmp = hist_tmp.T
            hist_norm = hist_tmp/np.sum(hist_tmp)
            #hist_img[0:DC_histgram_dimension] = hist_norm[:, 0]
            hist_img[DC_histgram_dimension+(j-1)*AC_histgram_dimension:DC_histgram_dimension+j*AC_histgram_dimension] = hist_norm[:, 0]
    return hist_img


def extract_all_component_feature(dcallY, acallY, dcallCb, acallCb, dcallCr, acallCr, img_size, QF = 100):
    
    image_num = len(dcallY)
    hist64_rgb_Y = np.zeros([AC_histgram_dimension*63+DC_histgram_dimension, image_num])
    hist64_rgb_Cb = np.zeros([AC_histgram_dimension*63+DC_histgram_dimension, image_num])
    hist64_rgb_Cr = np.zeros([AC_histgram_dimension*63+DC_histgram_dimension, image_num])
    for k in tqdm.tqdm(range(image_num)):
        hist64_rgb_Y[:, k] = extract_feature(dcallY[k].astype(np.int8), acallY[k].astype(np.int8), img_size[k], "Y", QF)
        hist64_rgb_Cb[:, k] = extract_feature(dcallCb[k].astype(np.int8), acallCb[k].astype(np.int8), img_size[k], "C", QF)
        hist64_rgb_Cr[:, k] = extract_feature(dcallCr[k].astype(np.int8), acallCr[k].astype(np.int8), img_size[k], "C", QF)
    return np.concatenate([hist64_rgb_Y.T, hist64_rgb_Cb.T, hist64_rgb_Cr.T], axis=1)