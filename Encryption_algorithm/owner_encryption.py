## laod plain-images and secret keys
import scipy.io as scio
from encryption_utils import ksa
from encryption_utils import prga
from encryption_utils import yates_shuffle
import tqdm
from encryption_utils import loadImageSet
from JPEG.rgbandycbcr import rgb2ycbcr
import cv2
import copy
from JPEG.jdcencColor import jdcencColor
from JPEG.zigzag import zigzag
from JPEG.jacencColor import jacencColor
from JPEG.Quantization import *
from cipherimageRgbGenerate import Gen_cipher_images

def encryption_each_component(image_component, keys, type, row, col, N, QF):
    # generate value replacement list
    QT = getQuantizationTable(type, QF)
    dc_replace_scope = [i for i in range(-int(np.floor(1024/QT[0,0])), int(np.floor(1024/QT[0,0])))]
    ac_replace_scope = [i for i in range(-200, 201)]
    del ac_replace_scope[int(len(ac_replace_scope)/2)]
    
    dc_replace_list = yates_shuffle(dc_replace_scope, keys)
    keys = keys[6290:]
    ac_replace_list = yates_shuffle(ac_replace_scope, keys)
    keys = keys[1608:]
    ac_replace_list.insert(int(len(ac_replace_list)/2), 0)
    
    # generate block permutation vector
    block8_number = int((row*col)/(8*8))
    data = [i for i in range(1, block8_number+1)] 
    p_block = yates_shuffle(data, keys) 
    
    allblock8 = np.zeros([8, 8, int(row * col / (8 * 8))])
    allblock8_number = 0
    for m in range(0, row, N):
        for n in range(0, col, N):
            t = image_component[m:m + N, n:n + N] - 128
            allblock8[:, :, allblock8_number] = t
            allblock8_number = allblock8_number + 1
    
    # block permutation
    allblock8_permute = copy.copy(allblock8)
    for i in range(0, len(p_block)):
        allblock8_permute[:, :, i] = allblock8[:, :, p_block[i]-1]
    allblock8 = copy.copy(allblock8_permute)
    del allblock8_permute
    
    # Huffman coding
    dc = 0
    dccof= []
    accof = []
    for i in range(0, allblock8_number):
        t = copy.copy(allblock8[:, :, i])
        t = cv2.dct(t)  # DCT
        temp = Quantization(t, type=type)  # Quanlity
        temp[0,0] = dc_replace_list[int(temp[0,0]+np.floor(1024/QT[0,0]))]
        if i == 0:
            dc = temp[0, 0]
            dc_component = jdcencColor(dc, type)
            dccof = np.append(dccof, dc_component)
        else:
            dc = temp[0, 0] - dc
            dc_component = jdcencColor(dc, type)
            dccof = np.append(dccof, dc_component)
            dc = temp[0, 0]
        acseq = []
        aczigzag = zigzag(temp)
        eobi = 0
        for j in range(63, -1, -1):
            if aczigzag[j] != 0:
                eobi = j
                break
        if eobi == 0:
            acseq = np.append(acseq, [999])
        else:
            acseq = np.append(acseq, aczigzag[1: eobi + 1])
            acseq[acseq>200] = 200
            acseq[acseq<-200] = -200
            for j in range(0, len(acseq)):
                acseq[j] = ac_replace_list[int(acseq[j]+200)]
            acseq = np.append(acseq, [999])
        ac_component = jacencColor(acseq, type)
        accof = np.append(accof, ac_component)

    return dccof, accof


def encryption(img, keyY, keyCb, keyCr, QF, N=8):
    # N: block size
    # QF: quality factor
    row, col, _ = img.shape
    plainimage = rgb2ycbcr(img)
    plainimage = plainimage.astype(np.float64)
    Y = plainimage[:, :, 0]
    Cb = plainimage[:, :, 1]
    Cr = plainimage[:, :, 2]
    
    for i in range(0, int(16*np.ceil(col/16)-col)):
        Y=np.c_[Y, Y[:,-1]]
        Cb=np.c_[Cb, Cb[:,-1]]
        Cr=np.c_[Cr, Cr[:,-1]]
        
    for i in range(0, int(16*np.ceil(row/16)-row)):
        Y=np.r_[Y, [Y[-1,:]]]
        Cb=np.r_[Cb, [Cb[-1,:]]]
        Cr=np.r_[Cr, [Cr[-1,:]]]
    
    [row, col] = Y.shape
    
    Cb = cv2.resize(Cb, 
                   (int(col/2), int(row/2)), 
                   interpolation=cv2.INTER_CUBIC)
    Cr = cv2.resize(Cr, 
                   (int(col/2), int(row/2)), 
                   interpolation=cv2.INTER_CUBIC)
    
    # Y component
    dccofY, accofY = encryption_each_component(Y, keyY, type='Y', row=row, col=col, N=N, QF=QF)
    ## Cb and Cr component
    dccofCb, accofCb = encryption_each_component(Cb, keyCb, type='Cb', row=int(row/2), col=int(col/2), N=N, QF=QF)
    dccofCr, accofCr = encryption_each_component(Cr, keyCr, type='Cr', row=int(row/2), col=int(col/2), N=N, QF=QF)

    accofY = accofY.astype(np.int8)
    dccofY = dccofY.astype(np.int8)
    accofCb = accofCb.astype(np.int8)
    dccofCb = dccofCb.astype(np.int8)
    accofCr = accofCr.astype(np.int8)
    dccofCr = dccofCr.astype(np.int8)
    return accofY, dccofY, accofCb, dccofCb, accofCr, dccofCr


## read plain-images
def read_plain_images():
    plainimage_all = loadImageSet('../data/plainimages/*.jpg')
    # save size information
    img_size = []
    for temp in plainimage_all:
        row, col, _ = temp.shape
        img_size.append((row, col))
    np.save("../data/plainimages.npy", plainimage_all)
    np.save("../data/img_size.npy", img_size)
    return plainimage_all


## generate encryption key and embedding key
# keys are independent from plainimage
# encryption key generation - RC4
def generate_keys(control_length=256*284):
    # secret keys
    data_lenY = np.ones([1, int(control_length)])
    keyY = scio.loadmat('../data/key.mat')  # Y component encryption key
    keyY = keyY['key'][0]
    keyCb = scio.loadmat('../data/keyCb.mat')  # Cb component encryption key
    keyCb = keyCb['keyCb'][0]
    keyCr = scio.loadmat('../data/keyCr.mat')  # Cr component encryption key
    keyCr = keyCr['keyCr'][0]
    # keys stream
    s = ksa(keyY)
    r = prga(s, data_lenY)
    encryption_keyY = ''
    for i in range(0, len(r)):
        temp1 = str(r[i])
        temp2 = bin(int(temp1, 10))
        temp2 = temp2[2:]
        for j in range(0, 8 - len(temp2)):
            temp2 = '0' + temp2
        encryption_keyY = encryption_keyY + temp2

    data_lenC = np.ones([1, int(control_length // 4)])
    s1 = ksa(keyCb)
    r1 = prga(s1, data_lenC)
    encryption_keyCb = ''
    for i in range(0, len(r1)):
        temp1 = str(r1[i])
        temp2 = bin(int(temp1, 10))
        temp2 = temp2[2:]
        for j in range(0, 8 - len(temp2)):
            temp2 = '0' + temp2
        encryption_keyCb = encryption_keyCb + temp2

    s2 = ksa(keyCr)
    r2 = prga(s2, data_lenC)
    encryption_keyCr = ''
    for i in range(0, len(r2)):
        temp1 = str(r2[i])
        temp2 = bin(int(temp1, 10))
        temp2 = temp2[2:]
        for j in range(0, 8 - len(temp2)):
            temp2 = '0' + temp2
        encryption_keyCr = encryption_keyCr + temp2

    return encryption_keyY, encryption_keyCb, encryption_keyCr


if __name__ == '__main__':
    ## image encryption
    QF = 100
    acallY = []
    dcallY = []
    acallCb = []
    dcallCb = []
    acallCr = []
    dcallCr = []
    plainimage_all = read_plain_images()
    encryption_keyY, encryption_keyCb, encryption_keyCr = generate_keys()
    #for k in tqdm.tqdm([i for i in range(len(plainimage_all))]):
    for k in tqdm.tqdm([i for i in range(0,1)]):
        img = plainimage_all[k]
        accofY, dccofY, accofCb, dccofCb, accofCr, dccofCr = encryption(img, encryption_keyY, encryption_keyCb, encryption_keyCr, QF)

        acallY.append(accofY)
        dcallY.append(dccofY)
        acallCb.append(accofCb)
        dcallCb.append(dccofCb)
        acallCr.append(accofCr)
        dcallCr.append(dccofCr)

    np.save('../data/acallY.npy', acallY)
    np.save('../data/dcallY.npy', dcallY)
    np.save('../data/acallCb.npy', acallCb)
    np.save('../data/dcallCb.npy', dcallCb)
    np.save('../data/acallCr.npy', acallCr)
    np.save('../data/dcallCr.npy', dcallCr)
    # generate cipher-images
    Gen_cipher_images()
