import sys
sys.path.append("..")
from dct_histogram import extract_all_component_feature
from Encryption_algorithm.encryption_utils import loadEncBit
import numpy as np


if __name__ == '__main__':
    dcallY, acallY, dcallCb, acallCb, dcallCr, acallCr, img_size = loadEncBit()   # load encrypted bitstream
    dct_histogram_feature = extract_all_component_feature(dcallY, acallY, dcallCb, acallCb, dcallCr, acallCr, img_size)
    np.save("../data/difffeature_matrix.npy", dct_histogram_feature)
    print('finish save features.')