import numpy as np
import scipy.signal


def numero_Saidas(goldSTD):
    numClasses = np.max(goldSTD).astype(int)
    return int(np.ceil(np.sqrt(numClasses))), numClasses


def dec2bin(goldSTD):
    a, b = numero_Saidas(goldSTD)
    array = np.zeros((goldSTD.shape[0], 1))
    array[:, 0] = goldSTD

    binario = np.unpackbits(array.astype(np.uint8), axis=1, bitorder='little')
    return binario[:, :a]


def fazerMascaraGauss2D(media, desvio):
    x = np.arange(media * 2 + 1)
    g = np.e ** ((-1 / 2) * ((x - media) / desvio) ** 2)

    g1 = np.zeros((media * 2 + 1, media * 2 + 1), float)
    g1[media, :] = g
    w_Gauss2D = scipy.signal.convolve2d(g1, g1.transpose(), 'same')
    w_Gauss2D_N = w_Gauss2D / np.sum(w_Gauss2D)

    return w_Gauss2D_N
