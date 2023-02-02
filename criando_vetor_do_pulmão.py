import cv2
import skimage as sk
import numpy as np
import matplotlib.pyplot as plt
from geradorAtibutos import isola_Pulmao2D


def plotar(*im, titles):
    if len(titles) == 0:
        titles = ['' for i in range(len(im))]
    elif type(titles) == str:
        titles = [titles]

    for img, pos, title in zip(im, range(1, len(im) + 1), titles):
        plt.subplot(1, len(im), pos)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()


path = r'C:\Users\lgxnt\Imagens\imgIC'

for i in range(1, 2):
    img = sk.img_as_float(cv2.imread(path + r'\img{}.pgm'.format(i), -1))
    gs = sk.img_as_float(cv2.imread(path + r'\gs{}.pgm'.format(i), -1))
    lung = isola_Pulmao2D(img)


def constroi_vetor_pulmao(img, imgPulmao):
    lista = []
    for i in range(imgPulmao.shape[0]):
        for j in range(imgPulmao.shape[1]):
            if imgPulmao[i, j] == 1:
                lista.append(img[i, j])

    return lista


def constroi_img_pulmao(vetorPulmao, imgPulmao):
    out = np.zeros_like(imgPulmao, float)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if imgPulmao[i, j] == 1:
                out[i, j] = vetorPulmao.pop(0)

    return out


vetor = constroi_vetor_pulmao(img, lung)
out = constroi_img_pulmao(vetor, lung)
plotar(img, out, titles=['img', 'out'])