import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage


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


# Importando imagens ---------------------------------------------------------------------------------------------------
path = r'C:\Users\lgxnt\Imagens\imgIC'

img = skimage.img_as_float(cv2.imread(path + r'\img1.pgm', -1))
plotar(img, titles=['Original'])


def controi_vetor_da_imagem(imagem):
    vetor_Circunferencia = []
    raio = imagem.shape[0] / 2

    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            if np.sqrt((i - img.shape[1] / 2) ** 2 + (j - img.shape[0] / 2) ** 2) <= raio + 2:
                vetor_Circunferencia.append(img[i, j])

    return vetor_Circunferencia


def controi_imagem_do_vetor(vetor, M, N):
    img_Output = np.zeros((M, N))
    raio = M / 2

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.sqrt((i - img.shape[1] / 2) ** 2 + (j - img.shape[0] / 2) ** 2) <= raio + 2:
                img_Output[i, j] = vetor.pop(0)

    return img_Output


vetorMatriz = controi_vetor_da_imagem(img)
imgOut = controi_imagem_do_vetor(vetorMatriz, img.shape[0], img.shape[1])
plotar(imgOut, titles=['Out'])

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i, j] != imgOut[i, j]:
            print("Achou um diferente")
