from geradorAtibutos import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
path = r'C:\Users\lgxnt\Imagens\imgIC'


def afinidade(seed, pixel, h, it, mode=0, Adj=1):
    hom = np.exp((-0.5) * (((np.abs(seed - pixel) - h[0]) / h[1]) ** 2))
    inte = np.exp((-0.5) * (((0.5 * (seed + pixel) - it[0]) / it[1]) ** 2))

    if mode == 0:
        wi = 0.5
        wh = 1 - wi
        return Adj * (wh * hom + wi * inte)
    if mode == 1:
        wi = inte / (hom + inte)
        wh = 1 - wi
        return Adj * (wh * hom + wi * inte)


img = skimage.img_as_float(cv2.imread(path + r'\img{}.pgm'.format(1), -1))

M, N = img.shape

connect = np.zeros_like(img)
afinit = np.zeros_like(img)
pathConnect = np.zeros_like(img)
exSeeds = np.zeros_like(img)

Cmin, Lmin, dC, dL = cv2.selectROI(windowName='window', img=img, showCrosshair=True)
cv2.destroyAllWindows()
C = Cmin + int(dC / 2)
L = Lmin + int(dL / 2)

seed0 = img[L, C]

connect[L, C] = 1
afinit[L, C] = 1

fila = [((L, C), connect[L, C])]

regiao = img[Lmin:Lmin + dL, Cmin:Cmin + dC]

I = abs(regiao + np.ones_like(regiao) * seed0)
H = abs(np.ones_like(regiao) * seed0 - regiao)
mediaH = np.mean(H)
stdH = np.std(H) + 1e-9
mediaI = np.mean(0.5 * I)
stdI = np.std(0.5 * I) + 1e-9
cont = 0
while len(fila) != 0:
    # print(len(fila))
    fila = sorted(fila, key=lambda pixel: pixel[1], reverse=True)

    seedAtual = fila.pop(0)
    seedAtual = seedAtual[0]

    pos = [(seedAtual[0] - 1, seedAtual[1]), (seedAtual[0] + 1, seedAtual[1]),
           (seedAtual[0], seedAtual[1] + 1), (seedAtual[0], seedAtual[1] - 1)]

    for i in pos:
        if (i[0] >= M) or (i[1] >= N) or (i[0] < 0) or (i[1] < 0):
            continue
        if exSeeds[i] != 1:
            exSeeds[i] = 1

            ua = afinidade(img[seedAtual], img[i], (mediaH, stdH), (mediaI, stdI))
            afinit[i] = ua

            uk = np.min([afinit[i], connect[seedAtual]])
            pathConnect[i] = uk

            mica = np.max([connect[i], pathConnect[i]])
            connect[i] = mica

            aux = (i, mica)
            fila.append(aux)

    '''cont += 1
    if cont % 1000 == 0:
        plt.imshow(connect, cmap='gray')
        plt.title("Conectividade {}".format(cont))
        plt.show()'''

plt.imshow(img, cmap='gray')
plt.title("Original")
plt.show()
plt.imshow(connect, cmap='gray')
plt.title("Conectividade")
plt.show()
plt.imshow(afinit, cmap='gray')
plt.title("Afinidade")
plt.show()
plt.imshow(pathConnect, cmap='gray')
plt.title("Conectividade Local")
plt.show()
plt.imshow(exSeeds, cmap='gray')
plt.title("ex Sementes")
plt.show()

