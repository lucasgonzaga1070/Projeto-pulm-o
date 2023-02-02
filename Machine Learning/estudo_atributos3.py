from geradorAtibutos import *
import scipy.ndimage.measurements


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


path = r'C:\Users\lgxnt\Imagens\imgIC'
# Importando imagens ---------------------------------------------------------------------------------------------------
teste = 0
imgs = []
gss = []
lungs = []

lista_ignora = np.array([11, 15, 16, 20, 22, 24, 25, 28])

numerador = []
denominador = []

for i in range(1, 29):
    if not np.any(i == lista_ignora):
        img = skimage.img_as_float(cv2.imread(path + r'\img{}.pgm'.format(i), -1))
        gs = skimage.img_as_float(cv2.imread(path + r'\gs{}.pgm'.format(i), -1)) > 0.5
        lung = skimage.img_as_float(cv2.imread(path + r'\lungs{}.pgm'.format(i), -1)) > 0.5

        imgs.append(img)
        gss.append(gs)
        lungs.append(lung)

# Ajuste de intensidade ------------------------------------------------------------------------------------------------
rescales = []

for img in imgs:
    hist = skimage.exposure.histogram(img)
    zeros = 0
    cont = 2

    lim = []
    while zeros < 2 and cont < hist[0].size:
        if hist[0][cont] > 100 and hist[0][cont - 1] <= 100 and zeros < 1:
            zeros += 1
            lim.append(cont)
            # print(len(lim))
        elif hist[0][cont] < 100 and hist[0][cont - 1] >= 100 and zeros >= 1:
            zeros += 1
            lim.append(cont)
            # print(len(lim))
        cont += 1

    rescales.append(skimage.exposure.rescale_intensity(img, in_range=(hist[1][lim[0]], hist[1][lim[1]])))

# Média e Desvio Padrão de intensidade dos tumores ---------------------------------------------------------------------
a = rescales[teste].copy()
a[~gss[teste]] = 0
num = np.sum(a)
den = np.sum(gss[teste])

'''media = num/den
denominador.append(media)
'''

plt.imshow(rescales[teste], cmap='gray')
plt.show()

img_copy = rescales[teste].copy()
img_copy[~lungs[teste]] = 0
plt.imshow(img_copy, cmap='gray')
plt.show()

plt.imshow(gss[teste], cmap='gray')
plt.show()

'''Cmin, Lmin, dC, dL = cv2.selectROI(windowName='window', img=img_copy, showCrosshair=True)
cv2.destroyAllWindows()

Cmin2, Lmin2, dC2, dL2 = cv2.selectROI(windowName='window', img=img_copy, showCrosshair=True)
cv2.destroyAllWindows()'''

# Fuzzy ----------------------------------------------------------------------------------------------------------------
img = rescales[teste]
M, N = img.shape
seed0 = np.round(scipy.ndimage.measurements.center_of_mass(gss[teste])).astype(int)
seed0_val = rescales[teste][seed0[0], seed0[1]]

connect = np.zeros_like(img)
afinit = np.zeros_like(img)
pathConnect = np.zeros_like(img)
exSeeds = np.zeros_like(img)

connect[seed0[0], seed0[1]] = 1
afinit[seed0[0], seed0[1]] = 1

fila = [((seed0[0], seed0[1]), connect[seed0[0], seed0[1]])]

vetorI = []
vetorH = []

index_nonzero = np.nonzero(a)
for i in range(index_nonzero[0].shape[0]):
    vetorH.append(rescales[teste][index_nonzero[0][i], index_nonzero[1][i]])
    vetorI.append(rescales[teste][index_nonzero[0][i], index_nonzero[1][i]])

vetorI = np.array(vetorI)
vetorH = np.array(vetorH)

mediaI = np.sum(0.5 * np.abs(vetorI + seed0_val)) / den
mediaH = np.sum(np.abs(vetorH - seed0_val)) / den

stdI = np.sqrt(sum(((vetorI + seed0_val)*0.5 - mediaI) ** 2) / den) + 1e-9
stdH = np.sqrt(sum(((vetorH - seed0_val) - mediaI) ** 2) / den) + 1e-9

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


plt.imshow(connect, cmap='gray')
plt.title("Conectividade")
plt.show()
plt.imshow(afinit, cmap='gray')
plt.title("Afinidade")
plt.show()
plt.imshow(pathConnect, cmap='gray')
plt.title("Conectividade Local")
plt.show()
