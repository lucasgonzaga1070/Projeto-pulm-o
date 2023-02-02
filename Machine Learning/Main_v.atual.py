import cv2
import numpy as np
import keras
from keras import backend as K
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sb
import matplotlib.pyplot as plt
import skimage as sk
from geradorAtibutos import *

path = r'C:\Users\lgxnt\Imagens\imgIC'


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


# Contrução da função de ativação --------------------------------------------------------------------------------------
def gaussian(x, mean, std):
    return K.exp(-1 * (((0.5 * (x - mean)) / std) ** 2))


# Importando imagens ---------------------------------------------------------------------------------------------------
teste = 0
imgs = []
gss = []
lungs = []

lista_ignora = np.array([11, 15, 16, 20, 22, 24, 25, 28])

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
        if hist[0][cont] > 1000 and hist[0][cont - 1] <= 1000 and zeros < 1:
            zeros += 1
            lim.append(cont)
        elif hist[0][cont] < 1000 and hist[0][cont - 1] >= 1000 and zeros >= 1:
            zeros += 1
            lim.append(cont)
        cont += 1

    rescales.append(skimage.exposure.rescale_intensity(img, in_range=(hist[1][lim[0]], hist[1][lim[1]])))

plt.imshow(rescales[teste], cmap='gray')
plt.show()

img_copy = rescales[teste].copy()
img_copy[~lungs[teste]] = 0
plt.imshow(img_copy, cmap='gray')
plt.show()

plt.imshow(gss[teste], cmap='gray')
plt.show()

# Gerando Atributos ----------------------------------------------------------------------------------------------------
atributo = []  # Média
atributo2 = []  # Desvio Padrão

tam = 35
for rescale in rescales:
    mean = scipy.signal.convolve2d(rescale, np.ones((tam, tam)) * (1 / tam ** 2), 'same')
    dp = scipy.ndimage.generic_filter(rescale, function=np.std, size=tam)
    atributo.append(mean)
    atributo2.append(dp)

# Vetorizando imagens para obtenção dos atributos ----------------------------------------------------------------------
imgTreino = []
gsTreino = []
mediaTreino = []
dpTreino = []

for i in range(len(imgs)):
    if i != teste:
        imgTreino.append(constroi_vetor_pulmao(rescales[i], lungs[i]))
        gsTreino.append(constroi_vetor_pulmao(gss[i], lungs[i]))
        mediaTreino.append(constroi_vetor_pulmao(atributo[i], lungs[i]))
        dpTreino.append(constroi_vetor_pulmao(atributo2[i], lungs[i]))

    else:
        imgTeste = constroi_vetor_pulmao(rescales[i], lungs[i])
        gsTeste = constroi_vetor_pulmao(gss[i], lungs[i])
        mediaTeste = constroi_vetor_pulmao(atributo[i], lungs[i])
        dpTeste = constroi_vetor_pulmao(atributo2[i], lungs[i])

imgTreino = np.concatenate(imgTreino)
gsTreino = np.concatenate(gsTreino)
mediaTreino = np.concatenate(mediaTreino)
dpTreino = np.concatenate(dpTreino)

# Separação dados de treino e teste ------------------------------------------------------------------------------------
xTreino = np.column_stack((imgTreino, dpTreino, mediaTreino))
xTeste = np.column_stack((imgTeste, dpTeste, mediaTeste))

yTreino = np.array(gsTreino)
yTeste = np.array(gsTeste)

# Teste ----------------------------------------------------------------------------------------------------------------
'''Cmin, Lmin, dC, dL = cv2.selectROI(windowName='window', img=rescales[teste], showCrosshair=True)
cv2.destroyAllWindows()

mediaTeste = np.mean(rescales[teste][Lmin:Lmin+dL, Cmin:Cmin+dC])
stdTeste = np.std(rescales[teste][Lmin:Lmin+dL, Cmin:Cmin+dC])'''

# Construção do neuronio no formato standalone input layer -------------------------------------------------------------
inputs = tf.keras.layers.Input((xTreino.shape[1],))
layer1 = tf.keras.layers.Dense(1, kernel_initializer='one', bias_initializer='zeros', input_shape=(xTreino.shape[1],))(
         inputs)

layer1 = gaussian(layer1, mean=xTreino.shape[1]*0.5021228626943764, std=0.16584392401312767)


# Compilar e treinar neuronio ------------------------------------------------------------------------------------------
model = tf.keras.Model(inputs=[inputs], outputs=[layer1])
a = model.get_weights()
opt = tf.keras.optimizers.Adam(learning_rate=0.000001)

model.summary()
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(xTreino, yTreino, validation_data=(xTeste, yTeste), batch_size=100000, epochs=1000)
b = model.get_weights()
# Testar neuronio ------------------------------------------------------------------------------------------------------
outPrediction = model.predict(xTeste)
outPrediction = outPrediction > 0.5

precisao = accuracy_score(yTeste, outPrediction)
print("\nPrecisão: {}".format(precisao))

matConfusao = confusion_matrix(yTeste, outPrediction)

# Reconstruir e mostrar imagem -----------------------------------------------------------------------------------------
output1 = constroi_img_pulmao(list(outPrediction), lungs[teste])

plt.figure()
plt.imshow(output1, cmap='gray')
plt.show()

