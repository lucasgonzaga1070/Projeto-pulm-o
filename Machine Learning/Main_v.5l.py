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


# Contrução da função de ativação --------------------------------------------------------------------------------------
def gaussian(x, mean, std):
    return K.exp(-1 * (((0.5 * (x - mean)) / std) ** 2))


# Importando imagens ---------------------------------------------------------------------------------------------------
img = sk.img_as_float(cv2.imread(path + r'\img1.pgm', -1))
gs = sk.img_as_float(cv2.imread(path + r'\gs1.pgm', -1)) > 0.5
img[~isola_Pulmao2D(img)] = 0.0

M, N = img.shape
teste = 0

imgs = np.zeros((M, N, 29))
gss = np.zeros((M, N, 29))

imgs[:, :, 0] = img
gss[:, :, 0] = gs

for i in range(1, 30):
    img = sk.img_as_float(cv2.imread(path + r'\img{}.pgm'.format(i), -1))
    gs = sk.img_as_float(cv2.imread(path + r'\gs{}.pgm'.format(i), -1)) > 0.5
    lung = isola_Pulmao2D(img)
    img[~lung] = 0.0

    hist = skimage.exposure.histogram(img)
    zeros = 0
    cont = 2

    lim = []
    while zeros < 2 and cont < hist[0].size:
        if hist[0][cont] > 10 and hist[0][cont - 1] <= 10 and zeros < 1:
            zeros += 1
            lim.append(cont)
            # print(len(lim))
        elif hist[0][cont] < 10 and hist[0][cont - 1] >= 10 and zeros >= 1:
            zeros += 1
            lim.append(cont)
            # print(len(lim))
        cont += 1

    img = skimage.exposure.rescale_intensity(img, in_range=(hist[1][lim[0]], hist[1][lim[1]]))

    imgs[:, :, i - 1] = img
    gss[:, :, i - 1] = gs

plt.imshow(imgs[:, :, teste], cmap='gray')
plt.axis('off')
plt.show()

plt.imshow(gss[:, :, teste], cmap='gray')
plt.axis('off')
plt.show()

# Teste ----------------------------------------------------------------------------------------------------------------
Cmin, Lmin, dC, dL = cv2.selectROI(windowName='window', img=imgs[:, :, teste], showCrosshair=True)
cv2.destroyAllWindows()

mediaTeste = np.mean(imgs[Lmin:Lmin+dL, Cmin:Cmin+dC, teste])

# Vetorizando imagens para obtenção dops atributos ---------------------------------------------------------------------
slice = [x for x in range(29) if x != teste]

img_vetor = imgs[:, :, slice].flatten()
gs_vetor = gss[:, :, slice].flatten()

imgTeste_vetor = imgs[:, :, teste].flatten()
gsTeste_vetor = gss[:, :, teste].flatten()

# Separação dados de treino e teste ------------------------------------------------------------------------------------
xTreino = np.transpose([img_vetor, img_vetor, img_vetor])
yTreino = gs_vetor

xTeste = np.transpose([imgTeste_vetor, imgTeste_vetor, imgTeste_vetor])
yTeste = gsTeste_vetor

# Construção do neuronio no formato standalone input layer -------------------------------------------------------------
inputs = tf.keras.layers.Input((xTreino.shape[1],))
layer1 = tf.keras.layers.Dense(1, kernel_initializer='one', bias_initializer='zeros', input_shape=(xTreino.shape[1],))(
    inputs)

layer1 = gaussian(layer1, mean=xTreino.shape[1]*mediaTeste, std=0.05)

# Compilar e treinar neuronio ------------------------------------------------------------------------------------------
model = tf.keras.Model(inputs=[inputs], outputs=[layer1])
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(xTreino, yTreino, validation_data=(xTeste, yTeste), batch_size=100000, epochs=250)

# Testar neuronio ------------------------------------------------------------------------------------------------------
outPrediction = model.predict(xTeste)
outPrediction = outPrediction > 0.5

precisao = accuracy_score(yTeste, outPrediction)
print("\nPrecisão: {}".format(precisao))

matConfusao = confusion_matrix(yTeste, outPrediction)

# Reconstruir e mostrar imagem -----------------------------------------------------------------------------------------
output = outPrediction.reshape(M, N)

plt.figure()
plt.axis('off')
plt.imshow(output, cmap='gray')
plt.show()
