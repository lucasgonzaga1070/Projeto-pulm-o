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
from bibNodulos import *
from time import time
from bibIA import numero_Saidas
from geradorAtibutos import *


def checa_Minimo(im):
    if im.min() < 0:
        return 1
    return 0


# Contrução da função de ativação --------------------------------------------------------------------------------------
def gaussian(x, mean, std):
    return K.exp(-1 * (((0.5 * (x - mean)) / std) ** 2))


# Importando imagens ---------------------------------------------------------------------------------------------------
imgs = []
gss = []
path = r'C:\Users\lgxnt\Imagens\imgIC'

for num in range(1, 11):
    img = skimage.img_as_float(cv2.imread(path + r'\img{}.pgm'.format(num), 0))
    gs = skimage.img_as_float(cv2.imread(path + r'\gs{}.pgm'.format(num), 0))
    if checa_Minimo(img):
        print(img.min())
    img = skimage.exposure.rescale_intensity(img, in_range=(0, img.max()))
    gs = gs > 0.5

    imgs.append(img)
    gss.append(gs)


# Gerando Atributos ----------------------------------------------------------------------------------------------------
atributo1 = []  # Desvio Padrão
atributo2 = []  # Média
atributo3 = []  # Só o pulmão
atributo4 = []  # Wavelet

for img0 in imgs:
    dp = scipy.ndimage.generic_filter(img0, function=np.std, size=5)
    if checa_Minimo(dp):
        print('2')
    mean = scipy.signal.convolve2d(img0, np.ones((5, 5))*(1/25), 'same')
    if checa_Minimo(mean):
        print('3')
    lung = isola_Pulmao2D(img0)
    if checa_Minimo(lung):
        print('4')
    wlt = faz_TWC(img0, 'db1')
    if checa_Minimo(wlt):
        print(wlt.min())

    atributo1.append(mean)
    atributo2.append(dp)
    atributo3.append(lung)
    atributo4.append(wlt)

# Vetorizando imagens para obtenção dops atributos ---------------------------------------------------------------------
teste = 0
imgTreino = []
gsTreino = []
a1Treino = []
a2Treino = []
a3Treino = []
a4Treino = []


for i in range(len(imgs)):
    if i != teste:
        imgTreino.append(imgs[i].flatten())
        gsTreino.append(gss[i].flatten())
        a1Treino.append(atributo1[i].flatten())
        a2Treino.append(atributo2[i].flatten())
        a3Treino.append(atributo3[i].flatten())
        a4Treino.append(atributo4[i].flatten())

imgTreino = np.concatenate(imgTreino)
gsTreino = np.concatenate(gsTreino)
a1Treino = np.concatenate(a1Treino)
a2Treino = np.concatenate(a2Treino)
a3Treino = np.concatenate(a3Treino)
a4Treino = np.concatenate(a4Treino)

imgTeste = imgs[teste].flatten()
gsTeste = gss[teste].flatten()
a1Teste = atributo1[teste].flatten()
a2Teste = atributo2[teste].flatten()
a3Teste = atributo3[teste].flatten()
a4Teste = atributo4[teste].flatten()

# Separação dados de treino e teste ------------------------------------------------------------------------------------
xTreino = np.column_stack((imgTreino, a1Treino, a2Treino, a3Treino, a4Treino))
xTeste = np.column_stack((imgTeste, a1Teste, a2Teste, a3Teste, a4Teste))

yTreino = gsTreino
yTeste = gsTeste

# Construção do neuronio no formato standalone input layer -------------------------------------------------------------
inputs = tf.keras.layers.Input((xTreino.shape[1],))
layer1 = tf.keras.layers.Dense(8, bias_initializer='ones', activation=tf.nn.relu)(inputs)
layer2 = tf.keras.layers.Dense(8, activation=tf.nn.relu)(layer1)

outputs = tf.keras.layers.Dense(1, activation=tf.nn.softmax)(layer2)

# Compilar e treinar neuronio ------------------------------------------------------------------------------------------
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.summary()
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(xTreino, yTreino, validation_data=(xTeste, yTeste), batch_size=64000, epochs=250)

# Testar neuronio ------------------------------------------------------------------------------------------------------
outPrediction = model.predict(xTeste)
outPrediction = outPrediction > 0.5

precisao = accuracy_score(yTeste, outPrediction)
print("\nPrecisão: {}".format(precisao))

matConfusao = confusion_matrix(yTeste, outPrediction)

# Reconstruir e mostrar imagem -----------------------------------------------------------------------------------------
dim = gss[teste].shape
output = outPrediction.reshape(dim)
goldSTD = yTeste.reshape(dim)

plt.figure()
plt.subplot(121)
plt.axis('off')
plt.imshow(output,  cmap='gray')
plt.title('Output')
plt.subplot(122)
plt.axis('off')
plt.imshow(goldSTD,  cmap='gray')
plt.title('Gold Standart')
plt.show()
