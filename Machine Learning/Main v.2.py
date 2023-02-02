from bibNodulos import *
from time import time
from bibIA import numero_Saidas
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import scipy.ndimage
from geradorAtibutos import *
import tensorflow as tf

s = time()
# Importando imagens ---------------------------------------------------------------------------------------------------
imgs = []
gss = []
path = r'C:\Users\lgxnt\Imagens\imgIC'

for num in range(1, 11):
    img = skimage.img_as_float(cv2.imread(path+r'\img{}.pgm'.format(num), 0))
    gs = skimage.img_as_float(cv2.imread(path + r'\gs{}.pgm'.format(num), 0))

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
    mean = scipy.signal.convolve2d(img0, np.ones((5, 5))*(1/25), 'same')
    lung = isola_Pulmao2D(img0)
    wlt = faz_TWC(img0, 'db1')

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

# Número de saídas e de classes automaticamente ------------------------------------------------------------------------
numSaidas, numClasses = numero_Saidas(yTreino)

# Projetando a rede de neuronios ---------------------------------------------------------------------------------------
model = Sequential()
model.add(Dense(units=xTeste.shape[1], activation='linear', input_dim=xTeste.shape[1]))
model.add(Dense(units=12, activation='sigmoid'))
model.add(Dense(units=numSaidas, activation='sigmoid'))

opt = tf.keras.optimizers.Adam(learning_rate=0.000000001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
print('4')
# Treinando a IA -------------------------------------------------------------------------------------------------------
model.fit(xTreino, yTreino, validation_data=(xTeste, yTeste), batch_size=5000, epochs=50)

# Testando a IA --------------------------------------------------------------------------------------------------------
outPrediction = model.predict(xTeste)
outPrediction = outPrediction > 0.5

precisao = accuracy_score(yTeste, outPrediction)
print("\nPrecisão: {}".format(precisao))

# Matriz de confusão ---------------------------------------------------------------------------------------------------
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

if not output.max():
    print('\nNão funcionou. :(')
else:
    print('\nTalvez tenha funcionado. :)')

f = time()
print('\nTempo de duração: {:2f}'.format(f-s))