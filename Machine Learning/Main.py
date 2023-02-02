from bibNodulos import *
from time import time
from bibIA import numero_Saidas
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import scipy.ndimage


def vetorTT(listOfFiles):
    numImg = len(listOfFiles)
    # IMPORTANDO IMAGENS E ARMAZENANDO EM LISTAS -----------------------------------------------------------------------
    imgs = []
    gss = []
    nods = []
    for file in listOfFiles:
        img, gs, nod = create_GoldSTD(file)

        imgs.append(img[:, :, sliceWITHnod(gs)])
        gss.append(gs[:, :, sliceWITHnod(gs)])
        nods.append(nod)

    dim = imgs[numImg - 1].shape
    # GERANDO ATRIBUTOS ------------------------------------------------------------------------------------------------
    atributo1 = []
    # atributo2 = []
    for i in range(numImg):
        atributo1.append(scipy.ndimage.median_filter(imgs[i], 7))
        # atributo2.append(atributoHighPass(imgs[i]))

    # SEPARANDO VETORES DE TREINO E DE TESTE ---------------------------------------------------------------------------
    imgTreino = []
    gsTreino = []
    a1Treino = []
    # a2Treino = []

    for i in range(1, numImg - 2):
        imgTreino.append(imgs[i].flatten())
        gsTreino.append(gss[i].flatten())
        a1Treino.append(atributo1[i].flatten())
        # a2Treino.append(atributo2[i].flatten())

    imgTreino = np.concatenate(imgTreino)
    gsTreino = np.concatenate(gsTreino)
    a1Treino = np.concatenate(a1Treino)
    # a2Treino = np.concatenate(a2Treino)

    imgTeste = imgs[0].flatten()
    gsTeste = gss[0].flatten()
    a1Teste = atributo1[0].flatten()
    # a2Teste = atributo2[numImg-1].flatten()

    xTreino = np.column_stack((imgTreino, a1Treino))
    xTeste = np.column_stack((imgTeste, a1Teste))

    yTreino = gsTreino
    yTeste = gsTeste

    return xTreino, xTeste, yTreino, yTeste, dim


xTreino, xTeste, yTreino, yTeste, dim = vetorTT(['LIDC-IDRI-{:04d}'.format(x) for x in range(1, 21)])

# Número de saídas e de classes automaticamente ------------------------------------------------------------------------
numSaidas, numClasses = numero_Saidas(yTreino)

# Projetando a rede de neuronios ---------------------------------------------------------------------------------------
model = Sequential()
model.add(Dense(units=xTreino.shape[1], activation='linear', input_dim=xTreino.shape[1]))
model.add(Dense(units=12, activation='sigmoid'))
model.add(Dense(units=numSaidas, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinando a IA -------------------------------------------------------------------------------------------------------
model.fit(xTreino, yTreino, validation_data=(xTeste, yTeste), batch_size=int(xTreino.shape[0]/10), epochs=500)

# Testando a IA --------------------------------------------------------------------------------------------------------
outPrediction = model.predict(xTeste)
outPrediction = outPrediction > 0.5

precisao = accuracy_score(yTeste, outPrediction)
print("\nPrecisão: {}".format(precisao))

# Matriz de confusão ---------------------------------------------------------------------------------------------------
matConfusao = confusion_matrix(yTeste, outPrediction)

# Reconstruir e mostrar imagem -----------------------------------------------------------------------------------------
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
