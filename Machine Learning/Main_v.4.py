from bibNodulos import *
from time import time
from bibIA import numero_Saidas
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers.schedules import ExponentialDecay
import scipy.ndimage
from keras import backend as K
from geradorAtibutos import *
import tensorflow as tf
import skimage.exposure


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


wxPriwitt = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
wyPriwitt = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
s = time()
# Importando imagens ---------------------------------------------------------------------------------------------------
imgs = []
gss = []
lungs = []
path = r'C:\Users\lgxnt\Imagens\imgIC'

for num in range(1, 30):
    # print('{}'.format(num))
    img = skimage.img_as_float(cv2.imread(path + r'\img{}.pgm'.format(num), -1))
    gs = skimage.img_as_float(cv2.imread(path + r'\gs{}.pgm'.format(num), -1))
    lung = isola_Pulmao2D(img)
    gs = gs > 0.5

    imgs.append(img)
    gss.append(gs)
    lungs.append(lung)

# Pré-processamento ----------------------------------------------------------------------------------------------------
imgRescale = []
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

    imgRescale.append(skimage.exposure.rescale_intensity(img, in_range=(hist[1][lim[0]], hist[1][lim[1]])))

# Gerando Atributos ----------------------------------------------------------------------------------------------------
'''atributo1 = []  # Prewitt
atributo2 = []  # Média
atributo3 = []  # TWC

for img0 in imgRescale:
    xP = scipy.signal.convolve2d(img0, wxPriwitt, 'same')
    yP = scipy.signal.convolve2d(img0, wyPriwitt, 'same')

    # dp = scipy.ndimage.generic_filter(img0, function=np.std, size=9)
    mean = scipy.signal.convolve2d(img0, np.ones((9, 9))*(1/9**2), 'same')
    twc = faz_TWC(img0, 'haar', 5, 0)
    atributo1.append(mean)
    atributo2.append(np.power((np.power(xP, 2) + np.power(yP, 2)), 0.5))
    atributo3.append(twc)'''


# Vetorizando imagens para obtenção dops atributos ---------------------------------------------------------------------
teste = 0
imgTreino = []
gsTreino = []
# a1Treino = []
# a2Treino = []
# a3Treino = []

for i in range(len(imgs)):
    if i != teste:
        imgTreino.append(constroi_vetor_pulmao(imgRescale[i], lungs[i]))
        gsTreino.append(constroi_vetor_pulmao(gss[i], lungs[i]))
        # a1Treino.append(constroi_vetor_pulmao(atributo1[i], lungs[i]))
        # a2Treino.append(constroi_vetor_pulmao(atributo2[i], lungs[i]))
        # a3Treino.append(constroi_vetor_pulmao(atributo3[i], lungs[i]))

    else:
        imgTeste = constroi_vetor_pulmao(imgRescale[i], lungs[i])
        gsTeste = constroi_vetor_pulmao(gss[i], lungs[i])
        # a1Teste = constroi_vetor_pulmao(atributo1[i], lungs[i])
        # a2Teste = constroi_vetor_pulmao(atributo2[i], lungs[i])
        # a3Teste = constroi_vetor_pulmao(atributo3[i], lungs[i])

imgTreino = np.concatenate(imgTreino)
gsTreino = np.concatenate(gsTreino)
# a1Treino = np.concatenate(a1Treino)
# a2Treino = np.concatenate(a2Treino)
# a3Treino = np.concatenate(a3Treino)

# Separação dados de treino e teste ------------------------------------------------------------------------------------
xTreino = np.column_stack((imgTreino, imgTreino))
xTeste = np.column_stack((imgTeste, imgTeste))

yTreino = np.array(gsTreino)
yTeste = np.array(gsTeste)

# Projetando a rede de neuronios ---------------------------------------------------------------------------------------
model = Sequential()
model.add(Dense(units=xTreino.shape[1], activation='relu', input_dim=xTreino.shape[1]))
model.add(Dense(units=1, activation='relu', bias_initializer='ones'))
model.add(Dense(units=1, activation='softmax', bias_initializer='ones'))

opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Treinando a IA -------------------------------------------------------------------------------------------------------
model.fit(xTreino, yTreino, validation_data=(xTeste, yTeste), batch_size=5000, epochs=500)

# Testando a IA --------------------------------------------------------------------------------------------------------
outPrediction = model.predict(xTeste)
outPrediction = outPrediction > 0.5

precisao = accuracy_score(yTeste, outPrediction)
print("\nPrecisão: {}".format(precisao))

# Matriz de confusão ---------------------------------------------------------------------------------------------------
matConfusao = confusion_matrix(yTeste, outPrediction)

# Reconstruir e mostrar imagem -----------------------------------------------------------------------------------------
output = constroi_img_pulmao(list(outPrediction), lungs[teste])
plotar(imgs[teste], output, titles=['Original', 'Output'])
