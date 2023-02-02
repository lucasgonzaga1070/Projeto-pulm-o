from geradorAtibutos import *

path = r'C:\Users\lgxnt\Imagens\imgIC'

imgs = []
gss = []
lungs = []

rescales = []
limites = []
medias = []

for i in range(1, 30):
    img = skimage.img_as_float(cv2.imread(path + r'\img{}.pgm'.format(i), -1))
    gs = skimage.img_as_float(cv2.imread(path + r'\gs{}.pgm'.format(i), -1))

    '''plt.imshow(gs, cmap='gray')
    plt.show()'''

    lung = isola_Pulmao2D(img)

    imgLung = img.copy()
    imgLung[~lung] = 0.0

    hist = skimage.exposure.histogram(imgLung)
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

    imgRescale = skimage.exposure.rescale_intensity(imgLung, in_range=(hist[1][lim[0]], hist[1][lim[1]]))
    Cmin, Lmin, dC, dL = cv2.selectROI(windowName='window', img=img, showCrosshair=True)
    cv2.destroyAllWindows()

    print(imgRescale[Lmin, Cmin])

    '''plt.imshow(imgRescale, cmap='gray')
    plt.show()'''
    hist2 = skimage.exposure.histogram(imgRescale)
    '''plt.stem(hist2[1][1:-1], hist2[0][1:-1])
    plt.show()'''

    cont = 253
    zeros = 0

    lim2 = []
    while zeros < 2 and cont > 0:
        if hist2[0][cont] > 150 and hist2[0][cont + 1] <= 150 and zeros < 1:
            zeros += 1
            lim2.append(cont)
            # print(hist2[1][cont])
        elif hist2[0][cont] < 70 and hist2[0][cont + 1] >= 70 and zeros >= 1:
            zeros += 1
            lim2.append(cont)
            # print(hist2[1][cont])

        cont -= 1

    histC = np.zeros_like(hist2[0])
    histC[lim2[1]: lim2[0]] = hist2[0][lim2[1]: lim2[0]]

    '''plt.stem(hist2[1], histC)
    plt.show()'''

    # medias.append(np.max(hist2[1][histC != 0]))
    media = np.mean(hist2[1][histC != 0])
    medias.append(media)

    limE = abs(media - hist2[1][np.where(histC != 0)[0][0]])
    limD = abs(media - hist2[1][np.where(histC != 0)[0][-1]])

    limites.append(max(limE, limD))

print(str(np.mean(medias) - np.mean(limites)) + ' ± ' + str(np.std(medias)))
