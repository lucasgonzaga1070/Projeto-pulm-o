from bibNodulos import *
from skimage.morphology import disk
from scipy.ndimage import *
import pywt
from skimage.filters.thresholding import threshold_otsu
from scipy.ndimage import label
from skimage.morphology import binary_dilation, remove_small_holes
import collections as cl
import vedo as vt


def isolarPulmao(imgN):
    hist = skimage.exposure.histogram(imgN)
    peaks, _ = scipy.signal.find_peaks(hist[0][1:], height=(1 / 4) * np.max(hist[0][1:]))

    limD = limE = peaks[0]
    while hist[0][1:][limE] > (1 / 10) * hist[0][1:][peaks[0]]:
        limE -= 1

    while hist[0][1:][limD] > (1 / 10) * hist[0][1:][peaks[0]]:
        limD += 1

    img_pico1 = (imgN > hist[1][1:][limE]) == (imgN < hist[1][1:][limD])

    '''for i in range(img.shape[2]):
        img_pico1[:, :, i] = scipy.ndimage.binary_closing(img_pico1[:, :, i], structure=disk(5))'''

    labels, num = scipy.ndimage.label(img_pico1)

    layer = int(imgN.shape[2] / 2)

    val = labels[int(imgN.shape[0] / 2), int(imgN.shape[1] / 4), layer]

    imgBin = labels == val

    # imgN[imgBin] = 0
    return imgBin


def isola_pulmao_v3(img):
    hist = skimage.exposure.histogram(img)
    hist_Height = hist[0]

    start = np.where(hist_Height[1:] != 0)[0][0]
    zeros = 0
    cont = start

    lim = []
    anterior = hist_Height[start]
    while zeros < 2 and cont < hist_Height.size:
        if hist_Height[cont] != 0:
            atual = hist_Height[cont]
            # print(atual, anterior)
            if hist_Height[start + 1] == hist_Height[1:].max() and zeros < 1:
                # print(1)
                zeros += 1
                lim.append(cont)
            if atual > 5000 and anterior <= 5000 and zeros < 1:
                # print(2)
                zeros += 1
                lim.append(cont)
            if atual < 5000 and anterior > 5000 and zeros >= 1:
                # print(3)
                zeros += 1
                lim.append(cont)
            anterior = atual
        cont += 1

    img = (img > lim[0]) & (img < lim[1])
    labels = scipy.ndimage.label(img)

    val = labels[0][256, 170, int(img.shape[2]/2)]
    out = labels[0] == val

    out = skimage.morphology.binary_closing(out, skimage.morphology.ball(10))
    out = out * 65535
    out = out.astype(np.uint16)

    return out


img, gs, nod = create_GoldSTD('LIDC-IDRI-0015')
lung = isola_pulmao_v3(img)

'''vp = vt.Plotter(N=1)
vp.show(vt.Volume(lung))'''