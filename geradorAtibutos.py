from bibNodulos import *
from skimage.morphology import disk
from scipy.ndimage import *
import pywt
from skimage.filters.thresholding import threshold_otsu
from scipy.ndimage import label
from skimage.morphology import binary_dilation, remove_small_holes
import collections as cl

path = r'C:\Users\lgxnt\Imagens\imgIC'

img = skimage.img_as_float(cv2.imread(path + r'\img1.pgm', 0))
gs = skimage.img_as_float(cv2.imread(path + r'\gs1.pgm', 0))
img = skimage.exposure.rescale_intensity(img, in_range=(0, img.max()))
gs = gs > 0.5


def isola_Pulmao2D(img):
    imgC = img.copy()

    hist = skimage.exposure.histogram(imgC)
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

    # print(hist[0])
    imgR = skimage.exposure.rescale_intensity(img, in_range=(hist[1][lim[0]], hist[1][lim[1]]))

    thresh = threshold_otsu(imgR)
    imgR = imgR < thresh
    imgR = binary_dilation(imgR, disk(6))
    labels = label(imgR)
    val = [item for item, count in cl.Counter(labels[0][250:300, 150:350].flatten()).items() if count > 1 and item != 0]
    # print(len(val))
    bins = [labels[0] == x for x in val]
    imgBin = np.sum(bins, axis=0, dtype=bool)
    imgBin = binary_dilation(imgBin, disk(20))
    return imgBin


def faz_TWC(img, wlt, level, decomp, mode=0):
    wavelet = pywt.swt2(img, wlt, level=level)
    # [(cA, (cDv, cDh, cDd))]
    if not mode:
        imgOutput = wavelet[decomp][1][2] + wavelet[decomp][1][0] + wavelet[decomp][1][1]
    else:
        imgOutput = wavelet[decomp][0] + wavelet[decomp][1][0] + wavelet[decomp][1][1]
    imgOutput = imgOutput / np.max(imgOutput)
    imgOutput = skimage.exposure.rescale_intensity(imgOutput, in_range=(0, 1))

    return imgOutput


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


def plot_img_interactive(*listas):
    root = tkinter.Tk()
    root.wm_title("Plot Interativo")

    pos = tkinter.DoubleVar()

    fig, axis = plt.subplots(1, len(listas))

    for x in range(len(listas)):
        axis[x].imshow(listas[x][0], cmap='gray')
        axis[x].axis('off')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=5)

    def _quit():
        root.quit()
        root.destroy()

    def _next():
        if -1 < pos.get() < len(listas[0]) - 1:
            i = int(pos.get()) + 1
            for x in range(len(listas)):
                axis[x].imshow(listas[x][i], cmap='gray')
                axis[x].axis('off')
            pos.set(i)
            canvas.draw()
            label.config(text="img {}".format(i))
        else:
            print("Erro: Alcançou a primeira layer.")

    def _previous():
        if 0 < pos.get() < len(img):
            i = int(pos.get()) - 1
            for x in range(len(listas)):
                axis[x].imshow(listas[x][i], cmap='gray')
                axis[x].axis('off')
            pos.set(i)
            canvas.draw()
            label.config(text="img {}".format(i))
        else:
            print("Erro: Alcançou a primeira layer.")

    button = tkinter.Button(master=root, text="Quit", command=_quit)
    button.pack(side=tkinter.BOTTOM)

    next_bt = tkinter.Button(master=root, text="Next", command=_next)
    next_bt.pack(side=tkinter.RIGHT)

    previous_bt = tkinter.Button(master=root, text="Previous", command=_previous)
    previous_bt.pack(side=tkinter.LEFT)

    label = tkinter.Label(root, text="img 0")
    label.pack(side=tkinter.TOP)

    tkinter.mainloop()


def fazerMascaraGauss2D(media, desvio):
    x = np.arange(media * 2 + 1)
    g = np.e ** ((-1 / 2) * ((x - media) / desvio) ** 2)

    g1 = np.zeros((media * 2 + 1, media * 2 + 1), float)
    g1[media, :] = g
    w_Gauss2D = scipy.signal.convolve2d(g1, g1.transpose(), 'same')
    w_Gauss2D_N = w_Gauss2D / np.sum(w_Gauss2D)

    return w_Gauss2D_N