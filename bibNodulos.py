import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter
import scipy.signal
import scipy.ndimage
import pylidc as pl
from pylidc.utils import consensus
import skimage
import skimage.exposure
import cv2
from skimage.morphology import disk, rectangle
import vedo as vtk


def create_GoldSTD(file):
    # Query for a scan, and convert it to an array volume.
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == file).first()
    vol = scan.to_volume(verbose=False)

    if vol.min() < 0:
        vol = (vol - vol.min())/(vol.max() - vol.min())
    else:
        vol = (vol + vol.min()) / (vol.max() + vol.min())

    vol = vol*65535
    vol = vol.astype(np.uint16)
    boolean = np.zeros_like(vol)

    # Cluster the annotations for the scan, and grab one.
    nods = scan.cluster_annotations()

    for nod in nods:
        cmask, cbbox, masks = consensus(nod, clevel=0.5, pad=[(20, 20), (20, 20), (0, 0)])

        boolean[cbbox] = cmask*65535

    return vol, boolean, nods


def slider_3DImage(img, img2=None, view=1):
    M, N, O = img.shape
    global out
    out = 0
    root = tkinter.Tk()
    root.wm_title("Plot Interativo")

    fig = plt.figure(figsize=(8, 4), dpi=160)

    pos = tkinter.DoubleVar()
    layerInicial = 0
    if view == 1:
        while img[:, :, layerInicial].max() <= 0 and layerInicial < img.shape[2]:
            layerInicial += 1
        plt.title("Imagem 1")
        plt.axis('off')
        im = plt.imshow(img[:, :, layerInicial], cmap='gray')

    if view == 2:
        while ((img[:, :, layerInicial].max() <= 0) or (img2[:, :, layerInicial].max() <= 0)) and layerInicial < img.shape[2] - 1:
            layerInicial += 1

        plt.subplot(121)
        plt.title("Imagem 1")
        plt.axis('off')
        im = plt.imshow(img[:, :, layerInicial], cmap='gray')
        plt.subplot(122)
        plt.title("Imagem 2")
        plt.axis('off')
        im2 = plt.imshow(img2[:, :, layerInicial], cmap='gray')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

    def _quit():
        root.quit()
        root.destroy()

    def _slide():
        global out
        i = int(slider.get())
        out = i
        im.set_data(img[:, :, i])
        if view == 2:
            im2.set_data(img2[:, :, i])
        canvas.draw()
        label.config(text="Layer {}".format(i))

    def _next():
        global out
        if -1 < pos.get() < O - 1:
            i = int(pos.get()) + 1
            out = i
            im.set_data(img[:, :, i])
            if view == 2:
                im2.set_data(img2[:, :, i])
            pos.set(i)
            canvas.draw()
            label.config(text="Layer {}".format(i))
        else:
            print("Erro: Alcançou última layer possível.")

    def _previous():
        global out
        if 0 < pos.get() < O:
            i = int(pos.get()) - 1
            out = i
            im.set_data(img[:, :, i])
            if view == 2:
                im2.set_data(img2[:, :, i])
            pos.set(i)
            canvas.draw()
            label.config(text="Layer {}".format(i))
        else:
            print("Erro: Alcançou a primeira layer.")

    next_bt = tkinter.Button(master=root, text="Next", command=_next)
    next_bt.pack(side=tkinter.RIGHT)

    previous_bt = tkinter.Button(master=root, text="Previous", command=_previous)
    previous_bt.pack(side=tkinter.LEFT)

    button = tkinter.Button(master=root, text="Quit", command=_quit)
    button.pack(side=tkinter.BOTTOM)

    label = tkinter.Label(root, text="Layer 0")
    label.pack(side=tkinter.TOP)

    slider = tkinter.Scale(root, variable=pos, command=_slide, from_=0, to=O - 1, orient=tkinter.HORIZONTAL)
    slider.set(layerInicial)
    slider.pack(anchor=tkinter.CENTER)

    tkinter.mainloop()
    return out


def isolarPulmao(imgN):
    img = imgN.copy()
    hist = skimage.exposure.histogram(imgN)
    plt.stem(hist[1][1:], hist[0][1:])
    peaks, _ = scipy.signal.find_peaks(hist[0][1:], height=(1 / 4) * np.max(hist[0][1:]))

    limD = limE = peaks[0]
    while hist[0][1:][limE] > (1 / 16) * hist[0][1:][peaks[0]]:
        limE -= 1

    while hist[0][1:][limD] > (1 / 16) * hist[0][1:][peaks[0]]:
        limD += 1

    img_pico1 = (imgN > hist[1][1:][limE]) == (imgN < hist[1][1:][limD])
    print(limD, limE)
    '''for i in range(imgN.shape[2]):
        img_pico1[:, :, i] = scipy.ndimage.binary_dilation(img_pico1[:, :, i], structure=disk(10))'''

    labels, num = scipy.ndimage.label(img_pico1)

    layer = int(imgN.shape[2] / 2)
    val = labels[int(imgN.shape[0] / 2), int(imgN.shape[1] / 4), layer]

    imgBin = labels == val

    imgBin2 = np.logical_not(imgBin)
    img[imgBin2] = 0
    imgRescale = skimage.exposure.rescale_intensity(img, in_range=(hist[1][1:][limE], hist[1][1:][limD]))
    return imgBin


def atributoHighPass(img):
    imgHighPass = np.zeros_like(img)

    for i in range(img.shape[2]):
        imgFiltro = scipy.ndimage.gaussian_filter(img[:, :, i], 7)
        imgOutput = img[:, :, i] - imgFiltro
        imgOutput = imgOutput / np.max(imgOutput)
        imgHighPass[:, :, i] = imgOutput

    return imgHighPass


def sliceWITHnod(gs):
    slice = []
    for i in range(gs.shape[2]):
        slice.append(np.sum(gs[:, :, i]))

    return slice.index(max(slice))


def isola_pulmao_v2(img):
    index_meio_pulmao = int(img.shape[2]/2)
    lim = skimage.filters.threshold_multiotsu(img)

    img = (img > lim[0]) & (img < lim[1])
    labels = scipy.ndimage.label(img)

    val = labels[0][256, 128, index_meio_pulmao]
    out = labels[0] == val

    out = skimage.morphology.binary_closing(out, skimage.morphology.ball(10))
    out = out * 65535
    out = out.astype(np.uint16)

    return out
