from matplotlib import use
use('Agg')
use('TkAgg')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import pylidc
import skimage.exposure
from bibNodulos import create_GoldSTD, sliceWITHnod, isola_pulmao_v2, slider_3DImage
import vedo
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from skimage.morphology import *
from matplotlib.backend_bases import key_press_handler
from matplotlib.widgets import EllipseSelector, RectangleSelector


# Classe ---------------------------------------------------------------------------------------------------------------


class Layer_selector(tkinter.Frame):
    def __init__(self, ig, master):
        super().__init__(master)
        self.master = master
        self.place()

        self.layer = 0
        self.fig = None
        self.ax = None
        self.img_size = ig.shape
        self.img_class = ig
        self.ROI = None
        self.switch_variable = tkinter.BooleanVar(self.master, value=True)

        self.slider = tkinter.Scale(self.master, command=self._slide, from_=0, to=self.img_size[2] - 1,
                                    orient=tkinter.HORIZONTAL, length=399)
        self.slider.pack(side=tkinter.BOTTOM)

        self.switch = tkinter.Checkbutton(self.master, text="Select ROI", variable=self.switch_variable,
                                          indicatoron=False, onvalue=False, offvalue=True, width=8, command=self.toggle_selector)

        self.switch.pack(side=tkinter.BOTTOM)

        self.quit_buttom = tkinter.Button(self.master, text="Quit", command=self.close_window)
        self.quit_buttom.pack(side=tkinter.BOTTOM)

        self.im = None
        self.canvas = None
        self.create_canvas()

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tkinter.TOP, fill=tkinter.X)

        self.selector = RectangleSelector(self.ax, self.select_callback,
                                          useblit=True,
                                          button=[1, 3],  # disable middle button
                                          minspanx=5, minspany=5,
                                          spancoords='pixels',
                                          interactive=True)
        self.selector.set_active(False)

        self.slider.set(self.layer)

        # self.close_window()

    def select_callback(self, eclick, erelease):
        if eclick.button == 1:
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            self.ROI = (x1, x2, y1, y2)
            # print(self.ROI)

    def create_canvas(self):
        self.fig, self.ax = plt.subplots(1, 1)
        plt.axis('off')

        self.im = self.ax.imshow(img[:, :, int(self.layer)], cmap='gray')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.BOTTOM, expand=True, fill=tkinter.BOTH)

    def _slide(self, value):
        self.im.set_data(self.img_class[:, :, self.slider.get()])
        self.canvas.draw()
        self.layer = self.slider.get()

    def toggle_selector(self):
        # self.switch_variable.set(~self.switch_variable.get())
        # print(self.switch_variable.get())
        if self.switch_variable.get():
            self.selector.set_active(False)
        else:
            self.selector.set_active(True)

    def close_window(self):
        self.master.quit()
        self.master.destroy()


# Funções --------------------------------------------------------------------------------------------------------------
def afinidade(seed, pixel, h, it, mode=0, Adj=1):
    hom = np.exp((-0.5) * (((np.abs(seed - pixel) - h[0]) / h[1]) ** 2))
    inte = np.exp((-0.5) * (((0.5 * (seed + pixel) - it[0]) / it[1]) ** 2))

    if mode == 0:
        wi = 0.5
        wh = 1 - wi
        return Adj * (wh * hom + wi * inte)
    if mode == 1:
        wi = inte / (hom + inte)
        wh = 1 - wi
        return Adj * (wh * hom + wi * inte)


def avaliarSegmentacao(objSegmentado, goldSTD):
    intersec = objSegmentado * goldSTD
    area_intersec = intersec.sum()
    area_goldStandart = goldSTD.sum()
    area_objSegmentado = objSegmentado.sum()
    tam_imagem = goldSTD.size

    vp = (area_intersec / area_goldStandart) * 100
    fp = ((area_objSegmentado - area_intersec) / (tam_imagem - area_goldStandart)) * 100
    fn = ((area_goldStandart - area_intersec) / area_goldStandart) * 100
    od = (200 * vp) / (2 * vp + fn + fp)
    Or = (100 * vp) / (vp + fp + fn)

    return [vp, fp, fn, od, Or]


def fuzzy_conectdness(mediaH, stdH, mediaI, stdI):
    while len(fila) != 0 and fila[0][1] > 0.5 and exSeeds.sum() < 10 * (Cmax - Cmin) * (Lmax - Lmin) ** 2:
        fila = sorted(fila, key=lambda pixel: pixel[1], reverse=True)
        seedAtual = fila.pop(0)
        # print(len(fila))
        if cont == 0:
            anterior = seedAtual[1]
            cont += 1
        else:
            if anterior != seedAtual[1]:
                anterior = seedAtual[1]
                cont = 1
            elif anterior == seedAtual[1]:
                cont += 1
        if cont > 20e3:
            cont = 1
            pass
        seedAtual = seedAtual[0]
        pos = [(seedAtual[0] - 1, seedAtual[1], seedAtual[2]), (seedAtual[0] + 1, seedAtual[1], seedAtual[2]),
               (seedAtual[0], seedAtual[1] + 1, seedAtual[2]), (seedAtual[0], seedAtual[1] - 1, seedAtual[2]),
               (seedAtual[0] - 1, seedAtual[1], seedAtual[2] + 1), (seedAtual[0] + 1, seedAtual[1], seedAtual[2] + 1),
               (seedAtual[0], seedAtual[1] + 1, seedAtual[2] + 1), (seedAtual[0], seedAtual[1] - 1, seedAtual[2] + 1),
               (seedAtual[0] - 1, seedAtual[1], seedAtual[2] - 1), (seedAtual[0] + 1, seedAtual[1], seedAtual[2] - 1),
               (seedAtual[0], seedAtual[1] + 1, seedAtual[2] - 1), (seedAtual[0], seedAtual[1] - 1, seedAtual[2] - 1)]
        for i in pos:
            if (i[0] >= M) or (i[1] >= N) or (i[2] >= O) or (i[0] < 0) or (i[1] < 0) or (i[2] < 0):
                continue
            if exSeeds[i] != 1 and lung[i] == 1:
                exSeeds[i] = 1
                # print(np.sum(exSeeds))
                ua = afinidade(img[seedAtual], img[i], (mediaH, stdH), (mediaI, stdI), mode=1)
                afinit[i] = ua
                uk = np.min([afinit[i], connect[seedAtual]])
                pathConnect[i] = uk
                mica = np.max([connect[i], pathConnect[i]])
                connect[i] = mica
                # print("ua: {} / uk: {} / mica: {}".format(ua, uk, mica))
                aux = (i, mica)
                fila.append(aux)


# Importando imagens como uint16 e transformando em float --------------------------------------------------------------
print("----------------------------Importando imagens como uint16 e transformando em float----------------------------")
img, gs, nod = create_GoldSTD('LIDC-IDRI-0001')
M, N, O = img.shape

img = img / img.max()
img = img.astype(float)

gs = gs / gs.max()
gs = gs.astype(float)

gs_slice = gs.sum(axis=(0, 1)).argmax()
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img[:, :, gs_slice], cmap='gray')
ax[1].imshow(gs[:, :, gs_slice], cmap='gray')
plt.title(f"Slice: {gs_slice}")
plt.show()

# Selecionando região de interesse -------------------------------------------------------------------------------------
print("---------------------------------------Selecionando região de interesse-----------------------------------------")
root = tkinter.Tk()
root.geometry('700x700')
root.title('Projeto Imagens Biomédicas 2021')
app = Layer_selector(ig=img.copy(), master=root)
app.mainloop()
print("Exited!")

# Pré segmentação do pulmão --------------------------------------------------------------------------------------------
print("----------------------------------------Pré segmentação do pulmão-----------------------------------------------")
lung = isola_pulmao_v2(img)
lung = lung / lung.max()
lung = lung.astype(float)

# Criando matrizes auxiliares (conectividade local e global, afinidade e ex seeds) -------------------------------------
print("----------------Criando matrizes auxiliares (conectividade local e global, afinidade e ex seeds)----------------")
connect = np.zeros_like(img)
afinit = np.zeros_like(img)
pathConnect = np.zeros_like(img)
exSeeds = np.zeros_like(img)

# Ajuste de intesidade para melhorar contraste -------------------------------------------------------------------------
print("----------------------------------Ajuste de intesidade para melhorar contraste----------------------------------")
hist = skimage.exposure.histogram(img)
# plt.stem(hist[1][1:], hist[0][1:])
# plt.show()
zeros = 0
cont = 2

lim = []
while zeros < 2 and cont < hist[0].size:
    if hist[0][cont] > 5000 and hist[0][cont - 1] <= 5000 and zeros < 1:
        zeros += 1
        lim.append(cont)
    elif hist[0][cont] < 5000 and hist[0][cont - 1] >= 5000 and zeros >= 1:
        zeros += 1
        lim.append(cont)
    cont += 1

img = skimage.exposure.rescale_intensity(img, in_range=(hist[1][lim[0]], hist[1][lim[1]]))
# hist = skimage.exposure.histogram(img)
# plt.stem(hist[1][1:], hist[0][1:])
# plt.show()

# Selecionando região de interesse -------------------------------------------------------------------------------------
print("----------------------------------------Selecionando região de interesse-----------------------------------------")
sliceNod = app.layer
Cmin, Cmax, Lmin, Lmax = app.ROI
# plt.figure()
# plt.imshow(img[Lmin:Lmax, Cmin:Cmax, sliceNod])
# plt.show()
C = Cmin + int((Cmax - Cmin) / 2)
L = Lmin + int((Lmax - Lmin) / 2)

seed0 = img[L, C, sliceNod]

connect[L, C, sliceNod] = 1
afinit[L, C, sliceNod] = 1
