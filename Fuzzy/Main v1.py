import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
import pylidc
import skimage.exposure
from bibNodulos import create_GoldSTD, sliceWITHnod, isola_pulmao_v2, slider_3DImage
import vedo
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.morphology import *


class Layer_slide(tkinter.Frame):
    def __init__(self, ig, ig2, master):
        super().__init__(master)
        self.master = master
        self.place()

        self.layer = 89
        self.img_size = ig.shape
        self.img_class = ig

        self.img2_size = ig2.shape
        self.img2_class = ig2

        self.im = None
        self.im2 = None

        self.canvas = None
        self.create_canvas()

        self.slider = tkinter.Scale(self.master, command=self._slide, from_=0, to=self.img_size[2] - 1,
                                    orient=tkinter.HORIZONTAL, length=399)
        self.slider.set(self.layer)
        self.slider.pack(side=tkinter.TOP)

    def create_canvas(self):
        fig, axs = plt.subplots(1, 2)
        plt.axis('off')

        self.im = axs[0].imshow(self.img_class[:, :, int(self.layer)], cmap='gray')
        self.im2 = axs[1].imshow(self.img2_class[:, :, int(self.layer)], cmap='gray')

        self.canvas = FigureCanvasTkAgg(fig, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.BOTTOM, expand=True, fill=tkinter.BOTH)

    def _slide(self, value):
        self.im.set_data(self.img_class[:, :, self.slider.get()])
        self.im2.set_data(self.img2_class[:, :, self.slider.get()])
        self.canvas.draw()
        self.layer = self.slider.get()


# Classe ---------------------------------------------------------------------------------------------------------------

class Layer_selector(tkinter.Frame):
    def __init__(self, ig, master):
        super().__init__(master)
        self.master = master
        self.place()

        self.layer = 0
        self.img_size = ig.shape
        self.img_class = ig
        self.ROI = None

        self.im = None
        self.canvas = None
        self.create_canvas()

        self.slider = tkinter.Scale(self.master, command=self._slide, from_=0, to=self.img_size[2] - 1,
                                    orient=tkinter.HORIZONTAL, length=399)
        self.slider.set(self.layer)
        self.slider.pack(side=tkinter.TOP)

        self.close_window()

    def create_canvas(self):
        fig = plt.figure()
        plt.axis('off')

        self.im = plt.imshow(self.img_class[:, :, int(self.layer)], cmap='gray')

        self.canvas = FigureCanvasTkAgg(fig, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.BOTTOM, expand=True, fill=tkinter.BOTH)

    def _slide(self, value):
        self.im.set_data(self.img_class[:, :, self.slider.get()])
        self.canvas.draw()
        self.layer = self.slider.get()

    def slct_ROI(self, value):
        self.ROI = cv2.selectROI('window', self.img_class[:, :, self.layer])
        cv2.destroyAllWindows()

    def close_window(self):
        self.slider.bind("<ButtonRelease-1>", self.slct_ROI)
        self.master.bind("<Return>", lambda e: self.master.destroy())


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


# Importando imagens como uint16 e transformando em float --------------------------------------------------------------
img, gs, nod = create_GoldSTD('LIDC-IDRI-0001')
M, N, O = img.shape
lung = isola_pulmao_v2(img)

img = img / img.max()
img = img.astype(float)

lung = lung / lung.max()
lung = lung.astype(float)

gs = gs / gs.max()
gs = gs.astype(float)

gs_slice = gs.sum(axis=(0, 1)).argmax()
plt.imshow(gs[:, :, gs_slice], cmap='gray')
plt.axis('off')
plt.title(f"GS {1} - slice {gs_slice}")
plt.show()

# Selecionando região de interesse -------------------------------------------------------------------------------------
root = tkinter.Tk()
root.geometry('700x700')
root.title('Projeto Imagens Biomédicas 2021')
app = Layer_selector(ig=img.copy(), master=root)
app.mainloop()

# Criando matrizes auxiliares (conectividade local e global, afinidade e ex seeds) -------------------------------------
connect = np.zeros_like(img)
afinit = np.zeros_like(img)
pathConnect = np.zeros_like(img)
exSeeds = np.zeros_like(img)

# Ajuste de intesidade para melhorar contraste -------------------------------------------------------------------------
hist = skimage.exposure.histogram(img)
plt.imshow(img[:, :, 66])
plt.show()
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
hist = skimage.exposure.histogram(img)
plt.imshow(img[:, :, 66])
plt.show()

# Selecionando região de interesse -------------------------------------------------------------------------------------
sliceNod = app.layer
Cmin, Lmin, dC, dL = app.ROI
print(sliceNod)

C = Cmin + int(dC / 2)
L = Lmin + int(dL / 2)

seed0 = img[L, C, sliceNod]

connect[L, C, sliceNod] = 1
afinit[L, C, sliceNod] = 1

# Inicio da fila e calculo dos valores de media e desvio da intesidade e homogeneidade ---------------------------------
fila = [((L, C, sliceNod), connect[L, C, sliceNod])]

regiao = img[Lmin:Lmin + dL, Cmin:Cmin + dC, sliceNod]

I = abs(regiao + np.ones_like(regiao) * seed0)
H = abs(np.ones_like(regiao) * seed0 - regiao)
mediaH = np.mean(H)
stdH = np.std(H) + 1e-9
mediaI = np.mean(0.5 * I)
stdI = np.std(0.5 * I) + 1e-9
cont = 0

# Iterações ------------------------------------------------------------------------------------------------------------
while len(fila) != 0 and fila[0][1] > 0.5:

    fila = sorted(fila, key=lambda pixel: pixel[1], reverse=True)

    seedAtual = fila.pop(0)
    # print(seedAtual, cont)

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

    pos = [(seedAtual[0] + round(np.sin(np.pi*x/180)), seedAtual[1] + round(np.cos(np.pi*x/180)), seedAtual[2] + y) for x in range(0, 360, 45) for y in (0, -1, 1)] + \
          [(seedAtual[0], seedAtual[1], seedAtual[2] - 1), (seedAtual[0], seedAtual[1], seedAtual[2] + 1)]

    #  pos = [(seedAtual[0] - 1, seedAtual[1], seedAtual[2]), (seedAtual[0] + 1, seedAtual[1], seedAtual[2]),
    #       (seedAtual[0], seedAtual[1] + 1, seedAtual[2]), (seedAtual[0], seedAtual[1] - 1, seedAtual[2]),
    #       (seedAtual[0] - 1, seedAtual[1], seedAtual[2] + 1), (seedAtual[0] + 1, seedAtual[1], seedAtual[2] + 1),
    #       (seedAtual[0], seedAtual[1] + 1, seedAtual[2] + 1), (seedAtual[0], seedAtual[1] - 1, seedAtual[2] + 1),
    #       (seedAtual[0] - 1, seedAtual[1], seedAtual[2] - 1), (seedAtual[0] + 1, seedAtual[1], seedAtual[2] - 1),
    #       (seedAtual[0], seedAtual[1] + 1, seedAtual[2] - 1), (seedAtual[0], seedAtual[1] - 1, seedAtual[2] - 1)]

    for i in pos:
        print(len(fila), img[i])
        if (i[0] >= M) or (i[1] >= N) or (i[2] >= O) or (i[0] < 0) or (i[1] < 0) or (i[2] < 0) or (img[i] < 0.15):
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

# Pós-processamento ----------------------------------------------------------------------------------------------------
imgBin = connect > 0.5
imgBin = binary_closing(imgBin, ball(10))

# Plots e avaliação ----------------------------------------------------------------------------------------------------
plt.imshow(img[:, :, sliceNod], cmap='gray')
plt.axis('off')
plt.title("Original")
plt.show()

plt.imshow(connect[:, :, sliceNod], cmap='gray')
plt.axis('off')
plt.title("Conectividade")
plt.show()

plt.imshow(afinit[:, :, sliceNod], cmap='gray')
plt.axis('off')
plt.title("Afinidade")
plt.show()

plt.imshow(pathConnect[:, :, sliceNod], cmap='gray')
plt.axis('off')
plt.title("Conectividade Local")
plt.show()

plt.imshow(imgBin[:, :, sliceNod], cmap='gray')
plt.axis('off')
plt.title("Objeto Segmentado")
plt.show()

plt.imshow(gs[:, :, sliceNod], cmap='gray')
plt.axis('off')
plt.title("Gold Standart")
plt.show()

vp = vedo.Plotter(N=1)
vp.show(vedo.Volume(connect > 0.2))


[VP, FP, FN, OD, oR] = avaliarSegmentacao(imgBin, gs)
print("Verdadeiro Positivo: {:.3f}%\nFalso Positivo: {:.3f}%\nFalso Negativo: {:.3f}%\nOverlap Dice: {:.3f}%\nOverlap "
      "Ratio: {:.3f}%".format(VP, FP, FN, OD, oR))

print("Falso positivo ta errado. Usa o tamanho da imagem pro calculo. Mas a imagem é muito maior que o objeto que "
      "queremos segmentar.")

'''for i in range(gs.shape[2]):
    if np.sum(gs[:, :, i] != 0):
        plt.imshow(imgBin[:, :, i], cmap='gray')
        plt.title("Bin {}".format(i))
        plt.show()

        plt.imshow(gs[:, :, i], cmap='gray')
        plt.title("GS {}".format(i))
        plt.show()
        print(avaliarSegmentacao(imgBin[:, :, i], gs[:, :, i]))
'''