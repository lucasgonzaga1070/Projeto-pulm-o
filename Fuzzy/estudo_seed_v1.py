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


# Funções --------------------------------------------------------------------------------------------------------------
class Layer_selector(tkinter.Frame):
    def __init__(self, img, master):
        super().__init__(master)
        self.master = master
        self.place()

        self.layer = 0
        self.img_size = img.shape
        self.img_class = img
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

        self.im = plt.imshow(img[:, :, int(self.layer)], cmap='gray')

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


img, gs, nod = create_GoldSTD('LIDC-IDRI-0002')
root = tkinter.Tk()
root.geometry('700x700')
root.title('Projeto Imagens Biomédicas 2021')
app = Layer_selector(img=img, master=root)
app.mainloop()

print('oi')

# Importando imagens como uint16 e transformando em float --------------------------------------------------------------
'''img, gs, nod = create_GoldSTD('LIDC-IDRI-0001')
M, N, O = img.shape
lung = isola_pulmao_v2(img)

img = img / img.max()
img = img.astype(float)

lung = lung / lung.max()
lung = lung.astype(float)
lung = lung > 0.5

gs = gs / gs.max()
gs = gs.astype(float)

sliceNod = sliceWITHnod(gs)

# Ajuste de intesidade para melhorar contraste -------------------------------------------------------------------------
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

Cmin, Lmin, dC, dL = cv2.selectROI(windowName='window', img=img[:, :, sliceNod], showCrosshair=True)
cv2.destroyAllWindows()

# Avaliando quadrantes da imagem ---------------------------------------------------------------------------------------
img = img[:, :, sliceNod]
img[~lung[:, :, sliceNod]] = 0
seed_Atual = [int(M/2), int(N/2)]
cont = 0

while cont < 250:

    esquerda = img[cont:M - cont, cont:seed_Atual[1]]
    direita = img[cont:M - cont, seed_Atual[1]:N - cont]

    cima = img[cont:seed_Atual[0], cont:N - cont]
    baixo = img[seed_Atual[0]:M - cont, cont:N - cont]

    horizontal = np.argmax([esquerda.mean(), direita.mean()])
    vertical = np.argmax([cima.mean(), baixo.mean()])
    print(seed_Atual, cont, [esquerda.mean(), direita.mean(), cima.mean(), baixo.mean()])

    if abs(esquerda.mean() - direita.mean()) < 0.001 or abs(cima.mean() - baixo.mean()) < 0.001:
        break

    if horizontal == 0:
        seed_Atual[1] -= 1
    else:
        seed_Atual[1] += 1

    if vertical == 0:
        seed_Atual[0] -= 1
    else:
        seed_Atual[0] += 1

    cont += 1

plt.imshow(img, cmap='gray')
plt.show()'''
