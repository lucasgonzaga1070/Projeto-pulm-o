from bibNodulos import *

q = 1
for file in ['LIDC-IDRI-{:04d}'.format(x) for x in range(1, 30)]:
    img, gs, nod = create_GoldSTD(file)
    if nod:
        imgSave = img[:, :, sliceWITHnod(gs)]
        imgSave = imgSave / imgSave.max()
        imgSave = imgSave * 65535
        imgSave = imgSave.astype(np.uint16)

        gsSave = gs[:, :, sliceWITHnod(gs)]

        lung = isola_pulmao_v2(img)
        lungSave = lung[:, :, sliceWITHnod(gs)]

        if imgSave.max() <= 1.0:
            print('Menor')

        if lungSave.sum() > lungSave.shape[0] * lungSave.shape[1]:
            print(file)
            cv2.imwrite('img{}.pgm'.format(q), imgSave)
            cv2.imwrite('gs{}.pgm'.format(q), gsSave)
            cv2.imwrite('lungs{}.pgm'.format(q), lungSave)
        q += 1
    else:
        print('Vazio')

