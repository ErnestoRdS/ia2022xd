import matplotlib.pyplot as plt
import cv2
import numpy as np

# Distancia Euclídea
def distEuc(p, q):
    return np.sqrt(np.sum(np.square(p - q)))


# Generar centroides
def genCen(K, data):
    cens = []
    for i in range(K):
        cen_i = data[np.random.randint(len(data))]
        cens.append(cen_i)
    cens = np.array(cens)
    return cens


# Actualizar centroides
def updateCens(K, data, labels):
    cens = []
    for i in range(K):
        cen_i = np.mean(data[labels == i, :], axis=0)
        cens.append(cen_i)
    cens = np.array(cens)
    return cens


# K-MEANS
def kmeans(x, cens):
    distances = np.zeros(len(cens))
    for id, cen in enumerate(cens):
        distances[id] = distEuc(x, cen)
    return np.argmin(distances)


##################################################
if __name__ == "__main__":
    # NÚMERO DE GRUPOS
    K = 3
    # CARGAR DATOS
    img = cv2.imread("testimgsxd/pear.jpeg")
    re2 = img.shape[2]
    data = img.reshape(-1, re2)  # Imagen reshapeada para manejarla más fácil
    # GENERAR CENTROIDES
    cens = genCen(K, np.unique(data, axis=0))

    labels = np.zeros(len(data), dtype=int)  # <<<Para guardar etiquetas
    # Primera clusterización
    for id, x in enumerate(data):
        labels[id] = kmeans(x, cens)
    newCens = updateCens(K, data, labels)  # <<<Calcular nuevos centroides

    # Para iterar (por si acasoxd)
    iters = 10
    t = 0

    # REPETIR
    while not (newCens == cens).all() and (t in range(iters)):
        # Loopear otra vez la imagen
        for id, x in enumerate(data):
            labels[id] = kmeans(x, newCens)
        # Actualizar centroides
        cens = newCens.copy()
        newCens = updateCens(K, data, labels)
        t += 1  # Aumentar

    # PARA COLOREAR
    newCens = updateCens(K, data, labels)
    colorsArray = (
        []
    )  # np.zeros(len(data), dtype=int) #<<<Para guardar los valores de colores

    for i in range(len(labels)):
        colorsArray.append(newCens[labels[i]])

    colorsArray = np.array(colorsArray)
    colorsArray = colorsArray.astype(int)
    colorsArray = np.flip(colorsArray, axis=1)
    colorsArray = np.reshape(colorsArray, img.shape)

    # MOSTRARxd
    img_rgb = cv2.merge([colorsArray])
    plt.imshow(img_rgb)
    plt.show()
