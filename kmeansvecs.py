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


# K-Means
def kmeansVectorizado(data, cens):
    distances = []
    for cen in cens:
        # distances.append((np.sum((data - cen) ** 2, axis=1)) ** (1 / 2))
        distances.append(np.linalg.norm(data - cen, axis=1))
    distances = np.array(distances).T

    return np.argmin(distances, axis=1)


##################################################
if __name__ == "__main__":
    # NÚMERO DE GRUPOS
    K = 8
    # CARGAR DATOS
    img = cv2.imread("Paisaje.jpg")
    re2 = img.shape[2]
    data = img.reshape(-1, re2)  # Imagen reshapeada para manejarla más fácil
    data = data.astype("int32")

    # GENERAR CENTROIDES
    cens = genCen(K, np.unique(data, axis=0))

    # Primera agrupación
    labels = kmeansVectorizado(data, cens)
    newCens = updateCens(K, data, labels)  # <<<Calcular nuevos centroides

    iters = 15

    # REPETIR
    for _ in range(iters):
        labels = kmeansVectorizado(data, cens)

        # Actualizar centroides
        cens = newCens.copy()
        newCens = updateCens(K, data, labels)
        if (cens == newCens).all():
            break

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
