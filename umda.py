import numpy as np
import cv2
import matplotlib.pyplot as plt

# Función para evaluar -> Va a recibir las frecuencias y el individuo actual
def evaluate(freecss, umbs):
    if (umbs < 1).any():
        return 9999
    if (umbs > 254).any():
        return 9999
    # Copiar umbrales y ordenarlos
    umbs = umbs.copy()[np.argsort(umbs)]
    # Chekr que no haya repetidos
    if len(umbs) != len(set(umbs)):
        return 9999
    # Para guardar las probabilidades acumuladas y medias
    wk = np.zeros(len(umbs) + 1)
    mk = np.zeros(len(umbs) + 1)
    i = 0
    for id, u in enumerate(umbs):
        wk[id] = np.sum(freecss[1][i:u] / np.sum(freecss[1]))
        mk[id] = np.sum(
            (freecss[0][i:u] * (freecss[1][i:u] / np.sum(freecss[1]))) / wk[id]
        )
        i = u
    wk[-1] = np.sum(freecss[1][i:256] / np.sum(freecss[1]))
    mk[-1] = np.sum(
        (freecss[0][i:256] * (freecss[1][i:256] / np.sum(freecss[1]))) / wk[-1]
    )
    # Sakr intensidad
    mt = np.sum(np.multiply(wk, mk))
    # Sakr varianza
    g = np.sum(np.multiply(wk, np.power((mk - mt), 2)))
    return np.power(g, -1)


# Función para hacer el UMDAxd
def umda(K, N, M, gens, lI, lS, freecss):
    # GENERAR POBLACIÓN
    pop = np.random.randint(lI, lS, (N, K))
    # Para guardar los fitnesses
    fitnesses = np.zeros(len(pop))
    # REPETIR
    for _ in range(gens):
        for id, ind in enumerate(pop):
            fitnesses[id] = evaluate(freecss, ind)
        # Sakr media y desviación
        mu = np.mean(pop[np.argsort(fitnesses)[:M]], axis=0)
        g = np.sqrt(
            np.sum(np.power(pop[np.argsort(fitnesses)[:M]] - mu, 2), axis=0) / M
        )
        # Generar la población pero con la media y desviación estándar
        pop = np.random.normal(mu, g, (N, K)).astype(int)
    # RESULTADOSxd
    return pop[np.argsort(fitnesses)[0]]


if __name__ == "__main__":
    # VALORES
    K = 4  # Umbrales -> sería algo así como el tamaño del cromosoma (l)
    N = 20  # Pobladores
    M = 8  # Selección
    gens = 20  # Generaciones
    # CARGAR IMAGEN
    img = cv2.imread("Paisaje.jpg")
    # PASAR A ESCALA DE GRISES
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap=plt.get_cmap("gray"))
    plt.show()
    ogShape = gray.shape
    # SAKR HISTOGRAMA
    freecss = np.unique(gray, return_counts=True)

    gray = gray.reshape(-1)  # Reshapear para la hora de colorear
    # Dejar el límite Inf en 1, y el Sup en 254
    umbs = umda(K, N, M, gens, 1, 254, freecss)
    umbs = umbs[np.argsort(umbs)]
    print(umbs)

    # MOSTRAR HISTOGRAMA
    plt.plot(freecss[0], freecss[1])
    for u in umbs:
        plt.plot([u, u], [0, np.max(freecss[1])])
    plt.show()

    # COLOREAR
    gray[gray < umbs[0]] = 0
    gray[gray >= umbs[-1]] = 255
    if K != 1:
        for id, u in enumerate(umbs[: K - 1]):
            gray[(gray >= u) == (gray < umbs[id + 1])] = u
    # Reshapear
    gray = np.reshape(gray, ogShape)
    plt.imshow(gray, cmap=plt.get_cmap("gray"))
    plt.show()
