from sklearn.neighbors import KNeighborsClassifier  # clasificador
from sklearn.metrics import accuracy_score  # calcula la exactitud
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits  # carga el dataset
import numpy as np  # para manipular arreglos


# Función para evaluar -> Va a recibir el set X y el individuo actual
def evaluate(ind, knn):
    reduced = xTrain[:, ind == 1]
    knn.fit(reduced, yTrain)  # CREO QUE PUEDE QUEDAR knn.fit(xTrain[:,ind==1], yTrain)
    accReduced = accuracy_score(yTrain, knn.predict(reduced))
    fitness = 0.8 * (1 - accReduced) + 0.2 * (np.sum(ind) / xTrain.shape[-1])
    return fitness


# Función para hacer el UMDAxd
def umda(N, M, gens, knn):  # , lI, lS, freecss):
    # GENERAR VECTOR DE PROBABILIDAD
    pv = np.full(xTrain.shape[-1], 0.5)
    # GENERAR POBLACIÓN INICIAL
    pop = np.random.random((N, xTrain.shape[-1]))
    # Para guardar los fitnesses
    fitnesses = np.zeros(len(pop))

    # REPETIR
    for _ in range(gens):
        # COMPARAR CON EL VECTOR DE PROBABILIDAD
        pop = (pop < pv).astype(int)
        # EVALUAR
        for id, ind in enumerate(pop):
            fitnesses[id] = evaluate(ind, knn)
        # Sakr media para actualizar el vector de probabilidades
        """
            ¿C OQPA UN VALOR ALTO, O UNO BAJO EN EL FITNESS?
            Por el momento lo dejaré con valor alto
        """
        # Actualizar vector de probabilidad y redondear(?)
        # pv = np.around(np.sum(pop[np.argsort(-fitnesses)][:M], axis=0) / M, 4)
        pv = np.around(np.sum(pop[np.argsort(fitnesses)][:M], axis=0) / M, 4)
    # REGRESAR el mejor de la población(?)
    pop = (pop < pv).astype(int)
    for id, ind in enumerate(pop):
        fitnesses[id] = evaluate(ind, knn)
    # return pop[np.argsort(-fitnesses)][:1]
    return pop[np.argsort(fitnesses)][:1]


if __name__ == "__main__":
    # VALORES
    N = 10  # Número de individuos
    M = 3  # Mejores a seleccionar
    gens = 1
    test_size = 0.1
    # CARGAR DATOS
    x, y = load_digits(return_X_y=True)
    # SEPARAR LOS ACÁS
    xTr, xTe, yTr, yTe = train_test_split(x, y, test_size=test_size, stratify=y)
    # GLOBALIZAR
    global xTrain, yTrain
    xTrain, yTrain = xTr, yTr
    # KNN -> Clasificador a utilizar
    myKnn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    myKnn.fit(xTrain, yTrain)

    # MEJOR SOLUCIÓN
    sf = umda(N, M, gens, myKnn)
    sf = np.reshape(sf, xTrain.shape[-1])

    # REDUCIR ENTRENAMIENTO Y PRUEBA
    reducedXTrain = xTrain[:, sf == 1]
    reducedXTest = xTe[:, sf == 1]
    # ENTRENAR KNN-REDUCIDO Y COMPLETO
    ###Entrenar Completo
    myKnn.fit(xTrain, yTrain)
    ###Crear y entrenar reducido
    reducedKnn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    reducedKnn.fit(reducedXTrain, yTrain)
    # RESULTADOS FINALES
    print(f"Cantidad de características a usar = {np.sum(sf)}")
    ###Conjuntos de Entrenamiento
    print(
        f"Exactitud de Entrenamiento-Completo = {accuracy_score(yTrain, myKnn.predict(xTrain))}"
    )
    print(
        f"Exactitud de Entrenamiento-Reducido = {accuracy_score(yTrain, reducedKnn.predict(reducedXTrain))}"
    )
    ###Conjuntos de Prueba
    print(f"Exactitud de Prueba-Completo = {accuracy_score(yTe, myKnn.predict(xTe))}")
    print(
        f"Exactitud de Prueba-Reducido = {accuracy_score(yTe, reducedKnn.predict(reducedXTest))}"
    )
