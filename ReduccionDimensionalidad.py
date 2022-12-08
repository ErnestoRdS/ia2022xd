# Ilse Adriana Pacheco Torres

from sklearn.neighbors import KNeighborsClassifier  # Clasificador KNN
from sklearn.metrics import accuracy_score  # Permite calcular la exactitud
from sklearn.model_selection import train_test_split  # Separar los conjuntos
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

# Función para generar los individuos
def generate_individuals(p_vector, l):
    # Asignamos valores aleatorios
    a = np.random.rand(l)
    b = np.random.rand(l)

    # Comparamos los individuos contra el vector de probabilidades
    for i in range(l):
        a[i] = 1 if a[i] < p_vector[i] else 0
        b[i] = 1 if b[i] < p_vector[i] else 0

    return a, b


# Función para que compitan las evaluaciones obtenidas
def compete(a, b, clasf, tr_patterns, tr_labels):
    f_a = evaluate(a, clasf, tr_patterns, tr_labels)
    f_b = evaluate(b, clasf, tr_patterns, tr_labels)

    return (a, b) if f_a <= f_b else (b, a)


# Función para evaluar la función objetivo
def evaluate(indiv, clasf, tr_patterns, tr_labels):
    reduced_trpa = tr_patterns[:, indiv == 1]
    clasf.fit(reduced_trpa, tr_labels)

    acc_tr = accuracy_score(tr_labels, clasf.predict(reduced_trpa))
    return 0.8 * (1 - acc_tr) + 0.2 * (np.sum(indiv) / tr_patterns.shape[1])


# Función para actualizar las probabilidades
def update_probabilities(p_vector, winner, loser, l, n):
    for i in range(l):
        if winner[i] != loser[i]:
            if winner[i] == 1:
                p_vector[i] += 1 / n
            else:
                p_vector[i] -= 1 / n

    return np.around(p_vector, decimals = 2)


# Función para verificar si el vector ha convergido
def converging(p_vector):
    zeros = 0
    ones = 0
    for i in p_vector:
        if i == 0:
            zeros += 1
        elif i == 1:
            ones += 1

    return zeros + ones


def find_solution(n, l, clasf, tr_patterns, tr_labels):
    # Inicializamos el vector de probabilidades
    p_vector = np.full((l), 0.5, dtype = float)

    while converging(p_vector) != l:

        # Generamos los individuos
        a, b = generate_individuals(p_vector, l)

        # Evaluamos y definimos al winner y loser
        winner, loser = compete(a, b, clasf, tr_patterns, tr_labels)

        # Actualizamos las probabilidades
        p_vector = update_probabilities(p_vector, winner, loser, l, n)

    return np.array(p_vector, dtype = int)


if __name__ == "__main__":
    patterns, labels = load_digits(return_X_y=True)  # Cargamos el conjunto de datos

    # Separamos los datos en 90% entrenamiento y 10% prueba
    tr_patterns, te_patterns, tr_labels, te_labels = train_test_split(patterns, labels, test_size=0.1, stratify=labels)

    # Definimos el clasificador a utilizar (KNN)
    classifier = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

    n = 10  # Tamaño de la población
    l = patterns.shape[1]  # Tamaño del cromosoma
    solution = find_solution(n, l, classifier, tr_patterns, tr_labels) # Obtenemos la solución

    print(f"\nCaracteristicas seleccionadas: {np.sum(solution)}")
    print(f"Solución: {solution}")
    print(f"Índices correspondientes: {np.where(solution == 1)}")

    # Reducimos los conjuntos de entrenamiento y prueba
    reduced_tr_patterns = tr_patterns[:, solution == 1]
    reduced_te_patterns = te_patterns[:, solution == 1]

    # Clasificamos los datos reducidos
    classifier.fit(reduced_tr_patterns, tr_labels)

    # Creamos un clasificador para los datos completos
    exp_classifier = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    exp_classifier.fit(tr_patterns, tr_labels)

    # Mostramos la exactitud de los conjuntos de entrenamiento
    print(f"\nExactitud entrenamiento: {np.round(accuracy_score(tr_labels, exp_classifier.predict(tr_patterns)), 2)}")
    print(f"Exactitud entrenamiento reducido {np.round(accuracy_score(tr_labels, classifier.predict(reduced_tr_patterns)), 2)}")

    # Mostramos la exactitud de los conjuntos de prueba
    print(f"\nExactitud prueba: {np.round(accuracy_score(te_labels, exp_classifier.predict(te_patterns)), 2)}")
    print(f"Exactitud prueba reducido: {np.round(accuracy_score(te_labels, classifier.predict(reduced_te_patterns)), 2)}\n")

    # Mostramos una imagen de prueba
    id = 50
    img_filtrada = solution * te_patterns[id, :]

    plt.matshow(np.reshape(te_patterns[id,:], (8,8)))
    txt = f"Dimensión original | Etiqueta: {str(te_labels[id])}"
    plt.title(txt)
    plt.gray()
    plt.show()

    plt.matshow(np.reshape(img_filtrada, (8,8)))
    txt = f"Dimensión reducida | Etiqueta: {str(te_labels[id])}"
    plt.title(txt)
    plt.show()