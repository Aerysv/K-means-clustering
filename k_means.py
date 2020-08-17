import numpy as np
import matplotlib.pyplot as plt
import random


def generate_data(n):
    X = np.random.random_sample(n)
    Y = np.random.random_sample(n)
    return (X, Y)


def graficar(X, Y, K, grupos):
    plt.figure(1)
    plt.scatter(X, Y, color="black", s=10)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.figure(2)
    plt.scatter(grupos[0, :, 0], grupos[0, :, 1], color="red", s=5)
    plt.scatter(K[0, -1, 0], K[0, -1, 1], color="red", s=200)

    plt.scatter(grupos[1, :, 0], grupos[1, :, 1], color="blue", s=5)
    plt.scatter(K[1, -1, 0], K[1, -1, 1], color="blue", s=200)

    plt.scatter(grupos[2, :, 0], grupos[2, :, 1], color="green", s=5)
    plt.scatter(K[2, -1, 0], K[2, -1, 1], color="green", s=200)

    plt.scatter(grupos[3, :, 0], grupos[3, :, 1], color="purple", s=5)
    plt.scatter(K[3, -1, 0], K[3, -1, 1], color="purple", s=200)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.show()


def euclidean_distance(x, y, k_x, k_y):
    return ((x - k_x)**2 + (y - k_y)**2)**0.5


def manhatan_distance(x, y, k_x, k_y):
    return abs(x - k_x) + abs(y - k_y)


def mean(XY, k_x, k_y):
    len_x = 0
    sum_x = 0
    len_y = 0
    sum_y = 0

    for item in XY[:, 0]:
        if item != 0:
            len_x += 1
            sum_x += item
    for item in XY[:, 1]:
        if item != 0:
            len_y += 1
            sum_y += item

    mean_x = sum_x/len_x if len_x != 0 else k_x
    mean_y = sum_y/len_y if len_y != 0 else k_y
    return (mean_x, mean_y)


def cluster_points(X, Y, K_x, K_y, n_iter):
    d = np.zeros(len(K_x))
    groups = np.zeros([len(K_x), len(X), 2])
    groups_clustered = np.zeros([len(K_x), len(X), 2])
    counter = 0
    K = np.zeros([len(K_x), n_iter, 2])
    while counter < n_iter:
        for i in range(0, len(K_x)):
            K[i, counter, 0] = K_x[i]
            K[i, counter, 1] = K_y[i]
        counter += 1
        for point in range(0, len(X)):
            x = X[point]
            y = Y[point]
            for k in range(0, len(K_x)):
                d[k] = euclidean_distance(x, y, K_x[k], K_y[k])
                # d[k] = manhatan_distance(x, y, K_x[k], K_y[k])
            # Función de np para encontrar el indice del mínimo
            # Regresa un array, [0][0] da el resultado correcto
            nearest_K = np.where(d == np.amin(d))[0][0]
            groups[nearest_K][point] = [x, y]
        for i in range(0, len(K_x)):
            K_x[i], K_y[i] = mean(groups[i], K_x[i], K_y[i])

    for i in range(0, len(K_x)):
        groups_clustered[i] = groups[i]

    return (K, groups_clustered)


def main():
    n_puntos = 100   # número de puntos
    n_K = 4         # Número de grupos
    n_iter = 10     # Número de iteraciones

    # Se genera el conjunto de datos
    X, Y = generate_data(n_puntos)
    # Se inicializan aleatoriamente las K
    K_x, K_y = generate_data(n_K)

    K, grupos = cluster_points(X, Y, K_x, K_y, n_iter)
    graficar(X, Y, K, grupos)


if __name__ == "__main__":
    main()
