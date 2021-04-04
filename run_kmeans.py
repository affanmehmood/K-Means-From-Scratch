import numpy as np
import random


def calc_distance(x1, x2):
    return abs(x1 - x2)


def find_closest_centroid(init_centroids, x):
    assigned_centroids = []
    for point in x:
        distances = []
        for centroid in init_centroids:
            distances.append(calc_distance(point, centroid))
        assigned_centroids.append(np.argmin(distances))
    return assigned_centroids


def recalculate_centroids(x, total_centroids, assigned_centroids, old_centroids):
    for k in range(total_centroids):
        if sum(assigned_centroids == k) > 0:
            old_centroids[k] = sum(x[assigned_centroids == k]) / sum(assigned_centroids == k)
    return old_centroids


def predict(input_val, old_centroids):
    distances = []
    for centroid in old_centroids:
        distances.append(calc_distance(input_val, centroid))

    return np.argmin(distances) + 1


if __name__ == "__main__":
    K = 3
    X = np.append(np.random.default_rng().uniform(1, 5, 200),
                  [np.random.default_rng().uniform(300, 305, 200), np.random.default_rng().uniform(600, 605, 200)])
    centroids = []
    for i in random.sample(range(1, len(X)), K):
        centroids.append(X[i])

    centroids = np.array(centroids)
    classified_centroids = np.array(find_closest_centroid(centroids, X))

    print("Old centroids", centroids)
    while True:
        oc = centroids
        centroids = np.array(recalculate_centroids(x=X, total_centroids=K,
                                                   assigned_centroids=classified_centroids, old_centroids=centroids))
        classified_centroids = np.array(find_closest_centroid(centroids, X))
        if np.array_equal(oc, centroids):
            break
    print("New centroids", centroids)

    while True:
        print("The value belongs to number",
              predict(int(input("Enter an int you want to predict ")), centroids), "centroid")
