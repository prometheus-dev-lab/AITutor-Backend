import numpy as np


def edit_distance(s1, s2):
    n = len(s1)
    m = len(s2)

    # create matrix of zeroes of the lenghts of s1 and s2
    distance_matrix = np.zeros((n + 1, m + 1))
    for col in range(1, n + 1):
        distance_matrix[col, 0] = distance_matrix[col - 1, 0] + 1

    for row in range(1, m + 1):
        distance_matrix[0, row] = distance_matrix[0, row - 1] + 1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Calculate Insertion, Deletion and Subsitution cost:
            insertion = distance_matrix[i, j - 1] + 1
            deletion = distance_matrix[i - 1, j] + 1

            if s1[i - 1] != s2[j - 1]:  # Case Sub are different
                replace_same = distance_matrix[i - 1, j - 1] + 2

            else:  # Case Sub is same
                replace_same = distance_matrix[i - 1, j - 1]

            distance_matrix[i, j] = min([insertion, deletion, replace_same])

    return int(distance_matrix[n][m])
