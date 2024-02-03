from data.power_system import string_matrix, PowerSystem
import numpy as np


if __name__ == "__main__":

    Z = [[1, (1,2), 0.888, 0.008],
         [1, (1,3), 1.173, 0.008],
         [3, (2,), -0.501, 0.010],
         [2, (1,2), 0.568, 0.008],
         [2, (1,3), 0.663, 0.008],
         [4, (2,), -0.286, 0.010],
         [0, (1,), 1.006, 0.004],
         [0, (2,), 0.968, 0.004]]

    R = np.array([[0.00, 0.01, 0.02],
                 [0.01, 0.00, 0.03],
                 [0.02, 0.03, 0.00]])

    X = np.array([[0.00, 0.03, 0.05],
                 [0.03, 0.00, 0.08],
                 [0.05, 0.08, 0.00]])

    edges = [(1,2), (1,3), (2,3)]
    
    power_system = PowerSystem(Z, R, X, edges)

    h = power_system.get_h(x)

    power_system.describe_h(h)
