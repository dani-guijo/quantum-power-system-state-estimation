import numpy as np

def create_example(n_buses=3):
    """
    Returns an example of a PowerSystem object with 3 buses (AVAILABLE)
    Creates a PowerSystem object with the specified parameters (NOT AVAILABLE)

    n_buses: Number of buses in the PowerSystem

    return: PowerSystem
    """

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

    return power_system
