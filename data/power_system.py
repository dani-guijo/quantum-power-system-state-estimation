import numpy as np


class PowerSystem():
    """
    Class to create a Power System scenario.
    """

    def __init__(self,
                 Z: list,
                 R: np.ndarray,
                 X: np.ndarray,
                 edges: list
                 ):

        '''
        Initializes a Power System.

        Z: Array of measurements (type, nodes, value, sigma)
        
            Type | Description | 
            -----|------------
            0    | Voltage
            1    | Power Flow between two nodes (Real)
            2    | Power Flow between two nodes (Imaginary)
            3    | Power injection at a node (Real)
            4    | Power injection at a node (Imaginary)

        R: Resistance Matrix
        X: Reactance Matrix
        edges: List of edges (i, j) where i and j are the nodes
        ''' 
        self.Z = Z
        self.Z_values = np.array([value for _, _, value, _ in Z])
        self.R = R
        self.X = X
        self.n_measurements = len(Z)
        self.n_nodes = len(R)
        self.n_edges = len(edges)
        self.edges = edges
        
        self.admitance = self.get_admitance(R, X)
        self.bus_admitance = self.get_bus_admitance(self.admitance)

        self.weights = np.array([1 / sigma**2 for _, _, _, sigma in Z])
        W = np.diag(self.weights)
        self.W = W


    def get_admitance(self, R, X):
        '''
        Get Admitance Matrix

        R: Resistance Matrix
        X: Reactance Matrix
        '''
        admitance = np.zeros((self.n_nodes, self.n_nodes), dtype=np.complex128)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if R[i, j] != 0 or X[i, j] != 0:
                    admitance[i, j] = 1 / (R[i, j] + 1j * X[i, j])

        return admitance
    
    def get_bus_admitance(self, admitance):
        '''
        Get Bus Admitance Matrix

        admitance: Admitance Matrix
        '''
        bus_admitance = np.zeros(self.admitance.shape, dtype=np.complex128)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j:
                    bus_admitance[i, j] = np.sum(admitance[i, :]) - admitance[i, i]
                else:
                    bus_admitance[i, j] = - admitance[i, j]

        bus_admitance = bus_admitance + np.diag(np.sum(admitance, axis=1))
        return bus_admitance