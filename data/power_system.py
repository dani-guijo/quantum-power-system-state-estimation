import numpy as np


def string_matrix(A):
    col_widths = [max(len(str(item)) for item in col) for col in zip(*A)]

    # Create the format string with the column alignments
    format_str = ''
    for row in A:
        format_str_aux = '{:<%d}  ' % col_widths[0]
        for width in col_widths[1:-1]:
            format_str_aux += ' {:^%d} ' % width
        format_str_aux += ' {:>%d} ' % col_widths[-1]
        format_str += format_str_aux.format(*row)
        format_str += '\n'
    return format_str


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
                if R[i, j] != 0.00 or X[i, j] != 0.00:
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


    def describe(self):
        '''
        Print Measurements Description

        Z: Array of measurements (type, nodes, value, R)
        A: Matrix of Admitance
        

            Type | Description | 
            -----|------------
            0    | Voltage
            1    | Power Flow between two nodes (Real)
            2    | Power Flow between two nodes (Imaginary)
            3    | Power injection at a node (Real)
            4    | Power injection at a node (Imaginary)

        '''
        types = [lambda i: f'V_{i}', lambda i, j: f'P_{{{i}{j}}}', lambda i, j: f'Q_{{{i}{j}}}', lambda i: f'P_{{{i}}}', lambda i: f'Q_{{{i}}}']
        print('\n{:^60}\n'.format('Measurements'))
        print(f'Number of measurements: {self.n_measurements}')
        print()

        data = [['Measurement', 'Type', 'Value (pu)', 'Sigma (pu)']]
        for i, (type, nodes, value, sigma) in enumerate(self.Z):
            data.append([i+1, types[type](*nodes), value, sigma])

        # Find the maximum width for each column
        col_widths = [max(len(str(item)) for item in col) for col in zip(*data)]

        # Create the format string with different alignments
        format_str = '{:<%d} | ' % col_widths[0]
        for i, width in enumerate(col_widths[1:-1]):
            i += 1
            format_str += '{:^%d}' % width
            if i < len(col_widths)-1:
                format_str += ' | '
        format_str += '{:>%d}' % col_widths[-1]

        print('-' * len(format_str.format(*data[0])))
        for i, row in enumerate(data):
            print(format_str.format(*row))
            if i == 0:
                print('-' * len(format_str.format(*row)))
        
        print('-' * len(format_str.format(*row)))

        print('Resistance Matrix:\n')
        print(string_matrix(self.R.round(4)))
        print('Reactance Matrix:\n')
        print(string_matrix(self.X.round(4)))
        print('Admitance Matrix:\n')
        print(string_matrix(self.admitance.round(4)))
        print('Bus Admitance Matrix:\n')
        print(string_matrix(self.bus_admitance.round(4)))