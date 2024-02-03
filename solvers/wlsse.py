'''
Weighted Least Squares State Estimation
Classical method for state estimation in power systems using weighted least squares.

To Do:
    - Add the current measurements
    - Add the shunt admittance
    - Improve the performance of the code

Last Update: 18/01/2024
'''


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

class WeightedLeastSquaresSE:
    def __init__(self, Z: list, R: np.ndarray, X: np.ndarray, Edges: list):
        '''
        Weighted Least Squares State Estimation

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
        Edges: List of edges (i, j) where i and j are the nodes
        ''' 
        self.Z = Z
        self.Z_values = np.array([value for _, _, value, _ in Z])
        self.R = R
        self.X = X
        self.n_measurements = len(Z)
        self.n_nodes = len(R)
        self.n_edges = len(Edges)
        self.Edges = Edges
        
        self.Admitance = self.get_admitance(R, X)
        self.Bus_Admitance = self.get_Bus_Admitance(self.Admitance)

        self.weights = np.array([1 / sigma**2 for _, _, _, sigma in Z])
        W = np.diag(self.weights)
        self.W = W

    def get_admitance(self, R, X):
        '''
        Get Admitance Matrix

        R: Resistance Matrix
        X: Reactance Matrix
        '''
        Admitance = np.zeros((self.n_nodes, self.n_nodes), dtype=np.complex128)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if R[i, j] != 0 or X[i, j] != 0:
                    Admitance[i, j] = 1 / (R[i, j] + 1j * X[i, j])

        return Admitance
    
    def get_Bus_Admitance(self, Admitance):
        '''
        Get Bus Admitance Matrix

        Admitance: Admitance Matrix
        '''
        Bus_Admitance = np.zeros(self.Admitance.shape, dtype=np.complex128)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j:
                    Bus_Admitance[i, j] = np.sum(Admitance[i, :]) - Admitance[i, i]
                else:
                    Bus_Admitance[i, j] = - Admitance[i, j]

        Bus_Admitance = Bus_Admitance + np.diag(np.sum(Admitance, axis=1))
        return Bus_Admitance
    
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
        print(string_matrix(self.Admitance.round(4)))
        print('Bus Admitance Matrix:\n')
        print(string_matrix(self.Bus_Admitance.round(4)))

    def get_measurements_jacobian(self, x, z_type, z_nodes, gs=None):
        '''
        Get the Jacobian H matrix row for the measurements

        x: Array of variables (theta, V)
        z_type: int with the type of measurement
        z_nodes: array with the nodes of the measurement
        gs: Array with the shunt admittance of each node
        '''
        n_nodes = self.n_nodes
        A = self.Admitance
        Bus_Admitance = self.Bus_Admitance

        thetas = np.array([0] + list(x[:n_nodes-1]))
        V_mag = x[n_nodes-1:]
        V = x[n_nodes-1:] * np.exp(1j * thetas)
        z_nodes = np.array(z_nodes) - 1
        gs = np.zeros(n_nodes, dtype=np.complex128) if gs is None else gs

        row = np.zeros(len(x), dtype=np.float64)

        if z_type == 0:
            # Voltage
            row[z_nodes[0] + n_nodes - 1] = 1
            return row

        if z_type == 1:
            # Power Flow between two nodes (Real)
            i, j = z_nodes
            if i != 0:
                row[i - 1] = (-1j * V[i]*V[j].conjugate() * A[i, j].conjugate()).real
            if j != 0:
                row[j - 1] = (1j * V[i]*V[j].conjugate() * A[i, j].conjugate()).real
            
            row[i + n_nodes - 1] = (2 * V_mag[i] * ( A[i, j].conjugate() + gs[i].conjugate()) - V_mag[j] * A[i, j].conjugate() * np.exp(1j * (thetas[i] - thetas[j]))).real
            row[j + n_nodes - 1] = ( - V_mag[i] * A[i, j].conjugate() * np.exp(1j * (thetas[i] - thetas[j]))).real

            return row
        
        if z_type == 2:
            # Power Flow between two nodes (Imaginary)
            i, j = z_nodes
            if i != 0:
                row[i - 1] = (-1j * V[i]*V[j].conjugate() * A[i, j].conjugate()).imag
            if j != 0:
                row[j - 1] = (1j * V[i]*V[j].conjugate() * A[i, j].conjugate()).imag
            
            row[i + n_nodes - 1] = (2 * V_mag[i] * ( A[i, j].conjugate() + gs[i].conjugate()) - V_mag[j] * A[i, j].conjugate() * np.exp(1j * (thetas[i] - thetas[j]))).imag
            row[j + n_nodes - 1] = ( - V_mag[i] * A[i, j].conjugate() * np.exp(1j * (thetas[i] - thetas[j]))).imag

            return row
        
        if z_type == 3: 
            # Power injection at a node (Real)
            i = z_nodes[0]
            for j in range(n_nodes):
                if i == j:
                    continue
                if i != 0:
                    row[i - 1] += (1j * V[i]*V[j].conjugate() * Bus_Admitance[i, j].conjugate()).real
                if j != 0:
                    row[j - 1] = (-1j * V[i]*V[j].conjugate() * Bus_Admitance[i, j].conjugate()).real
                row[i + n_nodes - 1] += (V_mag[j] * Bus_Admitance[i, j].conjugate() * np.exp(1j * (thetas[i] - thetas[j]))).real
                row[j + n_nodes - 1] =  (V_mag[i] * Bus_Admitance[i, j].conjugate() * np.exp(1j * (thetas[i] - thetas[j]))).real
            
            row[i + n_nodes - 1] += V_mag[i] * Bus_Admitance[i, i].real

            return row
        
        if z_type == 4:
            # Power injection at a node (Imaginary)
            i = z_nodes[0]
            for j in range(n_nodes):
                if i == j:
                    continue
                if i != 0:
                    row[i - 1] += (1j * V[i]*V[j].conjugate() * Bus_Admitance[i, j].conjugate()).imag
                if j != 0:
                    row[j - 1] = (-1j * V[i]*V[j].conjugate() * Bus_Admitance[i, j].conjugate()).imag
                row[i + n_nodes - 1] += (V_mag[j] * Bus_Admitance[i, j].conjugate() * np.exp(1j * (thetas[i] - thetas[j]))).imag
                row[j + n_nodes - 1] =  (V_mag[i] * Bus_Admitance[i, j].conjugate() * np.exp(1j * (thetas[i] - thetas[j]))).imag

            row[i + n_nodes - 1] -= V_mag[i] * Bus_Admitance[i, i].imag

            return row
        
        # TODO: Add the current measurements
        return row

    def get_H(self, x):
        '''
        Get the Jacobian H matrix for the measurements

        x: Array of variables (theta, V)
        '''
        Z = self.Z
        A = self.Admitance
        Bus_Admitance = self.Bus_Admitance

        n_measurements = len(Z)
        n_nodes = len(A)
        n_variables = 2 * n_nodes - 1 # Number of variables (V and theta for each node minus the reference node)
        H = np.zeros((n_measurements, n_variables), dtype=np.float64)

        for i, (type, nodes, value, sigma) in enumerate(Z):
            H[i] = self.get_measurements_jacobian(x, type, nodes)

        return H

    def get_G(self, H):
        '''
        Get the G matrix for the measurements

        H: Jacobian H matrix
        '''
        return H.T @ self.W @ H
    
    def calculate_h(self, x, z_type, z_nodes, gs=None):
        '''
        Calculate the h vector for the measurements

        x: Array of variables (theta, V)
        z_type: int with the type of measurement
        z_nodes: array with the nodes of the measurement
        gs: Array with the shunt admittance of each node
        '''
        n_nodes = self.n_nodes
        A = self.Admitance
        Bus_Admitance = self.Bus_Admitance

        thetas = np.array([0] + list(x[:n_nodes-1]))
        V_mag = x[n_nodes-1:]
        V = x[n_nodes-1:] * np.exp(1j * thetas)
        z_nodes = np.array(z_nodes) - 1
        gs = np.zeros(n_nodes, dtype=np.complex128) if gs is None else gs

        if z_type == 0:
            # Voltage
            return np.abs(V[z_nodes[0]])

        if z_type == 1:
            # Power Flow between two nodes (Real)
            i, j = z_nodes
            return (V_mag[i] ** 2 * (A[i, j].conjugate() + gs[i].conjugate()) - V[i] * V[j].conjugate() * A[i, j].conjugate()).real
        if z_type == 2:
            # Power Flow between two nodes (Imaginary)
            i, j = z_nodes
            return (V_mag[i] ** 2 * (A[i, j].conjugate() + gs[i].conjugate()) - V[i] * V[j].conjugate() * A[i, j].conjugate()).imag
        
        if z_type == 3: 
            # Power injection at a node (Real)
            i = z_nodes[0]
            return (V[i] * np.sum(V.conjugate() * Bus_Admitance[i, :].conjugate()) - 1/2 * V_mag[i] ** 2 * Bus_Admitance[i, i].conjugate()).real
        
        if z_type == 4:
            # Power injection at a node (Imaginary)
            i = z_nodes[0]
            return (V[i] * np.sum(V.conjugate() * Bus_Admitance[i, :].conjugate()) - 1/2 * V_mag[i] ** 2 * Bus_Admitance[i, i].conjugate()).imag
    
    def get_h(self, x):
        '''
        Get the h vector for the measurements

        x: Array of variables (theta, V)
        '''
        n_measurements = self.n_measurements
        Z = self.Z
        A = self.Admitance
        Bus_Admitance = self.Bus_Admitance

        h = np.zeros(n_measurements, dtype=np.float64)

        for i, (type, nodes, value, sigma) in enumerate(Z):
            h[i] = self.calculate_h(x, type, nodes)

        return h
    
    def get_residuals(self, h):
        '''
        Get the residuals vector for the measurements

        h: h vector for the measurements
        x: Array of variables (theta, V)
        '''
        residuals = self.Z_values - h
        return residuals
    
    def describe_h(self, h):
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
        print('\n{:^60}\n'.format('Estimated Measurements'))
        print(f'Number of measurements: {self.n_measurements}')
        print()

        data = [['Measurement', 'Type', 'Value (pu)', 'h (pu)', 'residual (pu)']]
        for i, (type, nodes, value, sigma) in enumerate(self.Z):
            data.append([i+1, types[type](*nodes), value, h[i], (self.Z_values[i] - h[i]).round(3)])

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


    def solve(self, x0=None, convergence=1e-4, max_iter=10):
        x0 = np.concatenate((np.zeros(self.n_nodes - 1), np.ones(self.n_nodes))) if x0 is None else x0
        x = x0.copy()
        for i in range(max_iter):
            H = self.get_H(x)
            h = self.get_h(x)
            r = self.get_residuals(h)
            G = self.get_G(H)

            dx = np.linalg.inv(G) @ H.T @ self.W @ r
            x = x + dx

            if np.all(np.abs(dx) < convergence):
                break

        # Radians to degrees
        x[:self.n_nodes-1] = x[:self.n_nodes-1] * 180 / np.pi
        
        print(f'Number of iterations: {i+1}\n')
        
        thetas = x[:self.n_nodes-1]
        V_mag = x[self.n_nodes-1:]

        results_table = [f'Theta {i+1}' for i in range(self.n_nodes-1)] + [f'V {i+1}' for i in range(self.n_nodes)]
        results = np.concatenate((thetas, V_mag))
        results = np.array([f'{value:.3f}' for value in results])
        results = np.stack((results_table, results), axis=1)

        print(string_matrix(results))
        self.describe_h(h.round(3))

        return x, r, G, H, h