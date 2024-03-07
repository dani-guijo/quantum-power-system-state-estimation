import pennylane as qml
import pennylane.numpy as np
from itertools import product

import torch
import os


def matrix2paulis(A):
    """
    Convert a matrix to a list of Pauli operators.

    Args:
        A (np.ndarray): A square matrix.

    Returns:
        list: A list of Pauli operators and their coefficients.
        dict: A dictionary of Pauli operators.
    """
    n = A.shape[0]
    assert A.shape[1] == n

    m = np.ceil(np.log2(n)).astype(int)


    if n != 2 ** m:
        # pad A with zeros
        A = np.pad(A, ((0, 2 ** m - n), (0, 2 ** m - n)), 'constant')
        # print('A is padded with zeros to size {} x {}.'.format(2 ** m, 2 ** m))
        n = 2 ** m

    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    paulis_2x2 = [I, X, Y, Z]
    paulis_index = ['I', 'X', 'Y', 'Z']
    paulis_dict = dict(zip(paulis_index, paulis_2x2))

    A_in_paulis_basis = []
    basis_operators = {}

    for p in product(paulis_index, repeat=m):
        basis_operator = paulis_dict[p[0]]
        basis_key = ''.join(p)
        for i in range(1, m):
            basis_operator = np.kron(basis_operator, paulis_dict[p[i]])
        basis_operators[basis_key] = basis_operator
        value = np.trace(np.matmul(A, basis_operator)) / n
        A_in_paulis_basis.append([value, basis_key])
    
    return A_in_paulis_basis, basis_operators, A

class VQLS:
    """
    Variational Quantum Linear Solver (VQLS) class.    
    """
    def __init__(self, weights_shape=None,
                 variational_block=None,
                 n_shots=10**6,
                 steps=1,
                 learning_rate=0.001,
                 q_delta=0.001,
                 gamma=0.005,
                 best_weights_path='data/weights/'):
        """
        Args:
            A (np.ndarray): A square matrix or list of Pauli operators.
            b (np.ndarray): A vector.
            n_qubits (int): The number of qubits to be used.
            variational_block (function): A function that returns a quantum circuit.
            weights_shape (tuple): The shape of the weights to be used in the variational block.
            n_shots (int): The number of shots to be used in the quantum circuit.
            steps (int): The number of steps to be used in the optimization.
            learning_rate (float): The learning rate to be used in the optimization.
            q_delta (float): The step size to be used in the finite difference method.
            gamma (float): The threshold to be used in the optimization.
            best_weights_path (str): The path to save the best weights.
        """ 
        self.n_shots = n_shots
        self.steps = steps
        self.learning_rate = learning_rate
        self.q_delta = q_delta

        self.variational_block = variational_block
        self.weights_shape = weights_shape
        
        self.gamma = gamma

        self.best_weights_path = best_weights_path

    
    def set_problem(self, A, b, tol=1e-4, iter=0):
        self.A = A
        self.iter = iter
        
        if isinstance(A, np.ndarray):
            m = A.shape[0]
            n = A.shape[1]
            if m != n:
                raise ValueError("A must be a square matrix.")

            A_in_paulis_basis, self.basis_operators, self.A_pad = matrix2paulis(A)
        else:
            A_in_paulis_basis = A
        
        self.A_in_paulis_basis = []
        for coeff, op in A_in_paulis_basis:
            if np.abs(coeff) < tol:
                continue
            self.A_in_paulis_basis.append([coeff, op])

        self.n = len(self.A_in_paulis_basis[0][1])
        b_shape = len(b)
        self.b_original_shape = b_shape
        if b_shape != 2 ** self.n:
            self.b = np.pad(b, (0, 2 ** self.n - b_shape), 'constant')
            # print('b is padded with zeros to size {}.'.format(2 ** self.n))
        else:
            self.b = np.array(b)
        

        # print('Number of qubits: ', self.n)
        # print('Number of Pauli operators: ', len(self.A_in_paulis_basis))
        

        self.b_norm = np.linalg.norm(self.b)
        self.b_normalized = torch.tensor(self.b / self.b_norm, requires_grad=False)

        # print('Normalized b: ', self.b_normalized)
        # print('Norm of b: ', self.b_norm)

        self.U = lambda : qml.QubitStateVector(self.b_normalized, wires=list(range(self.n)))

        if self.variational_block is None:
            self.variational_block = lambda weights : qml.BasicEntanglerLayers(weights, wires=list(range(self.n)))
            self.weights_shape = qml.BasicEntanglerLayers.shape(n_layers=2, n_wires=self.n)
        else:
            if self.weights_shape is None:
                # Raise an error if the variational block is provided but the weights shape is not
                raise ValueError("The weights shape must be provided if the variational block is provided.")

        if type(self.weights_shape) == int:
            weights = np.random.rand(self.weights_shape) * self.q_delta
            self.weights = torch.tensor(weights, requires_grad=True)
            # print('Total number of weights: ', self.weights.size)
        else:
            weights = np.random.rand(*self.weights_shape) * self.q_delta
            self.weights = torch.tensor(weights, requires_grad=True)
            # print('Total number of weights: ', np.prod(self.weights.shape))

        # print('Variational block: ')
        # print(self.variational_block(self.weights))

    def op_to_gate(self, Al):
        for i, op in enumerate(Al):
            if op == 'I':
                continue
            if op == 'X':
                qml.CNOT(wires=[self.n, i])
            if op == 'Y':
                qml.PhaseShift(torch.pi / 2, wires=i)
                qml.CNOT(wires=[self.n, i])
                qml.PhaseShift(-torch.pi / 2, wires=i)
            if op == 'Z':
                qml.CZ(wires=[self.n, i])

    def evaluate_norm(self, alpha):
        dev_mu = qml.device("lightning.qubit", wires=self.n+1)

        @qml.qnode(dev_mu, interface="torch")
        def hadamard_test(Al, Al_, complex=False):
            qml.Hadamard(wires=self.n)
            self.variational_block(alpha)
            if complex:
                qml.PhaseShift(-torch.pi / 2, wires=self.n)
            self.op_to_gate(Al)
            self.op_to_gate(Al_)
            qml.Hadamard(wires=self.n)

            return qml.expval(qml.PauliZ(wires=self.n))
            
        value = torch.tensor(0.0 + 0.0j)

        for l, (cl, op) in enumerate(self.A_in_paulis_basis):
            for m, (cm, oq) in enumerate(self.A_in_paulis_basis):
                coeff = cl * cm.conjugate()
                real_part = hadamard_test(op, oq)
                imag_part = hadamard_test(op, oq, complex=True)
                coeff *= (real_part + 1j * imag_part)
                value += coeff
        # Value is an array box, so we need to extract the value
        # print('\n\tNorm: ', value)
        return torch.abs(value)
        
    def evaluate_cost_local(self, alpha):
        dev_mu = qml.device("lightning.qubit", wires=self.n+1)

        @qml.qnode(dev_mu, interface="torch")
        def hadamard_test(Al, Al_, j, complex=False):
            qml.Hadamard(wires=self.n)
            self.variational_block(alpha)
            if complex:
                qml.PhaseShift(-np.pi / 2, wires=self.n)
            self.op_to_gate(Al)
            qml.adjoint(self.U)()
            qml.CZ(wires=[self.n, j])
            self.U()
            self.op_to_gate(Al_)
            qml.Hadamard(wires=self.n)
            return qml.expval(qml.PauliZ(wires=self.n))
        
        value = torch.tensor(0.0 + 0.0j)
        for j in range(0, self.n):
            for cl, op in self.A_in_paulis_basis:
                for cm, oq in self.A_in_paulis_basis:
                    coeff = cl * cm.conjugate()
                    real_part = hadamard_test(op, oq, j)
                    imag_part = hadamard_test(op, oq, j, complex=True)
                    coeff *= (real_part + 1j * imag_part)
                    value += coeff
        # print('\tLocal cost: ', value)
        return torch.abs(value)

    def cost(self, alpha):
        norm = self.evaluate_norm(alpha)
        local = self.evaluate_cost_local(alpha)
        return torch.abs(1/2 - 1/2 * local / ( self.n * norm ))
    
    def optimize(self):
        # See if best_params exists
        # print('\n\nOptimizing...\n\n')
        steps = self.steps
        if os.path.exists(self.best_weights_path + f'best_weights_{self.iter}.npy'):
            with open(self.best_weights_path + f'best_weights_{self.iter}.npy', 'rb') as f:
                params =np.load(f)
                best_weights = torch.tensor(params, requires_grad=True)
                self.weights = torch.tensor(params, requires_grad=True)
            if self.steps == 0:
                return self.weights.detach().numpy()

        else:
            if self.steps == 0:
                steps = 10
            best_weights = np.zeros(self.weights_shape)

        params = self.weights    
        
        opt = torch.optim.Adam([params], lr=self.learning_rate)
        self.history = []
        best_cost = self.cost(params)

        print("Step {:3d}/{:3d}       Cost_L = {:9.7f}".format(0,self.steps, best_cost))
        if best_cost < self.gamma:
            return params.detach().numpy()
        
        lr = self.learning_rate
        if best_cost > 0.08:
            lr = 0.1
            opt.param_groups[0]['lr'] = lr
        
        print('Initial learning rate: ', lr)

        num_iter_without_improvement = 0
        it = 0

        while best_cost > self.gamma and num_iter_without_improvement < steps:
            it += 1
            opt.zero_grad()
            loss = self.cost(self.weights)

            loss.backward()
            num_iter_without_improvement += 1

            if num_iter_without_improvement % 3 == 0:
                lr = lr / 10
                opt.param_groups[0]['lr'] = lr
            
            if loss <= best_cost:
                print('New best cost: ', loss)
                num_iter_without_improvement = 0
                best_cost = loss
                best_weights = params
                with open(self.best_weights_path + f'best_weights_{self.iter}.npy', 'wb') as f:
                    np.save(f, best_weights.detach().numpy())

            print("Step {:3d}       Cost_L = {:9.7f}".format(it, loss))
            self.history.append(loss)
            opt.step()
        
        self.weights = best_weights
        return self.weights.detach().numpy()
    
    def get_x(self, weights=None):
        weights = self.weights if weights is None else weights
        dev_x = qml.device("lightning.qubit", wires=self.n)

        @qml.qnode(dev_x)
        def prepare_and_state(weights):
            self.variational_block(weights)
            return qml.state()
        
        x = prepare_and_state(weights)

        return x
