import numpy as np
from solvers.VQLS import VQLS
# from data.power_system import PowerSystem


class Solver():
    """
    Class to create a solver.
    """

    def __init__(self,
                 tol= 1e-4,
                 max_iter=10):
        
        '''
        Initializes the solver

        tol: Maximum error to achieve convergence
        max_iter: Maximum number of iterations of the solver
        '''

        self.tol = tol
        self.max_iter = max_iter

    
class WLS(Solver):
    '''
    Class for the Weighted Least Square solver
    '''

    def solve(self,
              power_system,
              x0: np.array):
        '''
        Solves the system

        power_system: PowerSystem object for which we want to estimate the state
        x0: Initial state
        '''

        for i in range(self.max_iter):
            H = power_system.get_H(x0)
            h = power_system.get_h(x0)
            r = power_system.get_residuals(h)
            G = power_system.get_G(H)

            dx = np.linalg.inv(G) @ H.T @ power_system.W @ r
            x0 = x0 + dx

            if np.all(np.abs(dx) < self.tol):
                break
        x = x0

        return x, r, G, H, h

class QWLS(Solver):
    '''
    Class for the Quantum Weighted Least Square solver
    '''
    def __init__(self,        
                variational_block=None,
                weights_shape=None,
                n_shots=10**6,
                steps=30,
                learning_rate=0.8,
                q_delta=0.001,
                gamma=0.005,
                best_weights_path='data/weights/',
                **kwargs):
        '''
        Initializes the solver
        
        parallel: Whether to use parallel optimization
        variational_block: A function that returns a quantum circuit
        weights_shape: The shape of the weights to be used in the variational block
        n_shots: The number of shots to be used in the quantum circuit
        steps: The number of steps to be used in the optimization
        learning_rate: The learning rate to be used in the optimization
        q_delta: The step size to be used in the finite difference method
        '''
        super().__init__(**kwargs)

        self.variational_block = variational_block
        self.weights_shape = weights_shape
        self.n_shots = n_shots
        self.steps = steps
        self.learning_rate = learning_rate
        self.q_delta = q_delta
        self.vqls = VQLS(variational_block=self.variational_block,
                         weights_shape=self.weights_shape,
                         n_shots=self.n_shots,
                         steps=self.steps,
                         learning_rate=self.learning_rate,
                         q_delta=self.q_delta,
                         gamma=gamma,
                         best_weights_path=best_weights_path)

    def solve(self,
              power_system,
              x0: np.array):
        '''
        Solves the system

        power_system: PowerSystem object for which we want to estimate the state
        x0: Initial state
        '''

        for i in range(self.max_iter):
            H = power_system.get_H(x0)
            h = power_system.get_h(x0)
            r = power_system.get_residuals(h)
            G = power_system.get_G(H)

            G_norm = np.linalg.norm(G)
            G_normalized = G / G_norm

            tk = H.T @ power_system.W @ r

            self.vqls.set_problem(G_normalized, tk, tol=self.tol, iter=i)

            weights = self.vqls.optimize()
            dx_res = self.vqls.get_x(weights).real[:len(tk)]

            b_new = G @ dx_res
            b_new_norm = np.linalg.norm(b_new)
            dx_correction_norm = self.vqls.b_norm / b_new_norm

            dx = dx_res * dx_correction_norm
            
            x0 = x0 + dx

            if np.all(np.abs(dx) < self.tol):
                break
        x = x0

        return x, r, G, H, h