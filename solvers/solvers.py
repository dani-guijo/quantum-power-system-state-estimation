import numpy as np
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
              power_system,#: PowerSystem,
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
