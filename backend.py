from data.examples import create_example
from solvers.solvers import WLS, QWLS


if __name__ == "__main__":
    
    power_system = create_example(n_buses=3)

    print('\n' + '-'*65)
    print('Classical WLS'.center(65))
    print('-'*65 + '\n')
    wls = WLS()

    x, r, G, H, h = power_system.estimate_state(solver=wls)

    print('\n' + '-'*65)
    print('Quantum WLS'.center(65))
    print('-'*65 + '\n')
    # No optimization steps, use the best weights saved in the best_weights_path
    # The tol was set to 6.2e-4 in the QWLS class due to the fact that the best weights were already saved under this tolerance
    qwls = QWLS(steps=0, tol=6.2e-4)

    x, r, G, H, h = power_system.estimate_state(solver=qwls)