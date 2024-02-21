from data.examples import create_example
from solvers.solvers import WLS


if __name__ == "__main__":
    
    power_system = create_example(n_buses=3)

    wls = WLS()

    x, r, G, H, h = power_system.estimate_state(solver=wls)
