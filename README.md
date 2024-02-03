# Quantum Power System State Estimation
This project, developed within the [Quantum Open Source Foundation Mentorship Program](https://qosf.org/qc_mentorship/), aims to shed light to the practical uses of Variational Quantum Algorithms (VQAs) in the energy sector. In particular, we tackle the problem of power system state estimation in decentralized energy networks under the [Transactive Energy (TE) framework](https://www.nist.gov/el/smart-grid-menu/hot-topics/transactive-energy-overview). In this scenario, the individual users act as both consumers and producers in the network, and the problem lies in determining the future state (i.e. bus voltages and angles) of a power system based on the data measurements by phasor measurement units (PMUs) in the different locations of a network.

## Approach
State estimation is a crucial task in power system monitoring, especially during extreme events, but obtaining measurement data is limited by the high cost of widespread PMU placement. This leads to a high computational cost when accuracy is necessary, and regular control centers become insufficient to capture the system status. Since this problem requires solving linear systems of equations, quantum computers might help alleviating some of these issues, and here we tackle this problem using a [Variational Quantum Linear Solver](https://arxiv.org/abs/1909.05820).

## Data
Most of the examples in this project were extracted from the book [_Power System State Estimation: Theory and Implementation_](https://www.researchgate.net/publication/259296629_Power_System_State_Estimation_Theory_and_Implementation) by Ali Abur and Antonio Gómez-Expósito.

## Usage
To create the Docker image and run this repository in a container, open a terminal and run:
```
docker build . -t quantum-psse
```
To run the image:
```
docker run quantum-psse
```
You should see something similar to the following:
```
Theta 1   -0.022 
Theta 2   -0.048
V 1        1.000
V 2        0.974
V 3        0.944


                   Estimated Measurements

Number of measurements: 8

----------------------------------------------------------
Measurement |  Type  | Value (pu) | h (pu) | residual (pu)
----------------------------------------------------------
1           | P_{12} |   0.888    | 0.893  |        -0.005
2           | P_{13} |   1.173    | 1.171  |         0.002
3           | P_{2}  |   -0.501   | -0.496 |        -0.005
4           | Q_{12} |   0.568    | 0.559  |         0.009
5           | Q_{13} |   0.663    | 0.668  |        -0.005
6           | Q_{2}  |   -0.286   | -0.298 |         0.012
7           |  V_1   |   1.006    |  1.0   |         0.006
8           |  V_2   |   0.968    | 0.974  |        -0.006
----------------------------------------------------------
```
