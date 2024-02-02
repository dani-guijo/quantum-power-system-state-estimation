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
You should see the following:
```

                        Measurements

Number of measurements: 8

----------------------------------------------
Measurement |  Type  | Value (pu) | Sigma (pu)
----------------------------------------------
1           | P_{12} |   0.888    |      0.008
2           | P_{13} |   1.173    |      0.008
3           | P_{2}  |   -0.501   |       0.01
4           | Q_{12} |   0.568    |      0.008
5           | Q_{13} |   0.663    |      0.008
6           | Q_{2}  |   -0.286   |       0.01
7           |  V_1   |   1.006    |      0.004
8           |  V_2   |   0.968    |      0.004
----------------------------------------------
Resistance Matrix:

0.0    0.01  0.02
0.01   0.0   0.03
0.02   0.03   0.0

Reactance Matrix:

0.0    0.03  0.05
0.03   0.0   0.08
0.05   0.08   0.0

Admitance Matrix:

0j                      (10-30j)       (6.8966-17.2414j)
(10-30j)                   0j          (4.1096-10.9589j)
(6.8966-17.2414j)   (4.1096-10.9589j)                 0j

Bus Admitance Matrix:

(33.7931-94.4828j)       (-10+30j)       (-6.8966+17.2414j)
(-10+30j)            (28.2192-81.9178j)  (-4.1096+10.9589j)
(-6.8966+17.2414j)   (-4.1096+10.9589j)  (22.0123-56.4006j)
```
