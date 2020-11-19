# CFD_Euler_1D
One-dimensional solver for the compressible Euler equations, written in Python.
Multiple Riemann solvers (HLLC, Flux Vector Splitting) are implemented, based on [1]. The spatial scheme can be of order 1 (piecewise constant) or 2 (MUSCL-Hancok scheme with linear reconstruction). A minmod limiter is also avaible.

The test script implements the classical Sod shock tube problem, with reflective or transmissive boundaries.

Here is a comparison of the density field at t=0.01s, for the analytical solution and for the simulation with 100 cells, order 2 reconstruction and Flux Vector Splitting:
![comparison](https://raw.githubusercontent.com/laurent90git/CFD_Euler_1D/main/doc/comparison_density_FVS_r1l1_small.png)

Some fancy visualizations can be done with this code, for example the following wave diagrams, for a shock tube with reflective boundaries:
![schlieren](https://raw.githubusercontent.com/laurent90git/CFD_Euler_1D/main/doc/schlieren_mode2r1l1_bright_small.png)
![T_field](https://raw.githubusercontent.com/laurent90git/CFD_Euler_1D/main/doc/T_mode2r1l1_small.png)

References:
  [1] "Riemann Solvers and Numerical Methods for Fluid Dynamics, A Practical Introduction" by Toro, Eleuterio F., Springer

