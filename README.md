# digital-quantum-simulation

## Description
 Methods to simulate quantum many-body systems on gate-based quantum computers:

 1. Generic state preparation methods.

    a. Arbitrary states via Shende-Bullock-Markov and Plesch-Brukner methods.

    b. Matrix product states with open (deterministic) and periodic (probabilistic) boundary conditions.

 2. Unitary decompositions, including qubit connectivity constraints.

    a. Permutations of qubits. 

    b. CNOT, Toffoli, and Fredkin under linear qubit connectivity with separated active qubits.

    c. Multiplexor Ry and Rz gates.

    d. Conversion of multi-spin-1/2 states into fermionic versions at half-filling. 
 
 4. Bespoke state preparation methods.

    a. Quantum information (e.g., GHZ states, W states).

    b. Quantum spin models (e.g., AKLT states, spin waves, RVB approximations of ground states of spin-1/2 Heisenberg model).

    c. Lattice models of electrons (e.g., Slater determinants, Gutzwiller wave function).

## Organization
For the sake of clarity, the *notebooks* folder includes Jupyter notebooks that present examples of the usage of the methods. The respective code scripts are in the *scripts* folder. 
 
