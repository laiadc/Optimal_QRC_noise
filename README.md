# Taking advantage of noise in quantum reservoir computing

In this Letter we study how the presence of noise in quantum devices affects the performance of quantum reservoir computing. The presence of noise in the quantum devices is considered one of the biggest challenges of quantum computing. In this work we demonstrate that, in certain situations, certain noise models can actually be beneficial for the quantum machine learning algorithm.

The results are illustrated in a set of [Jupyter](https://jupyter.org/) notebooks for easier comprehensiopn. We also provide a python script to run the experiments. The whole code is the result of the work in  this paper. Any contribution or idea to continue the lines of the proposed work will be very welcome.

We study the effect of three noise models with different error probabilities. The first error model is the *amplitude damping channel*, which models the effect of energy dissipation, that is, the loss of energy of a quantum state to its environment. It provides a model of the decay of an excited two-level atom due to the spontaneous emission of a photon with probability p.  The second noise model is described by the *phase damping channel*, which models the loss of quantum information without loss of energy. The last error model is described by the *depolarizing channel*. In this case, a Pauli error $X$, $Y$ or $Z$ occurs with the same probability $p$.

The optimality of the quantum reservoir is illustrated by solving a quantum chemistry problem. In this context, the data used to train the QRC model is already a quantum state. Therefore, it is natural to use a QML algorithm to infer the properties of the system.

## Notebooks

All the notebooks used for this work can be found inside the folder **notebooks** .

### [Generate_data.ipynb](https://github.com/laiadc/Optimal_QRC_noise/blob/main/notebooks/Generate_data.ipynb) 
Generates the training data for the quantum reservoir computing task. The electronic Hamiltonian is mapped to the qubit space and the ground states and excited energies are obtained by direct diagonalization.

### [quantumRC.ipynb](https://github.com/laiadc/Optimal_QRC_noise/blob/main/notebooks/quantumRC.ipynb) 
Illustrates how to design the noisy quantum reservoir for the three error models studied in this work. It also illustrates how to train the quantum machine learning model.

### [Results.ipynb](https://github.com/laiadc/Optimal_QRC_noise/blob/main/notebooks/Results.ipynb) 
This notebook analyzes the outputs of the experiments to draw the corresponding conclusions.

### [Figures.ipynb](https://github.com/laiadc/Optimal_QRC_noise/blob/main/notebooks/Figures.ipynb) 
This notebook provides the figures and tables summarizing the results of the experiments. 

## Scripts

Additionally, we provide a script that can be used to simulate the quantum reservoirs with certain noise models.. The script `runQRC` can be used to run the simulations for the LiH molecule. To run them, just type:

`python runQC.py num_gates gates_set observable_type `

where 

+ `num_gates` is the number of gates (in this work we have used 20, 50, 100, 150, 200)
+ `gates_set` is the name of the gate set, from the list [G1, G2, G3, MG, D2, D3, Dn]
+ `observable_type` is either *single* which only returns the expected values, *fidelity* which returns the state fidelities or *all*, which returns the expected values, fidelitites and final states
+ `error_type` must be *amplitude_damping*, *depolarizing*, *phase_damping* or *fake*, which corresponds to a fake provider
+ `p1` is the error probability of qubit 1
+ `p2` is the error probability of qubit 2

## Contributions

Contributions are welcome!  For bug reports or requests please [submit an issue](https://github.com/laiadc/Optimal_QRC_noise/issues).

## Contact  

Feel free to contact me to discuss any issues, questions or comments.

* GitHub: [laiadc](https://github.com/laiadc)
* Email: [laia.domingo@icmat.es](laia.domingo@icmat.es)

### BibTex reference format for citation for the Code
```
@misc{QRCDomingo,
title={The advantage of noise in quantum reservoir computing},
url={https://github.com/laiadc/Optimal_QRC_noise},
note={GitHub repository containing a a study of the effect of different noise models in quantum reservoir computing.},
author={Laia Domingo Colomer},
  year={2022}
}
```

