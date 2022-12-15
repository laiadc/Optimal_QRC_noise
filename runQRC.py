import numpy as np
import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from qiskit import *
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector
from scipy.stats import unitary_group

from qiskit.quantum_info import Pauli
from qiskit.opflow import *
from qiskit.circuit.library import Diagonal
from qiskit.extensions import  UnitaryGate
from qiskit.providers.aer.noise import NoiseModel
from qiskit.quantum_info import Statevector, state_fidelity
import qiskit.providers.aer.noise as noise
from qiskit import  Aer
from qiskit.test.mock import FakeProvider
from qiskit.test.mock import *
from qiskit.quantum_info import Statevector, state_fidelity
from numpy.lib.scimath import sqrt as csqrt
from scipy.stats import unitary_group
import itertools
from qiskit.extensions import UnitaryGate

import random
import sys

class QuantumCircQiskit:
    def __init__(self, gates_name, num_gates=50,nqbits=8,observables_type = 'fidelity',
                 err_type='depolarizing', err_p1=0.001, err_p2=0.01, err_idle=0.00001):
        
        self.num_gates = num_gates
        self.gates_name = gates_name
        self.observables_type = observables_type
        self.gates_set = []
        self.qubits_set = []
        self.nqbits=nqbits
        self.coupling_map=None
        self.err_type = err_type
        if err_type is not None:
            noise_model, basis_gates = self.get_noise_model(err_type=err_type, p1=err_p1, p2=err_p2, p_idle=err_idle)
        else:
            noise_model, basis_gates = None, None
            
        self.noise_model = noise_model
        self.basis_gates = basis_gates
        if self.gates_name=='G1':
            gates = ['CNOT', 'H', 'X']
        if self.gates_name=='G2':
            gates = ['CNOT', 'H', 'S']
        if self.gates_name=='G3':
            gates = ['CNOT', 'H', 'T']  
        if self.gates_name=='HT':
            gates = ['H', 'T']  
        if self.gates_name=='Toffoli':
            gates = ['CCX'] 


        qubit_idx = list(range(self.nqbits))
        # Store gates
        if self.gates_name in ['G1', 'G2', 'G3', 'Toffoli']:
            for i in range(self.num_gates):
                # Select random gate
                gate = random.sample(gates,1)[0] 
                self.gates_set.append(gate)
                if gate=='CNOT':
                    # Select qubit 1 and 2 (different qubits)
                    qbit1 = random.sample(qubit_idx,1)[0]
                    qubit_idx2 = qubit_idx.copy()
                    qubit_idx2.remove(qbit1)
                    qbit2 = random.sample(qubit_idx2,1)[0]
                    self.qubits_set.append([qbit1, qbit2])
                elif gate=='CCX':
                    # Select qubit 1, 2 and 3 (different qubits)
                    qbit1 = random.sample(qubit_idx,1)[0]
                    qubit_idx2 = qubit_idx.copy()
                    qubit_idx2.remove(qbit1)
                    qbit2 = random.sample(qubit_idx2,1)[0]
                    qubit_idx3 = qubit_idx2.copy()
                    qubit_idx3.remove(qbit2)
                    qbit3 = random.sample(qubit_idx3,1)[0]
                    self.qubits_set.append([qbit1, qbit2, qbit3])
                else:
                    # Select qubit
                    qbit = random.sample(qubit_idx,1)[0]
                    self.qubits_set.append([qbit])
        elif self.gates_name=='D2':
            qubit_idx = list(range(self.nqbits))
            self.qubits_set = list(itertools.combinations(qubit_idx, 2))
            self.phis = np.random.uniform(0, 2*np.pi, size=(len(self.qubits_set), 2**2))
        elif self.gates_name=='D3':
            qubit_idx = list(range(self.nqbits))
            self.qubits_set = list(itertools.combinations(qubit_idx, 3))
            self.phis = np.random.uniform(0, 2*np.pi, size=(len(self.qubits_set), 2**3))
        elif self.gates_name=='Dn':
            self.phis = np.random.uniform(0, 2*np.pi, size=(2**self.nqbits))
        elif self.gates_name=='MG':
            for i in range(self.num_gates):
                G = self.matchgate()
                self.gates_set.append(G)
                qbit1 = random.sample(qubit_idx,1)[0]
                qubit_idx2 = qubit_idx.copy()
                qubit_idx2.remove(qbit1)
                qbit2 = random.sample(qubit_idx2,1)[0]
                self.qubits_set.append([qbit1, qbit2])

                
    def initialization(self, initial_state):
        # 1. INITIALIZATION
        # Define initial state
        initial_state = initial_state.round(6)
        initial_state/=np.sqrt(np.sum(initial_state**2))

        # Define qiskit circuit to initialize quantum state
        self.nqbits = int(np.log2(initial_state.shape[0]))
        qc_initial = QuantumCircuit(self.nqbits)
        qc_initial.initialize(initial_state, list(range(self.nqbits)))
        
        aer_sim = Aer.get_backend('unitary_simulator')
        job = aer_sim.run(transpile(qc_initial, aer_sim))
        U = job.result().get_unitary()
        U = UnitaryGate(U, label='unitary')

        return U

    def apply_G_gates(self, qc):
        # Apply random gates to random qubits
        for i in range(self.num_gates):
            # Select random gate
            # Select random gate
            gate = self.gates_set[i]
            if gate=='CNOT': # For 2-qubit gates
                # Select qubit 1 and 2 (different qubits)
                qbit1, qbit2 = self.qubits_set[i]
                # Apply gate to qubits
                qc.cx(qbit1, qbit2) 
                # Appply identity operator to all idle qubits if we use an ide noise model
                if self.err_type=='depolarizing_idle' or self.err_type=='amplitude_damping_idle' or self.err_type=='phase_damping_idle':
                    qubit_idx = list(range(self.nqbits))
                    qubit_idx.remove(qbit1)
                    qubit_idx.remove(qbit2)
                    # Apply identity gates to other gates
                    for qbit in qubit_idx:
                        qc.id(qbit)
            if gate=='CCX':
                # Select qubit 1, 2 and 3 (different qubits)
                qbit1, qbit2, qbit3 = self.qubits_set[i]
                # Apply gate to qubits
                qc.ccx(qbit1, qbit2, qbit3) 
                if self.err_type=='depolarizing_idle' or self.err_type=='amplitude_damping_idle' or self.err_type=='phase_damping_idle':
                    qubit_idx = list(range(self.nqbits))
                    qubit_idx.remove(qbit1)
                    qubit_idx.remove(qbit2)
                    qubit_idx.remove(qbit3)
                    # Apply identity gates to other gates
                    for qbit in qubit_idx:
                        qc.id(qbit)
            else: # For 1-qubit gates
                # Select qubit
                qbit = self.qubits_set[i][0]
                if gate=='X':# Apply gate
                    qc.x(qbit) 
                if gate=='S':
                    qc.s(qbit) 
                if gate=='H':
                    qc.h(qbit) 
                if gate=='T':
                    qc.t(qbit) 
                # Appply identity operator to all idle qubits if we use an ide noise model
                if self.err_type=='depolarizing_idle' or self.err_type=='amplitude_damping_idle' or self.err_type=='phase_damping_idle':
                    qubit_idx = list(range(self.nqbits))
                    qubit_idx.remove(qbit)
                    # Apply identity gates to other gates
                    for qbit in qubit_idx:
                        qc.id(qbit)
                
    
    def apply_matchgates(self, qc):
        for i in range(self.num_gates):
            gate = self.gates_set[i]
            qbit1, qbit2 = self.qubits_set[i]
            qc.unitary(gate, [qbit1, qbit2], label='MG')
            # Appply identity operator to all idle qubits if we use an ide noise model
            if self.err_type=='depolarizing_idle' or self.err_type=='amplitude_damping_idle' or self.err_type=='phase_damping_idle':
                qubit_idx = list(range(self.nqbits))
                qubit_idx.remove(qbit1)
                qubit_idx.remove(qbit2)
                # Apply identity gates to other gates
                for qbit in qubit_idx:
                    qc.id(qbit)
            
    def matchgate(self):
        A = unitary_group.rvs(2)
        B = unitary_group.rvs(2)
        detA = np.linalg.det(A)
        detB = np.linalg.det(B)
        B = B/np.sqrt(detB)*np.sqrt(detA)
        G = np.array([[A[0,0],0,0,A[0,1]],[0,B[0,0], B[0,1],0],
                      [0,B[1,0],B[1,1],0],[A[1,0],0,0,A[1,1]]])
        return G
    
    def apply_Dn(self, qc):
        # Apply Dn gate
        diagonals = np.exp(1j*self.phis)
        qc += Diagonal(diagonals)
        
    def apply_D2(self, qc):
        i=0
        for pair in self.qubits_set:
            # Apply D2 gate
            diagonals = np.diag(np.exp(1j*self.phis[i]))
            D2 = UnitaryGate(diagonals)
            qc.append(D2, [pair[0], pair[1]])
            i+=1
            # Appply identity operator to all idle qubits if we use an ide noise model
            if self.err_type=='depolarizing_idle' or self.err_type=='amplitude_damping_idle' or self.err_type=='phase_damping_idle':
                qubit_idx = list(range(self.nqbits))
                qubit_idx.remove(pair[0])
                qubit_idx.remove(pair[1])
                # Apply identity gates to other gates
                for qbit in qubit_idx:
                    qc.id(qbit)
            
    def apply_D3(self, qc):
        i=0
        for pair in self.qubits_set:
            # Apply D3 gate
            diagonals = np.diag(np.exp(1j*self.phis[i]))
            D3 = UnitaryGate(diagonals)
            qc.append(D3, [pair[0], pair[1], pair[2]])
            i+=1
            # Appply identity operator to all idle qubits if we use an ide noise model
            if self.err_type=='depolarizing_idle' or self.err_type=='amplitude_damping_idle' or err_type=='phase_damping_idle':
                qubit_idx = list(range(self.nqbits))
                qubit_idx.remove(pair[0])
                qubit_idx.remove(pair[1])
                qubit_idx.remove(pair[2])
                # Apply identity gates to other gates
                for qbit in qubit_idx:
                    qc.id(qbit)
                    
    def get_noise_model(self, err_type='depolarizing', p1=0.001, p2=0.01, p_idle=0.0001):
        # Error probabilities: p1=1-qubit gate, p2=2-qubit gate
        if err_type=='depolarizing' or err_type=='depolarizing_idle':
            # Depolarizing quantum errors
            error_1 = noise.depolarizing_error(p1, 1)
            error_2 = noise.depolarizing_error(p2, 2)
            error_idle = noise.depolarizing_error(p_idle, 1)
        elif err_type=='phase_damping' or err_type=='phase_damping_idle':
            # Depolarizing quantum errors
            error_1 = noise.phase_damping_error(p1, 1)
            error_2 = noise.phase_damping_error(p2, 2)
            error_2 = error_1.tensor(error_2)
            error_idle = noise.phase_damping_error(p_idle, 1)
        elif err_type=='amplitude_damping' or err_type=='amplitude_damping_idle':
            # Construct the error
            error_1 = noise.amplitude_damping_error(p1)
            error_2 = noise.amplitude_damping_error(p2)
            error_2 = error_1.tensor(error_2)
            error_idle = noise.amplitude_damping_error(p_idle, 1)
        elif err_type=='fake':
            provider = FakeProvider()
            names = [ b.name() for b in provider.backends() if b.configuration().n_qubits >= self.nqbits]
            if len(names)==0:
                raise ValueError('Error type not supported')
            fake = provider.get_backend(names[7])
            # Get coupling map from backend
            coupling_map = fake.configuration().coupling_map
            cmap = [[self.nqbits-1,1]]
            for i,j in coupling_map:
                if i<nqbits and j<nqbits:
                    cmap.append([i,j]) 
            self.coupling_map = cmap
            noise_model = NoiseModel.from_backend(fake)
            basis_gates = noise_model.basis_gates
            return noise_model, basis_gates
        else:
            raise ValueError('Error type not supported', err_type)
        # Add errors to noise model
        noise_model = noise.NoiseModel()
        
        noise_model.add_all_qubit_quantum_error(error_1, ['x','h','ry','rz','u1', 'u2', 'u3'])
        noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
        noise_model.add_basis_gates('unitary')
        if err_type=='depolarizing_idle' or err_type=='amplitude_damping_idle' or err_type=='phase_damping_idle':
            noise_model.add_all_qubit_quantum_error(error_idle, ['id'])

        # Get basis gates from noise model
        basis_gates = noise_model.basis_gates
        return noise_model, basis_gates
        
    def get_observables(self):
        observables = []
        name_gate=''
        for i in range(self.nqbits):
            name_gate+= 'I' 
        for i in range(self.nqbits):
            # X
            op_nameX = name_gate[:i] + 'X' + name_gate[(i+1):]
            obs = PauliOp(Pauli(op_nameX))
            observables.append(obs)
            # Y
            op_nameY = name_gate[:i] + 'Y' + name_gate[(i+1):]
            obs = PauliOp(Pauli(op_nameY))
            observables.append(obs)
            # Z
            op_nameZ = name_gate[:i] + 'Z' + name_gate[(i+1):]
            obs = PauliOp(Pauli(op_nameZ))
            observables.append(obs)
        return observables

    def run_circuit(self, initial_state):

        # 1. INITIALIZATION
        U = self.initialization(initial_state)
        
        qc =  QuantumCircuit(self.nqbits)
        qc.append(U, list(range(self.nqbits)))
        # 2. DEFINE RANDOM CIRCUIT
        if self.gates_name in ['G1', 'G2', 'G3', 'Toffoli']:
            self.apply_G_gates(qc)
        elif self.gates_name=='D2':
            self.apply_D2(qc)
        elif self.gates_name=='D3':
            self.apply_D3(qc)
        elif self.gates_name=='Dn':
            self.apply_Dn(qc)
        elif self.gates_name=='MG':
            self.apply_matchgates(qc)
        else:
            print('Unknown gate')

        # 3. DEFINE OBSERVABLES
        # Define observables to measure
        if self.observables_type=='single' or self.observables_type=='all':
            observables = self.get_observables()

        # 4. RUN CIRCUIT
        results = []
        results_noise = []
        
        backend = Aer.get_backend('statevector_simulator')
        if self.noise_model is not None:
            # Perform a noisy simulation
            if self.err_type=='depolarizing_idle' or self.err_type=='amplitude_damping_idle' or self.err_type=='phase_damping_idle':
                optimization_level=0
            else:
                optimization_level=1
            qc_state = execute(qc, backend,
                             basis_gates=self.basis_gates,
                             coupling_map = self.coupling_map,
                             noise_model=self.noise_model, optimization_level=optimization_level).result().get_statevector(qc)
            
        else:
            job = backend.run(transpile(qc, backend))
            qc_state = job.result().get_statevector(qc)
        
        if self.observables_type=='fidelity':
            backend = Aer.get_backend('statevector_simulator')
            job = backend.run(transpile(qc, backend))
            qc_state_noiseless = job.result().get_statevector(qc)
            state_noise = Statevector(qc_state)
            state_noiseless = Statevector(qc_state_noiseless)
            fidelity = state_fidelity(state_noise,state_noiseless)
            return np.array(state_noise), np.array(state_noiseless), fidelity
        
        if self.observables_type=='all':
            backend = Aer.get_backend('statevector_simulator')
            job = backend.run(transpile(qc, backend))
            qc_state_noiseless = job.result().get_statevector(qc)
            state_noise = Statevector(qc_state)
            state_noiseless = Statevector(qc_state_noiseless)
            fidelity = state_fidelity(state_noise,state_noiseless)
            for obs in observables:
                obs_mat = obs.to_spmatrix()
                expect = np.inner(np.conjugate(state_noise), obs_mat.dot(state_noise)).real
                results_noise.append(expect)
                expect = np.inner(np.conjugate(state_noiseless), obs_mat.dot(state_noiseless)).real
                results.append(expect)
            return np.array(state_noise), np.array(state_noiseless), fidelity, np.array(results), np.array(results_noise)
        
        if self.observables_type=='single':
            for obs in observables:
                obs_mat = obs.to_spmatrix()
                expect = np.inner(np.conjugate(qc_state), obs_mat.dot(qc_state)).real
                results.append(expect)

            return np.array(results)
        else:
            return qc_state
        
        
# Read user argument (number of gates and gates set)
if len(sys.argv)!=7:
    raise ValueError('Incorrect number of arguments: ', len(sys.argv))
else:
    num_gates = int(sys.argv[1])   
    gate_set = str(sys.argv[2])
    observables_type = str(sys.argv[3])
    err_type=str(sys.argv[4])
    err_p1=float(sys.argv[5])
    err_p2 = float(sys.argv[6])

print('Num gates: ', num_gates, ' gate_set: ', gate_set, ' observables_type:', observables_type,
      ' err_type: ', err_type, ' err_p1: ', err_p1, ' err_p2: ', err_p2  )

# Read data
with open('ground_states_LiH.npy', 'rb') as f:
        ground_states = np.load(f)

for j in range(100):
    # Run circuit for all values of ground states:
    
    num_states =ground_states.shape[0]
    qc = QuantumCircQiskit(gate_set, num_gates=num_gates,nqbits=8,observables_type = observables_type,
                      err_type=err_type, err_p1=err_p1, err_p2=err_p2)
    if observables_type=='single':
        obs_res = []
        for i in range(num_states):
            res = qc.run_circuit(ground_states[i])
            obs_res.append(res)
        obs_res = np.array(obs_res)    
    elif observables_type=='fidelity':
        state_noise_list, state_noiseless_list, fidelity_list = [],[],[]
        for i in range(num_states):
            state_noise, state_noiseless, fidelity = qc.run_circuit(ground_states[i])
            state_noise_list.append(state_noise)
            state_noiseless_list.append(state_noiseless)
            fidelity_list.append(fidelity)
    elif observables_type=='all':
        fidelity_list, obs_res, obs_noise = [],[],[]
        for i in range(num_states):
            _, _, fidelity, res, res_noise = qc.run_circuit(ground_states[i])
            fidelity_list.append(fidelity)
            obs_res.append(res)
            obs_noise.append(res_noise)
        obs_res = np.array(obs_res) 
        obs_noise = np.array(obs_noise)


    # Store results
    if observables_type=='single':
        rnd = random.randint(0,9999999)
        filename = 'obs_LiH_' + str(err_type) +'_'+ str(err_p1) +'_'+ str(err_p2) + '_' + str(gate_set) + '_' + str(num_gates)+'rand'+ str(rnd) + '.npy'
        with open(filename, 'wb') as f:
            np.save(f, obs_res, allow_pickle=True)
    

    elif observables_type=='fidelity':
        rnd = random.randint(0,9999999)
        filename = 'obs_LiH_' + str(err_type) +'_'+ str(err_p1) +'_'+ str(err_p2) + '_' + str(gate_set) + '_' + str(num_gates)+ str(observables_type)+'rand'+ str(rnd) + '.npy'
        result={
            'fidelity_list':fidelity_list
        }
        with open(filename, 'wb') as f:
            np.save(f, result, allow_pickle=True)

    elif observables_type=='all':
        rnd = random.randint(0,9999999)
        filename = 'obs_LiH_' + str(err_type) +'_'+ str(err_p1) +'_'+ str(err_p2) + '_' + str(gate_set) + '_' + str(num_gates)+'_'+ str(observables_type)+'_rand'+ str(rnd) + '.npy'
        result={
            'fidelity_list':fidelity_list,
            'observables':obs_res,
            'observables_noise':obs_noise
        }
        with open(filename, 'wb') as f:
            np.save(f, result, allow_pickle=True)
