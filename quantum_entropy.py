import numpy as np

def partial_trace_b(density_matrix_4x4: np.ndarray) -> np.ndarray:
    """
    Computes partial trace over subsystem B for a 2-qubit (4x4) density matrix rho_AB.
    Returns 2x2 reduced density matrix rho_A.
    """
    if density_matrix_4x4.shape != (4, 4):
        raise ValueError("Density matrix must be 4x4 for 2-qubit system.")

    rho = density_matrix_4x4.reshape(2, 2, 2, 2)
    # Trace out second qubit (subsystem B: axis 1 and 3)
    rho_a = np.trace(rho, axis1=1, axis2=3)
    return rho_a

def calculate_von_neumann_entropy(reduced_rho: np.ndarray) -> float:
    """
    Calculates Von Neumann Entanglement Entropy S(rho_A) = -Tr(rho_A log2(rho_A)).
    Max entanglement for 2-qubit state returns S = 1.0 bit.
    Separable pure state returns S = 0.0 bit.
    """
    eigenvalues = np.linalg.eigvalsh(reduced_rho)
    # Filter out non-positive eigenvalues for numerical stability
    positive_evals = eigenvalues[eigenvalues > 1e-12]

    if len(positive_evals) == 0:
        return 0.0

    entropy = -np.sum(positive_evals * np.log2(positive_evals))
    return float(np.abs(entropy))

def calculate_entanglement_of_state(state_vector: np.ndarray) -> float:
    """
    Computes entanglement entropy for a 2-qubit state vector |psi>.
    """
    if len(state_vector) != 4:
        raise ValueError("State vector must have length 4 (2 qubits).")

    density_matrix = np.outer(state_vector, np.conj(state_vector))
    rho_a = partial_trace_b(density_matrix)
    return calculate_von_neumann_entropy(rho_a)
