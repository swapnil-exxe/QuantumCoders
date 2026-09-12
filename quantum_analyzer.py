import numpy as np

def calculate_state_fidelity(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """
    Calculates quantum state fidelity F = |<psi|phi>|^2 between two pure state vectors.
    """
    if len(state_a) != len(state_b):
        raise ValueError("State vectors must have matching dimensions.")

    inner_product = np.vdot(state_a, state_b)
    return float(np.abs(inner_product) ** 2)

def calculate_purity(density_matrix: np.ndarray) -> float:
    """
    Calculates quantum state purity Tr(rho^2).
    Pure states have purity = 1.0, mixed states have purity < 1.0.
    """
    rho_squared = np.dot(density_matrix, density_matrix)
    return float(np.real(np.trace(rho_squared)))

def analyze_quantum_state(state: np.ndarray) -> dict:
    """
    Analyzes a quantum state vector to extract dimension, norm, purity, and max probability basis.
    """
    dim = len(state)
    probabilities = np.abs(state) ** 2
    max_prob_idx = int(np.argmax(probabilities))

    density_matrix = np.outer(state, np.conj(state))
    purity = calculate_purity(density_matrix)

    return {
        "dimension": dim,
        "n_qubits": int(np.log2(dim)),
        "purity": purity,
        "max_prob_basis": f"|{max_prob_idx:0{int(np.log2(dim))}b}>",
        "max_prob": float(probabilities[max_prob_idx])
    }
