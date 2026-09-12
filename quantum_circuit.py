import numpy as np

def validate_state_vector(state: np.ndarray, atol: float = 1e-6) -> bool:
    """
    Validates that state vector length is a power of 2 and has unit probability norm.
    """
    if not isinstance(state, np.ndarray):
        raise TypeError("State vector must be a numpy ndarray.")
    
    n = len(state)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"State vector length ({n}) must be a power of 2.")

    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0, atol=atol):
        raise ValueError(f"State vector probability norm ({norm:.6f}) is not normalized to 1.0.")

    return True

def normalize_state(state: np.ndarray) -> np.ndarray:
    """Normalizes a non-zero state vector to unit probability norm."""
    norm = np.linalg.norm(state)
    if np.isclose(norm, 0):
        raise ValueError("Cannot normalize zero vector.")
    return state / norm

def build_qft_matrix(n_qubits: int) -> np.ndarray:
    """
    Generates Quantum Fourier Transform (QFT) unitary matrix for an n-qubit system.
    """
    N = 2 ** n_qubits
    omega = np.exp(2j * np.pi / N)
    j_idx, k_idx = np.meshgrid(np.arange(N), np.arange(N))
    qft = (1.0 / np.sqrt(N)) * (omega ** (j_idx * k_idx))
    return qft.astype(np.complex128)

# Single Qubit Phase Gates
S_GATE = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
