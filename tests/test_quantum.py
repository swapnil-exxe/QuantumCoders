import pytest
import numpy as np
from quantum_gates import H, X, Y, Z, CNOT, apply_gate
from quantum_circuit import validate_state_vector, normalize_state, build_qft_matrix, S_GATE, T_GATE

def test_single_qubit_gates():
    q0 = np.array([1, 0], dtype=np.complex128)
    # X gate flips |0> to |1>
    q1 = apply_gate(X, q0)
    assert np.allclose(q1, [0, 1])

    # Z gate leaves |0> unchanged
    q_z = apply_gate(Z, q0)
    assert np.allclose(q_z, [1, 0])

def test_phase_gates():
    q1 = np.array([0, 1], dtype=np.complex128)
    q_s = apply_gate(S_GATE, q1)
    assert np.allclose(q_s, [0, 1j])

def test_state_vector_validation():
    valid_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex128)
    assert validate_state_vector(valid_state) is True

    invalid_norm = np.array([1.0, 1.0], dtype=np.complex128)
    with pytest.raises(ValueError, match="not normalized"):
        validate_state_vector(invalid_norm)

    invalid_dim = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
    with pytest.raises(ValueError, match="power of 2"):
        validate_state_vector(invalid_dim)

def test_state_normalization():
    unnormalized = np.array([3.0, 4.0], dtype=np.complex128)
    normalized = normalize_state(unnormalized)
    assert np.isclose(np.linalg.norm(normalized), 1.0)
    assert np.allclose(normalized, [0.6, 0.8])

def test_qft_unitarity():
    qft_2q = build_qft_matrix(2)
    assert qft_2q.shape == (4, 4)
    # Check unitary condition: QFT * QFT^dagger = Identity
    identity = np.dot(qft_2q, qft_2q.conj().T)
    assert np.allclose(identity, np.eye(4))
