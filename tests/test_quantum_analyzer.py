import pytest
import numpy as np
from quantum_analyzer import calculate_state_fidelity, calculate_purity, analyze_quantum_state

def test_fidelity_orthogonal_and_identical():
    q0 = np.array([1, 0], dtype=np.complex128)
    q1 = np.array([0, 1], dtype=np.complex128)

    assert np.isclose(calculate_state_fidelity(q0, q0), 1.0)
    assert np.isclose(calculate_state_fidelity(q0, q1), 0.0)

def test_quantum_state_analysis():
    # Bell state (|00> + |11>) / sqrt(2)
    bell_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
    analysis = analyze_quantum_state(bell_state)

    assert analysis["n_qubits"] == 2
    assert np.isclose(analysis["purity"], 1.0)
    assert analysis["max_prob_basis"] in ["|00>", "|11>"]
    assert np.isclose(analysis["max_prob"], 0.5)
