import pytest
import numpy as np
from quantum_entropy import calculate_entanglement_of_state, partial_trace_b, calculate_von_neumann_entropy

def test_bell_state_maximal_entanglement():
    # Bell state (|00> + |11>) / sqrt(2) -> Max entanglement S = 1.0 bit
    bell_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
    entropy = calculate_entanglement_of_state(bell_state)
    assert np.isclose(entropy, 1.0, atol=1e-5)

def test_separable_state_zero_entanglement():
    # Separable state |0> (x) |0> = |00> -> Zero entanglement S = 0.0 bit
    separable = np.array([1, 0, 0, 0], dtype=np.complex128)
    entropy = calculate_entanglement_of_state(separable)
    assert np.isclose(entropy, 0.0, atol=1e-5)
