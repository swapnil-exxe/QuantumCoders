import unittest
import numpy as np

# Single-Qubit Gates
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# Controlled-NOT Gate (CNOT) for 2-Qubit Systems
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=np.complex128)

def apply_gate(gate: np.ndarray, state: np.ndarray) -> np.ndarray:
    return np.dot(gate, state)

class TestQuantumGates(unittest.TestCase):
    def test_pauli_x_bit_flip(self):
        q0 = np.array([1, 0], dtype=np.complex128)
        q1 = apply_gate(X, q0)
        self.assertTrue(np.allclose(q1, np.array([0, 1])))

    def test_hadamard_superposition(self):
        q0 = np.array([1, 0], dtype=np.complex128)
        q_super = apply_gate(H, q0)
        expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        self.assertTrue(np.allclose(q_super, expected))

    def test_cnot_entangled(self):
        q10 = np.array([0, 0, 1, 0], dtype=np.complex128)
        q11 = apply_gate(CNOT, q10)
        self.assertTrue(np.allclose(q11, np.array([0, 0, 0, 1])))

if __name__ == "__main__":
    unittest.main()
