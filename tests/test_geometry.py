
import numpy as np
from safety.structure import kabsch
def test_kabsch_zero_rmsd():
    P = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
    Q = P.copy()
    assert abs(kabsch(P,Q)) < 1e-6
