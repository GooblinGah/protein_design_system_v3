
from safety.structure import _run_tmalign
def test_tmalign_missing_ok():
    assert _run_tmalign("no.pdb","no.pdb") is None
