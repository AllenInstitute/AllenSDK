import random

import numpy as np
from numpy.testing import assert_allclose
from statsmodels.sandbox.stats.multicomp import multipletests

from allensdk.brain_observatory.receptive_field_analysis.utilities \
    import holm_sidak_correction


def test_holm_sidak_correction():
    # tests rejections
    reject_none = np.full(100, 0.01)                   # tests: fail to reject any
    reject_one = np.hstack(([0], reject_none.copy()))  # tests: reject one

    assert holm_sidak_correction(reject_none)[0].sum() == 0
    assert holm_sidak_correction(reject_one)[0].sum() == 1

    # tests corrections against statsmodels
    p_values = [0]*10 + [1]*10 + [random.uniform(0, 0.1) for _ in range(80)]
    random.shuffle(p_values)

    hs_reject, hs_corrected = holm_sidak_correction(p_values)
    sm_reject, sm_corrected, _, _ = multipletests(p_values)

    # test against statsmodels
    assert all(~(hs_reject ^ sm_reject))  # XNOR
    assert_allclose(hs_corrected, sm_corrected)
