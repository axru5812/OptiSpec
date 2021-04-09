import pandas as pd
import numpy as np
from optispec.fitter import Fitter


def test_read_line_list():
    fitter = Fitter(0.2)

    df = fitter._read_line_list()
    assert df.wl.values[0] == 6564.61
    assert df.line.values[0] == 'Ha'