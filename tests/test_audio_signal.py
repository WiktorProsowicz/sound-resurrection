# -*- coding: utf-8 -*-
import numpy as np
import pytest
from preprocessing import audio_signal


@pytest.mark.xfail(raises=ValueError)
def test_making_audio_signal_with_wrong_dimensionality():

    meta_data = audio_signal.AudioMeta(100, 1, 16)
    data = np.ndarray(shape=(2, 3, 4), dtype=np.uint16)

    audio_signal.AudioSignal(data, meta_data)
