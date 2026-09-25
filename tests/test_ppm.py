from oneseg.ppm import PpmCorrection


class FakeSdr:
    def __init__(self):
        self.ppm = 0
        self.calls = []

    @property
    def freq_correction(self):
        return self.ppm

    @freq_correction.setter
    def freq_correction(self, ppm):
        self.calls.append(ppm)
        if self.ppm == ppm:
            raise ValueError("librtlsdr returned -2 on redundant ppm write")
        self.ppm = ppm


def test_zero_ppm_skips_initial_native_call():
    sdr = FakeSdr()
    correction = PpmCorrection()
    assert correction.apply(sdr, 0) is False
    assert sdr.calls == []


def test_only_changes_ppm_and_reset_to_zero():
    sdr = FakeSdr()
    correction = PpmCorrection()
    assert correction.apply(sdr, 15)
    assert not correction.apply(sdr, 15)
    assert correction.apply(sdr, 0)
    assert not correction.apply(sdr, 0)
    assert sdr.calls == [15, 0]


def test_failed_ppm_write_does_not_poison_cache():
    sdr = FakeSdr()
    correction = PpmCorrection()
    sdr.ppm = 7
    try:
        correction.apply(sdr, 7)
    except ValueError:
        pass
    else:
        raise AssertionError("expected native setter failure")
    assert correction.current == 0
