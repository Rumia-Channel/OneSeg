from oneseg.diagnostics import explain_receiver_error


class UsbError(Exception):
    def __init__(self, errno, message):
        super().__init__(message)
        self.errno = errno


def test_minus_12_reports_correct_usb_interface():
    result = explain_receiver_error(
        UsbError(-12, "LIBUSB_ERROR_NOT_SUPPORTED (device index = 0)")
    )
    assert "WinUSB" in result
    assert "Interface 0" in result
    assert "Interface 1" in result
    assert "unrelated" in result


def test_busy_and_disconnected():
    assert "another application" in explain_receiver_error(UsbError(-6, "busy"))
    assert "Reconnect" in explain_receiver_error(UsbError(-4, "gone"))


def test_other_failure_passed_through():
    assert explain_receiver_error(ValueError("bad rate")) == "ValueError: bad rate"
