"""Explain RTL-SDR USB startup failures without changing Windows drivers."""

from __future__ import annotations

DRIVER_HELP = (
    "Windows can see a compatible RTL-SDR, but libusb cannot open it. "
    "For LIBUSB_ERROR_NOT_SUPPORTED (-12), check that the correct "
    "DS-DT308SV / RTL2832U USB interface is bound to WinUSB.\n\n"
    "1. Close this application, SDR++, SDR#, and other tuner applications.\n"
    "2. Unplug and reconnect the tuner; if possible, try a USB 2.0 port.\n"
    "3. In Zadig (Options → List All Devices), identify the tuner by its "
    "USB ID and unplug/replug behavior. On common RTL2832U devices the "
    "interface is Bulk-In, Interface 0, or named RTL2832U/RTL2838; "
    "DO NOT accidentally select Interface 1 or an unrelated device.\n"
    "4. Verify that the correct interface uses WinUSB; install or replace "
    "its driver only after checking its hardware ID.\n"
    "5. Retry with: uv run oneseg-doctor\n\n"
    "Do not change mouse, keyboard, USB hub, or unrelated drivers. "
    "Restoring the vendor TV driver may be necessary to use CHUSEI PVR."
)


def explain_receiver_error(exception: BaseException) -> str:
    message = f"{type(exception).__name__}: {exception}"
    if getattr(exception, "errno", None) == -12 or (
        "LIBUSB_ERROR_NOT_SUPPORTED" in message or "libusb_open error -12" in message
    ):
        return message + "\n\n" + DRIVER_HELP
    if getattr(exception, "errno", None) == -6 or "LIBUSB_ERROR_BUSY" in message:
        return (
            message
            + "\n\nThis device may be in use by another application. "
            "Close other RTL-SDR programs, unplug/replug and retry."
        )
    if getattr(exception, "errno", None) == -4 or "LIBUSB_ERROR_NO_DEVICE" in message:
        return message + "\n\nThe tuner disconnected. Reconnect it and retry."
    return message


def probe() -> int:
    """Read-only probe. Returns a useful shell exit status; no Zadig automation."""
    try:
        from rtlsdr import RtlSdr
        from rtlsdr.librtlsdr import librtlsdr

        count = int(librtlsdr.rtlsdr_get_device_count())
        print(f"RTL-SDR compatible devices detected: {count}")
        for index in range(count):
            name = librtlsdr.rtlsdr_get_device_name(index)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            print(f"  [{index}] {name}")
        if count < 1:
            print("No RTL-SDR found. Check the USB connection.")
            return 2
        device = RtlSdr(device_index=0)
        try:
            print(
                "Device 0 opened successfully; "
                f"sample rate={device.sample_rate}, "
                f"center frequency={device.center_freq}."
            )
        finally:
            device.close()
        return 0
    except Exception as exc:
        print(explain_receiver_error(exc))
        return 2


def main() -> int:
    import argparse

    argparse.ArgumentParser(
        description="Read-only RTL-SDR USB driver and device-open diagnostic"
    ).parse_args()
    return probe()


if __name__ == "__main__":
    raise SystemExit(main())
