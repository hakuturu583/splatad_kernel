"""WebGPU adapter/device acquisition.

Kept in one place so the rest of the module never touches ``wgpu`` global
state, and so a host that already owns a device (a viewer, a test harness) can
hand it in instead.
"""

from __future__ import annotations

from typing import Any, Optional

_default_device: Optional[Any] = None


class WebGPUUnavailable(RuntimeError):
    """Raised when no WebGPU adapter can be obtained on this machine."""


def is_available() -> bool:
    """Whether a WebGPU device can be created here."""
    try:
        get_device()
    except Exception:
        return False
    return True


def get_device(device: Optional[Any] = None) -> Any:
    """Return ``device`` if given, else a cached default ``wgpu.GPUDevice``.

    The default device asks for the limits the pipeline actually needs
    (``maxStorageBufferBindingSize`` above all, which caps how many Gaussians
    fit in one call) rather than the spec defaults.
    """
    global _default_device
    if device is not None:
        return device
    if _default_device is not None:
        return _default_device

    try:
        import wgpu  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise WebGPUUnavailable(
            "the WebGPU backend needs the `wgpu` package: pip install wgpu"
        ) from exc

    import wgpu

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:  # pragma: no cover - driver dependent
        raise WebGPUUnavailable("no WebGPU adapter is available on this machine")

    limits = adapter.limits
    wanted = {}
    for key in (
        "max-storage-buffer-binding-size",
        "max-buffer-size",
        "max-compute-workgroup-storage-size",
        "max-compute-invocations-per-workgroup",
        "max-compute-workgroup-size-x",
        "max-compute-workgroup-size-y",
    ):
        value = limits.get(key)
        if value is not None:
            wanted[key] = value
    _default_device = adapter.request_device_sync(required_limits=wanted)
    return _default_device


def device_summary(device: Optional[Any] = None) -> str:
    """One-line description of the adapter in use, for logs and test output."""
    dev = get_device(device)
    info = dev.adapter.info
    return (
        f"{info.get('device', '?')} [{info.get('adapter_type', '?')}/"
        f"{info.get('backend_type', '?')}]"
    )
