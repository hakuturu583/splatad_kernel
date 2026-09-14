"""Bind group layouts, pipeline construction and a growable buffer pool.

Layouts are declared explicitly rather than derived from the shader, because
two entry points in one module (isect.wgsl's ``count`` and ``encode``) use
different subsets of the same bindings and an inferred layout would disagree
between them.
"""

from __future__ import annotations

import struct
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import wgpu

from ._shaders import build_shader

_UNIFORM = "uniform"
_RO = "read-only-storage"
_RW = "storage"

# binding 0 is Params (or the sort step) in every layout.
LAYOUTS: Dict[str, Sequence[str]] = {
    # P, gaussians, sensors, proj
    "project": (_UNIFORM, _RO, _RO, _RW),
    # P, proj, sensors, counts, cum, isects
    "isect": (_UNIFORM, _RO, _RO, _RW, _RO, _RW),
    # src, dst, block_sums
    "scan": (_RO, _RW, _RW),
    # step, isects
    "sort": (_UNIFORM, _RW),
    # params, src, dst, hist, hist_incl
    "radix": (_UNIFORM, _RO, _RW, _RW, _RO),
    # P, isects, tile_offsets
    "offsets": (_UNIFORM, _RO, _RW),
    # P, proj, features, raster_pts, tile_offsets, isects, out_colors, out_aux
    "raster": (_UNIFORM, _RO, _RO, _RO, _RO, _RO, _RW, _RW),
    # P, proj, gaussians, sensors, sh_coeffs, features
    "sh": (_UNIFORM, _RO, _RO, _RO, _RO, _RW),
}

STORAGE_USAGE = (
    wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
)
UNIFORM_USAGE = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST

# Every field of the Params struct in common.wgsl, in order. "I" is u32, "f" f32.
PARAM_FIELDS: Sequence[Tuple[str, str]] = (
    ("n_gaussians", "I"),
    ("n_cameras", "I"),
    ("n_features", "I"),
    ("flags", "I"),
    ("min_azimuth", "f"),
    ("max_azimuth", "f"),
    ("min_elevation", "f"),
    ("max_elevation", "f"),
    ("eps2d", "f"),
    ("near_plane", "f"),
    ("far_plane", "f"),
    ("radius_clip", "f"),
    ("n_tiles_azim", "I"),
    ("n_tiles_elev", "I"),
    ("tile_azim_resolution", "f"),
    ("tile_n_bits", "I"),
    ("image_width", "I"),
    ("image_height", "I"),
    ("tile_width", "I"),
    ("tile_height", "I"),
    ("tile_col_offset", "I"),
    ("alpha_sum_threshold", "f"),
    ("n_isects", "I"),
    ("sh_degree", "I"),
    ("raydrop_feature_index", "I"),
    ("elev_base", "I"),
    ("row_elev_base", "I"),
    ("n_isects_padded", "I"),
)
PARAMS_FORMAT = "<" + "".join(fmt for _, fmt in PARAM_FIELDS)
PARAMS_SIZE = struct.calcsize(PARAMS_FORMAT)

FLAG_HAS_VELOCITIES = 1
FLAG_HAS_VALID_MASK = 2
FLAG_CALC_COMPENSATIONS = 4
FLAG_EXACT_ROW_SPANS = 8
FLAG_HAS_ROW_ELEVATIONS = 16
FLAG_COMPUTE_ALPHA_SUM = 32


def pack_params(values: Mapping[str, Any]) -> bytes:
    missing = [name for name, _ in PARAM_FIELDS if name not in values]
    if missing:
        raise KeyError(f"missing shader params: {missing}")
    return struct.pack(PARAMS_FORMAT, *(values[name] for name, _ in PARAM_FIELDS))


class Resources:
    """Pipelines, bind group layouts and a pool of growable storage buffers."""

    def __init__(self, device: Any):
        self.device = device
        self._layouts: Dict[str, Any] = {}
        self._pipeline_layouts: Dict[str, Any] = {}
        self._pipelines: Dict[Tuple, Any] = {}
        self._buffers: Dict[str, Any] = {}

    # -- layouts ---------------------------------------------------------
    def layout(self, name: str) -> Any:
        if name not in self._layouts:
            self._layouts[name] = self.device.create_bind_group_layout(
                entries=[
                    {
                        "binding": i,
                        "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": kind},
                    }
                    for i, kind in enumerate(LAYOUTS[name])
                ]
            )
        return self._layouts[name]

    def pipeline_layout(self, name: str) -> Any:
        if name not in self._pipeline_layouts:
            self._pipeline_layouts[name] = self.device.create_pipeline_layout(
                bind_group_layouts=[self.layout(name)]
            )
        return self._pipeline_layouts[name]

    # -- pipelines -------------------------------------------------------
    def pipeline(
        self,
        layout_name: str,
        shader: str,
        entry_point: str,
        defines: Optional[Mapping[str, bool]] = None,
        template: Optional[Mapping[str, object]] = None,
    ) -> Any:
        defines = dict(defines or {})
        template = dict(template or {})
        key = (
            layout_name,
            shader,
            entry_point,
            tuple(sorted(defines.items())),
            tuple(sorted((k, str(v)) for k, v in template.items())),
        )
        if key not in self._pipelines:
            source = build_shader(shader, defines, template)
            module = self.device.create_shader_module(code=source, label=shader)
            self._pipelines[key] = self.device.create_compute_pipeline(
                layout=self.pipeline_layout(layout_name),
                compute={"module": module, "entry_point": entry_point},
                label=f"{shader}:{entry_point}",
            )
        return self._pipelines[key]

    # -- buffers ---------------------------------------------------------
    def buffer(self, name: str, nbytes: int, usage: int = STORAGE_USAGE) -> Any:
        """A buffer of at least ``nbytes``, reused across calls and grown in place.

        Bindings always carry an explicit size, so a buffer that is larger than
        the current call needs is harmless -- including for ``arrayLength()``,
        which reports the bound range rather than the allocation.
        """
        nbytes = max(int(nbytes), 4)
        # Round up so a slowly growing scene does not reallocate every frame.
        nbytes = (nbytes + 255) & ~255
        buf = self._buffers.get(name)
        if buf is None or buf.size < nbytes:
            buf = self.device.create_buffer(size=nbytes, usage=usage, label=name)
            self._buffers[name] = buf
        return buf

    def bind_group(
        self,
        layout_name: str,
        entries: Sequence[Tuple[Any, int]],
        offsets: Optional[Sequence[int]] = None,
    ) -> Any:
        """``entries`` is (buffer, size-in-bytes) per binding, in binding order."""
        offsets = offsets or [0] * len(entries)
        resources: List[Dict[str, Any]] = []
        for i, ((buf, size), offset) in enumerate(zip(entries, offsets)):
            resources.append(
                {
                    "binding": i,
                    "resource": {"buffer": buf, "offset": offset, "size": size},
                }
            )
        return self.device.create_bind_group(
            layout=self.layout(layout_name), entries=resources
        )
