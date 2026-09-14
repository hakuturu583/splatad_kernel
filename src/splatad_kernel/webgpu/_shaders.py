"""WGSL source loading: includes, conditional blocks and template variables.

The shaders are specialised the way the CUDA kernel is templated -- on the
feature count, the tile shape and the static/depth-compensation flags -- so the
source that reaches the driver is a concrete instantiation with no dynamic
branches on those axes. Everything is textual and happens once per distinct
configuration; the caller caches the resulting pipelines.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Mapping

_SHADER_DIR = Path(__file__).parent / "shaders"

_INCLUDE_RE = re.compile(r'^\s*//#include\s+"([^"]+)"\s*$')
_IF_RE = re.compile(r"^\s*//#if\s+(\w+)\s*$")
_ELSE_RE = re.compile(r"^\s*//#else\s*$")
_ENDIF_RE = re.compile(r"^\s*//#endif\s*$")
_VAR_RE = re.compile(r"\$\{(\w+)\}")

_source_cache: Dict[str, str] = {}


def read_shader(name: str) -> str:
    """Raw text of a shader file, with ``//#include`` directives expanded."""
    if name in _source_cache:
        return _source_cache[name]
    text = (_SHADER_DIR / name).read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        m = _INCLUDE_RE.match(line)
        if m:
            out.append(read_shader(m.group(1)))
        else:
            out.append(line)
    expanded = "\n".join(out)
    _source_cache[name] = expanded
    return expanded


def _apply_conditionals(text: str, defines: Mapping[str, bool]) -> str:
    """Resolve ``//#if NAME`` / ``//#else`` / ``//#endif`` blocks. Nestable."""
    out = []
    # Each entry is (this block's condition is currently satisfied, we are
    # inside a region whose enclosing blocks were all satisfied).
    stack = []

    def emitting() -> bool:
        return all(taken and enclosing for taken, enclosing in stack)

    for lineno, line in enumerate(text.splitlines(), 1):
        m = _IF_RE.match(line)
        if m:
            name = m.group(1)
            if name not in defines:
                raise KeyError(f"shader conditional #{name} has no value (line {lineno})")
            stack.append((bool(defines[name]), emitting()))
            continue
        if _ELSE_RE.match(line):
            if not stack:
                raise SyntaxError(f"//#else without //#if (line {lineno})")
            taken, enclosing = stack[-1]
            stack[-1] = (not taken, enclosing)
            continue
        if _ENDIF_RE.match(line):
            if not stack:
                raise SyntaxError(f"//#endif without //#if (line {lineno})")
            stack.pop()
            continue
        if emitting():
            out.append(line)
    if stack:
        raise SyntaxError("unterminated //#if in shader source")
    return "\n".join(out)


def build_shader(
    name: str,
    defines: Mapping[str, bool] = (),
    template: Mapping[str, object] = (),
) -> str:
    """Final WGSL for one specialisation of ``name``."""
    defines = dict(defines)
    template = dict(template)
    text = _apply_conditionals(read_shader(name), defines)

    def sub(m: "re.Match[str]") -> str:
        key = m.group(1)
        if key not in template:
            raise KeyError(f"shader template variable ${{{key}}} has no value")
        return str(template[key])

    return _VAR_RE.sub(sub, text)
