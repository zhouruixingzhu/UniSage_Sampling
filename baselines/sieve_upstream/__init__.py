"""Vendored Sieve (Huang et al., ICWS 2021) reference implementation.

Source: https://github.com/IntelligentDDS/Sieve (commit fetched 2026-05-28
shallow clone; the upstream repo has no tagged releases).

Files in this package are copied verbatim from the upstream repository with
the smallest set of patches required to import them in the macOS arm64 conda
env used by UniSage v0.2 baselines:

- `myrrcf.py` line 2: `from IPython.core.debugger import set_trace` replaced
  by a `set_trace` stub that raises on call, because IPython is not part of
  the `unisage` env. The only call sites are inside commented-out triple
  quoted blocks (`find_weak_dimensions`, two paths of `compact`), so they
  never execute. Patching the import keeps `import myrrcf` clean.
- `sieve.py` line 2: `import myrrcf` → `from . import myrrcf` so the file
  works as part of this Python package.
- `sieve.py` ~line 341: `pdb.set_trace()` (no `pdb` import, so the line
  would crash with `NameError` if its guarding `if` fires) replaced with an
  explicit `AssertionError`. This still surfaces the upstream invariant
  violation but does not enter a debugger and does not silently swallow the
  bug.

All other content is verbatim. The Chinese comments are part of the
upstream source. The companion adapter in `UniSage/baselines/sieve.py`
wraps these classes against the v0.2 normalized trace frame.
"""

from . import interface  # noqa: F401
from . import myrrcf  # noqa: F401
from . import sieve  # noqa: F401
