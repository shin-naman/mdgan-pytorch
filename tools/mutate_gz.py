"""Mutation testing for tests/G_z.

Each mutation breaks LatentTransitionGenerator in one specific way, then the
G_z suite runs in a fresh subprocess. A surviving mutant is a hole in the
tests. Goal: zero survivors.

Run from the repo root (venv active):
    python tools/mutate_gz.py            # all mutations
    python tools/mutate_gz.py unroll_detach   # one mutation, verbose pytest output
"""

import subprocess
import sys
import textwrap

MUTATIONS = {
    # --- data path ---------------------------------------------------------
    "concat": """
        def forward(self, z, r):
            import torch as _t
            x = _t.cat((r, z), dim=1)                    # [r, z] instead of [z, r]
            h = self.act(self.input_layer(x))
            h = self.act(self.hidden1(h))
            h = self.act(self.hidden2(h))
            return self.output_norm(self.output_layer(h))
        TG.forward = forward
    """,
    "skip_hidden2": """
        import torch as _t
        _orig = TG.forward
        def forward(self, z, r):
            self.hidden2 = _t.nn.Identity()
            return _orig(self, z, r)
        TG.forward = forward
    """,
    "act_missing_after_hidden1": """
        _orig_fwd = TG.forward
        def forward(self, z, r):
            import torch as _t
            x = _t.cat((z, r), dim=1)
            h = self.act(self.input_layer(x))
            h = self.hidden1(h)                          # no activation here
            h = self.act(self.hidden2(h))
            return self.output_norm(self.output_layer(h))
        TG.forward = forward
    """,
    "act_after_output": """
        _orig = TG.forward
        def forward(self, z, r):
            return self.act(_orig(self, z, r))
        TG.forward = forward
    """,
    "no_batchnorm": """
        import torch as _t
        _orig = TG.__init__
        def __init__(self, *a, **k):
            _orig(self, *a, **k)
            self.output_norm = _t.nn.Identity()
        TG.__init__ = __init__
    """,
    "batchnorm_tracks_running_stats_by_default": """
        import torch as _t
        _orig = TG.__init__
        def __init__(self, *a, **k):
            k.setdefault("use_running_stats", True)
            _orig(self, *a, **k)
        TG.__init__ = __init__
    """,
    "batchnorm_no_affine": """
        import torch as _t
        _orig = TG.__init__
        def __init__(self, *a, **k):
            _orig(self, *a, **k)
            self.output_norm = _t.nn.BatchNorm1d(self.latent_dim, affine=False,
                                                 track_running_stats=self.use_running_stats)
        TG.__init__ = __init__
    """,
    "wrong_negative_slope": """
        _orig = TG.__init__
        def __init__(self, *a, **k):
            k["negative_slope"] = 0.01
            _orig(self, *a, **k)
        TG.__init__ = __init__
    """,
    # --- validation ----------------------------------------------------------
    "accept_batch_of_one": """
        _orig = TG.forward
        def forward(self, z, r):
            if z.shape[0] == 1 and z.dim() == 2 and r.dim() == 2:
                import torch as _t
                return _orig(self, _t.cat([z, z]), _t.cat([r, r]))[:1]
            return _orig(self, z, r)
        TG.forward = forward
    """,
    "validate_finite_default_on": """
        _orig = TG.__init__
        def __init__(self, *a, **k):
            k.setdefault("validate_finite", True)
            _orig(self, *a, **k)
        TG.__init__ = __init__
    """,
    # --- randomness ------------------------------------------------------------
    "forward_consumes_global_rng": """
        import torch as _t
        _orig = TG.forward
        def forward(self, z, r):
            _t.rand(1)
            return _orig(self, z, r)
        TG.forward = forward
    """,
    "sample_noise_normal": """
        import torch as _t
        def sample_noise(self, batch, *, device=None, dtype=None, generator=None):
            return _t.randn(batch, self.noise_dim, device=device, dtype=dtype, generator=generator)
        TG.sample_noise = sample_noise
    """,
    "unroll_ignores_generator": """
        def unroll(self, z0, steps, generator=None):
            import torch as _t
            if steps < 0:
                raise ValueError("steps must be non-negative")
            states, z = [z0], z0
            for _ in range(steps):
                z = self(z, self.sample_noise(z.shape[0], device=z.device, dtype=z.dtype))
                states.append(z)
            return _t.stack(states, dim=0)
        TG.unroll = unroll
    """,
    # --- unroll semantics ---------------------------------------------------------
    "unroll_detach": """
        def unroll(self, z0, steps, generator=None):
            import torch as _t
            if steps < 0:
                raise ValueError("steps must be non-negative")
            states, z = [z0], z0
            for _ in range(steps):
                z = self(z.detach(), self.sample_noise(z.shape[0], device=z.device, dtype=z.dtype, generator=generator))
                states.append(z)
            return _t.stack(states, dim=0)
        TG.unroll = unroll
    """,
    "unroll_off_by_one": """
        _orig = TG.unroll
        def unroll(self, z0, steps, generator=None):
            return _orig(self, z0, max(steps - 1, 0), generator)
        TG.unroll = unroll
    """,
    "unroll_drops_z0": """
        _orig = TG.unroll
        def unroll(self, z0, steps, generator=None):
            out = _orig(self, z0, steps, generator)
            return out[1:] if steps > 0 else out
        TG.unroll = unroll
    """,
}

RUNNER = """
import sys
import mdgan.models.latent_transition as mod
TG = mod.LatentTransitionGenerator
{patch}
import pytest
sys.exit(pytest.main(["tests/G_z", "-x", "-q", "-p", "no:cacheprovider", "--no-header"{extra}]))
"""


def run(name: str, verbose: bool) -> bool:
    patch = textwrap.dedent(MUTATIONS[name])
    extra = "" if verbose else ', "-q"'
    code = RUNNER.format(patch=patch, extra=extra)
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=not verbose, text=True)
    killed = proc.returncode != 0
    if verbose and not killed:
        print(proc.stdout)
    return killed


def main() -> int:
    names = [sys.argv[1]] if len(sys.argv) > 1 else list(MUTATIONS)
    verbose = len(sys.argv) > 1
    survivors = []
    for name in names:
        killed = run(name, verbose)
        print(f"{'KILLED  ' if killed else 'SURVIVED'}  {name}")
        if not killed:
            survivors.append(name)
    print()
    if survivors:
        print(f"{len(survivors)} mutant(s) survived: {', '.join(survivors)}")
        print("Each survivor is a hole in tests/G_z.")
        return 1
    print(f"All {len(names)} mutants killed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())