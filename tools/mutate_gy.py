"""Mutation testing for tests/G_Y.

Each mutation breaks TrajectoryGenerator in one specific way, then the G_Y
test suite is run in a fresh subprocess. A mutation that leaves the suite
green has found a hole in the tests. The script reports which mutants
survived; the goal is zero.

Run from the repo root (venv active):
    python tools/mutate_gy.py            # all mutations
    python tools/mutate_gy.py concat     # one mutation, verbose pytest output
"""

import subprocess
import sys
import textwrap

MUTATIONS = {
    # --- data path ---------------------------------------------------------
    "concat": """
        def forward(self, z, r_prime, u):
            import torch as _t
            b = z.shape[0]
            x = _t.cat((z, r_prime, self._mix_noise(u)), dim=1)   # [z, r', u'] instead of [r', z, u']
            h = self.input_act(self.input_layer(x))
            h = self.post_act(self.residual(h))
            return self.output_layer(h).reshape(b, self.block_len, self.n_channels)
        TG.forward = forward
    """,
    "flatten_channel_major": """
        def _mix_noise(self, u):
            b = u.shape[0]
            mixed = self.noise_mix(u.transpose(1, 2))      # (B, C, M), never transposed back
            return mixed.reshape(b, self.flat_dim)
        TG._mix_noise = _mix_noise
    """,
    "reshape_channel_major": """
        _orig = TG.forward
        def forward(self, z, r_prime, u):
            y = _orig(self, z, r_prime, u)                  # (B, M, C)
            return y.reshape(y.shape[0], self.flat_dim).reshape(y.shape[0], self.n_channels, self.block_len).transpose(1, 2)
        TG.forward = forward
    """,
    "temporal_leak_in_noise_mix": """
        _orig = TG._mix_noise
        def _mix_noise(self, u):
            return _orig(self, u + 0.5 * u.roll(1, dims=1))  # step t sees step t-1
        TG._mix_noise = _mix_noise
    """,
    # --- residual block ----------------------------------------------------
    "residual_no_skip": """
        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))
        RB.forward = forward
    """,
    "residual_act_after_add": """
        def forward(self, x):
            return self.act(x + self.fc2(self.act(self.fc1(x))))
        RB.forward = forward
    """,
    "skip_post_act": """
        import torch as _t
        _orig = TG.forward
        def forward(self, z, r_prime, u):
            self.post_act = _t.nn.Identity()
            return _orig(self, z, r_prime, u)
        TG.forward = forward
    """,
    # --- noise / randomness ------------------------------------------------
    "ignore_u": """
        _orig = TG._mix_noise
        def _mix_noise(self, u):
            return _orig(self, u * 0.0)
        TG._mix_noise = _mix_noise
    """,
    "forward_consumes_global_rng": """
        import torch as _t
        _orig = TG.forward
        def forward(self, z, r_prime, u):
            _t.rand(1)
            return _orig(self, z, r_prime, u)
        TG.forward = forward
    """,
    "generate_ignores_generator": """
        def generate(self, z, generator=None):
            r_prime, u = self.sample_noise(z.shape[0], device=z.device, dtype=z.dtype)
            return self(z, r_prime, u)
        TG.generate = generate
    """,
    "u_sampled_uniform": """
        import torch as _t
        def sample_noise(self, batch, *, device=None, dtype=None, generator=None):
            r = _t.rand(batch, self.noise_dim, device=device, dtype=dtype, generator=generator)
            u = _t.rand(batch, self.block_len, self.n_channels, device=device, dtype=dtype, generator=generator)
            return r, u
        TG.sample_noise = sample_noise
    """,
    # --- structure -----------------------------------------------------------
    "wrong_negative_slope": """
        import torch as _t
        _orig = TG.__init__
        def __init__(self, *a, **k):
            k["negative_slope"] = 0.01
            _orig(self, *a, **k)
        TG.__init__ = __init__
    """,
    "extra_buffer": """
        import torch as _t
        _orig = TG.__init__
        def __init__(self, *a, **k):
            _orig(self, *a, **k)
            self.register_buffer("scale", _t.ones(1))
        TG.__init__ = __init__
    """,
    "tanh_on_output": """
        import torch as _t
        _orig = TG.forward
        def forward(self, z, r_prime, u):
            return _t.tanh(_orig(self, z, r_prime, u))
        TG.forward = forward
    """,
}

RUNNER = """
import sys
import mdgan.models.trajectory_generator as mod
TG = mod.TrajectoryGenerator
RB = mod.ResidualAffineBlock
{patch}
import pytest
sys.exit(pytest.main(["tests/G_Y", "-x", "-q", "-p", "no:cacheprovider", "--no-header"{extra}]))
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
        print("Each survivor is a hole in tests/G_Y.")
        return 1
    print(f"All {len(names)} mutants killed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())