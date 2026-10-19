"""Mutation testing for tests/D_G. Goal: zero survivors.

Run from the repo root (venv active):
    python tools/mutate_dg.py            # all mutations
    python tools/mutate_dg.py <name>     # one mutation, verbose pytest output
"""

import subprocess
import sys
import textwrap

MUTATIONS = {
    "accept_any_multiple_of_window": """
        _orig = D.forward
        def forward(self, window):
            import torch as _t
            if window.dim() == 3 and window.shape[2] == self.n_channels and window.shape[1] % self.window_len == 0:
                h = self.input_conv(window.transpose(1, 2)).mean(-1)
                h = self.act(h)
                h = self.act(self.hidden1(h)); h = self.act(self.hidden2(h)); h = self.act(self.hidden3(h))
                return self.output_layer(h).squeeze(-1)
            return _orig(self, window)
        D.forward = forward
    """,
    "squeeze_without_axis": """
        _orig = D.forward
        def forward(self, window):
            if window.dim() != 3 or window.shape[1:] != (self.window_len, self.n_channels):
                raise ValueError("bad")
            h = self.act(self.input_conv(window.transpose(1, 2)).squeeze())
            h = self.act(self.hidden1(h)); h = self.act(self.hidden2(h)); h = self.act(self.hidden3(h))
            return self.output_layer(h).squeeze()
        D.forward = forward
    """,
    "output_shape_B1": """
        _orig = D.forward
        def forward(self, window):
            return _orig(self, window).unsqueeze(-1)
        D.forward = forward
    """,
    "reshape_instead_of_transpose": """
        _orig = D.forward
        def forward(self, window):
            if window.dim() != 3 or window.shape[1:] != (self.window_len, self.n_channels):
                raise ValueError("bad")
            h = self.input_conv(window.reshape(window.shape[0], self.n_channels, self.window_len))
            h = self.act(h.squeeze(-1))
            h = self.act(self.hidden1(h)); h = self.act(self.hidden2(h)); h = self.act(self.hidden3(h))
            return self.output_layer(h).squeeze(-1)
        D.forward = forward
    """,
    "batchnorm_added": """
        import torch as _t
        _orig_init = D.__init__
        def __init__(self, *a, **k):
            _orig_init(self, *a, **k)
            self.bn = _t.nn.BatchNorm1d(self.hidden_dim)
        D.__init__ = __init__
        def forward(self, window):
            if window.dim() != 3 or window.shape[1:] != (self.window_len, self.n_channels):
                raise ValueError("bad")
            h = self.act(self.bn(self.input_conv(window.transpose(1, 2)).squeeze(-1)))
            h = self.act(self.hidden1(h)); h = self.act(self.hidden2(h)); h = self.act(self.hidden3(h))
            return self.output_layer(h).squeeze(-1)
        D.forward = forward
    """,
    "detach_input": """
        _orig = D.forward
        def forward(self, window):
            return _orig(self, window.detach())
        D.forward = forward
    """,
    "sigmoid_on_output": """
        import torch as _t
        _orig = D.forward
        def forward(self, window):
            return _t.sigmoid(_orig(self, window))
        D.forward = forward
    """,
    "skip_hidden2": """
        import torch as _t
        _orig = D.forward
        def forward(self, window):
            self.hidden2 = _t.nn.Identity()
            return _orig(self, window)
        D.forward = forward
    """,
    "act_missing_after_hidden3": """
        def forward(self, window):
            if window.dim() != 3 or window.shape[1:] != (self.window_len, self.n_channels):
                raise ValueError("bad")
            h = self.act(self.input_conv(window.transpose(1, 2)).squeeze(-1))
            h = self.act(self.hidden1(h)); h = self.act(self.hidden2(h)); h = self.hidden3(h)
            return self.output_layer(h).squeeze(-1)
        D.forward = forward
    """,
    "act_after_output": """
        _orig = D.forward
        def forward(self, window):
            return self.act(_orig(self, window))
        D.forward = forward
    """,
    "ignore_second_block": """
        _orig = D.forward
        def forward(self, window):
            import torch as _t
            w = _t.cat([window[:, :self.block_len], window[:, :self.block_len]], dim=1)
            return _orig(self, w)
        D.forward = forward
    """,
    "wrong_negative_slope": """
        _orig = D.__init__
        def __init__(self, *a, **k):
            k["negative_slope"] = 0.01
            _orig(self, *a, **k)
        D.__init__ = __init__
    """,
    "conv_stride_1": """
        import torch as _t
        _orig = D.__init__
        def __init__(self, *a, **k):
            _orig(self, *a, **k)
            self.input_conv = _t.nn.Conv1d(self.n_channels, self.hidden_dim,
                                           kernel_size=self.window_len, stride=1)
        D.__init__ = __init__
    """,
    "extra_buffer": """
        import torch as _t
        _orig = D.__init__
        def __init__(self, *a, **k):
            _orig(self, *a, **k)
            self.register_buffer("scale", _t.ones(1))
        D.__init__ = __init__
    """,
    "forward_consumes_global_rng": """
        import torch as _t
        _orig = D.forward
        def forward(self, window):
            _t.rand(1)
            return _orig(self, window)
        D.forward = forward
    """,
    "inplace_on_input": """
        _orig = D.forward
        def forward(self, window):
            window.mul_(1.0)
            return _orig(self, window)
        D.forward = forward
    """,
}

RUNNER = """
import sys
import mdgan.models.critic as mod
D = mod.Critic
{patch}
import pytest
sys.exit(pytest.main(["tests/D_G", "-x", "-q", "-p", "no:cacheprovider", "--no-header"{extra}]))
"""


def run(name: str, verbose: bool) -> bool:
    patch = textwrap.dedent(MUTATIONS[name])
    extra = "" if verbose else ', "-q"'
    code = RUNNER.format(patch=patch, extra=extra)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=not verbose, text=True)
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
        print("Each survivor is a hole in tests/D_G.")
        return 1
    print(f"All {len(names)} mutants killed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())