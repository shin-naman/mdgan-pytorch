"""Is G_z's noise sampled properly, and does G_z use it? Three checks on a snapshot.

1. Logged noise. A hook on G_z records the r actually passed in during the real estimation
   rollout (rollout.rollout). Over the first `steps` calls: no two steps identical, no two chains
   sharing an r within a step, every pair of steps uncorrelated across the batch (max |corr|
   against the sampling-noise level 4/sqrt(N)), and the moments of U[0,1) (and N(0,1) for the
   fresh-noise part, if latent_noise_share > 0).
2. Swapped stream. Two latent rollouts from the same z_0: one with the code's torch generator,
   one with r drawn by NumPy and copied in. The period (latent_cycle's spectrum analysis) and the
   pooled ACF must agree if the code's sampling is clean.
3. Perturbed r. (a) Two rollouts from the same z_0 with different r streams: distance between
   the two trajectories at each step, relative to the distance between independent chains
   (0 = G_z ignores r; 1 = fully mixed). (b) Per-chain Jacobians at a stationary latent (after
   50 steps): dz'/dz and dz'/dr, their Frobenius norms, the noise gain ||J_r|| / ||J_z|| (raw and
   weighted by the noise's std: r ~ U[0,1) has std 1/sqrt(12)), the spectral radius of J_z, and the
   top singular values of the chain-averaged J_z. BatchNorm with batch statistics couples chains,
   so the step is replicated per chain with the batch's mean and variance held fixed (exact up to
   O(1/batch)); the replica is checked against the real module before any Jacobian is used.

Usage (repository root):
    python tools/noise_checks.py --self-test                    # positive controls first
    python tools/noise_checks.py runs/chtc/run_c6954730_seed1/snap_110000.pt --device mps
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

_lc_spec = importlib.util.spec_from_file_location("latent_cycle", Path(__file__).with_name("latent_cycle.py"))
latent_cycle = importlib.util.module_from_spec(_lc_spec)
_lc_spec.loader.exec_module(latent_cycle)


def _noise_dim(gz) -> int:
    """Width of the transition noise proper (excluding dropout masks / fresh-noise columns)."""
    return getattr(gz, "_base_noise_width", gz.noise_dim)


def _width(gz) -> int:
    return getattr(gz, "noise_width", gz.noise_dim)


def _additive_std(gz) -> float:
    """> 0 for G_z with additive Gaussian input noise (Config.gz_additive_noise_std), else 0 (r ~ U[0,1))."""
    return float(getattr(gz, "gz_additive_noise_std", 0.0) or 0.0)


# ============================================================================ check 1

def check_logged_noise(gz, driver, steps: int = 20) -> dict:
    """driver() runs the real code path; a pre-hook on gz records every r it receives."""
    seen = []
    handle = gz.register_forward_pre_hook(lambda m, inp: seen.append(inp[1].detach().cpu().double().clone()))
    try:
        driver()
    finally:
        handle.remove()
    if len(seen) < 2:
        raise RuntimeError(f"only {len(seen)} G_z calls recorded")
    r = torch.stack(seen[:steps])                                         # (S, B, width)
    s, b, w = r.shape
    nd = _noise_dim(gz)
    identical_steps = [(i, j) for i in range(s) for j in range(i + 1, s) if torch.equal(r[i], r[j])]
    shared_rows = [i for i in range(s) if torch.unique(r[i], dim=0).shape[0] < b]
    flat = (r - r.mean(dim=1, keepdim=True)).reshape(s, -1)
    flat = flat / flat.norm(dim=1, keepdim=True)
    corr = (flat @ flat.T).numpy()
    off = np.abs(corr[~np.eye(s, dtype=bool)])
    level = 4 / math.sqrt(b * w)
    u = r[..., :nd]
    s_add = _additive_std(gz)
    out = {"steps": s, "batch": b, "width": w, "identical_step_pairs": identical_steps,
           "steps_with_shared_rows": shared_rows, "max_abs_corr": float(off.max()), "noise_level": level,
           "distribution": f"N(0, {s_add:g}^2)" if s_add > 0 else "U[0,1)",
           "u_mean": float(u.mean()), "u_var": float(u.var()), "u_min": float(u.min()), "u_max": float(u.max())}
    if w > nd:
        e = r[..., nd:]
        out.update({"eps_mean": float(e.mean()), "eps_std": float(e.std())})
    if s_add > 0:
        moments_ok = abs(out["u_mean"]) < 0.02 * s_add and abs(out["u_var"] / s_add ** 2 - 1) < 0.03
    else:
        moments_ok = (abs(out["u_mean"] - 0.5) < 0.01 and abs(out["u_var"] - 1 / 12) < 0.005
                      and out["u_min"] >= 0 and out["u_max"] < 1)
    out["ok"] = (not identical_steps and not shared_rows and out["max_abs_corr"] < level and moments_ok
                 and (w == nd or (abs(out["eps_mean"]) < 0.02 and abs(out["eps_std"] - 1) < 0.02)))
    return out


# ============================================================================ check 2

def _numpy_noise(gz, batch: int, rng: np.random.Generator, device, dtype) -> torch.Tensor:
    nd, w = _noise_dim(gz), _width(gz)
    s_add = _additive_std(gz)
    r = rng.standard_normal((batch, nd)) * s_add if s_add > 0 else rng.random((batch, nd))
    if w > nd:
        r = np.concatenate([r, rng.standard_normal((batch, w - nd))], axis=1)
    return torch.as_tensor(r, dtype=dtype, device=device)


def latents_torch(gz, z0, n_steps: int, seed: int) -> np.ndarray:
    with torch.no_grad():
        return gz.unroll(z0, n_steps, generator=torch.Generator().manual_seed(seed)).cpu().double().numpy()


def latents_numpy(gz, z0, n_steps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z, out = z0, [z0]
    with torch.no_grad():
        for _ in range(n_steps):
            z = gz(z, _numpy_noise(gz, z.shape[0], rng, z.device, z.dtype))
            out.append(z)
    return torch.stack(out).cpu().double().numpy()


def _cycle(z: np.ndarray) -> dict:
    per, pooled = latent_cycle.autocorrelation(z)
    f, p = latent_cycle.spectrum(z)
    cyc = latent_cycle.find_cycle(f, p, pooled, z.shape[0])
    return {"period_blocks": cyc["period_blocks"], "acf_first_peak": cyc["acf_first_peak_lag"],
            "cycle_fraction": cyc["cycle_fraction_of_total_variance"],
            "pooled_acf_1_6": [float(pooled[k]) for k in range(1, 7)]}


def check_swapped_stream(gz, z0, burn_in: int, n_blocks: int) -> dict:
    n = burn_in + n_blocks - 1
    a = _cycle(latents_torch(gz, z0, n, seed=1)[burn_in:])
    b = _cycle(latents_numpy(gz, z0, n, seed=1)[burn_in:])
    same = abs(a["period_blocks"] - b["period_blocks"]) <= 0.05 * a["period_blocks"]
    acf_close = max(abs(x - y) for x, y in zip(a["pooled_acf_1_6"], b["pooled_acf_1_6"], strict=True)) < 0.05
    return {"torch_stream": a, "numpy_stream": b, "ok": bool(same and acf_close)}


# ============================================================================ check 3

def check_r_sensitivity(gz, z0, n_steps: int, report_steps=(1, 2, 5, 10, 20, 50, 100)) -> dict:
    a = latents_torch(gz, z0, n_steps, seed=11)
    b = latents_torch(gz, z0, n_steps, seed=12)
    rng = np.random.default_rng(0)
    out = {}
    for t in report_steps:
        if t > n_steps:
            continue
        d_pair = np.linalg.norm(a[t] - b[t], axis=1).mean()                    # same chain, different r
        d_indep = np.linalg.norm(a[t] - a[t][rng.permutation(a.shape[1])], axis=1).mean()   # different chains
        out[t] = float(d_pair / d_indep) if d_indep > 0 else float("nan")
    return out


def _per_chain_step(gz, z_batch, r_batch):
    """f(z_i, r_i) for one chain, BatchNorm statistics frozen at the batch's values. Uses the model's
    own _trunk / norm / _head when it has them (any BatchNorm position, additive noise, dropout)."""
    if hasattr(gz, "_trunk"):
        bn = gz.norm
        with torch.no_grad():
            p = gz._trunk(z_batch, r_batch)
            if bn.training or not bn.track_running_stats:
                mu, var = p.mean(0), p.var(0, unbiased=False)
            else:
                mu, var = bn.running_mean, bn.running_var

        def trunk_step(z, r):
            normed = bn.weight * (gz._trunk(z[None], r[None])[0] - mu) / torch.sqrt(var + bn.eps) + bn.bias
            return gz._head(normed[None], r[None])[0]
        return trunk_step
    if not hasattr(gz, "output_norm"):
        return lambda z, r: gz(z[None], r[None])[0]
    nd, bn = gz.noise_dim, gz.output_norm
    a = float(getattr(gz, "latent_noise_share", 0.0))

    def pre(z, r):
        h = gz.act(gz.input_layer(torch.cat((z, r[..., :nd]), dim=-1)))
        h = gz.act(gz.hidden2(gz.act(gz.hidden1(h))))
        return gz.output_layer(h)

    with torch.no_grad():
        p = pre(z_batch, r_batch)
        if bn.training or not bn.track_running_stats:
            mu, var = p.mean(0), p.var(0, unbiased=False)
        else:
            mu, var = bn.running_mean, bn.running_var

    def step(z, r):
        out = bn.weight * (pre(z, r) - mu) / torch.sqrt(var + bn.eps) + bn.bias
        if a > 0:
            out = math.sqrt(1.0 - a * a) * out + a * r[nd:]
        return out
    return step


def check_jacobians(gz, z0, stationary_steps: int = 50, n_chains: int = 256, seed: int = 21) -> dict:
    from torch.func import jacrev, vmap
    with torch.no_grad():
        z = gz.unroll(z0, stationary_steps, generator=torch.Generator().manual_seed(seed))[-1]
        r = gz.sample_noise(z.shape[0], device=z.device, dtype=z.dtype,
                            generator=torch.Generator().manual_seed(seed + 1))
        step = _per_chain_step(gz, z, r)
        replica = torch.stack([step(z[i], r[i]) for i in range(min(16, z.shape[0]))])
        real = gz(z, r)[: replica.shape[0]]
    err = float((replica - real).abs().max())
    if err > 1e-4:
        raise RuntimeError(f"per-chain replica differs from G_z by {err:.2e}; Jacobians would be wrong")
    jz, jr = vmap(jacrev(step, argnums=(0, 1)))(z[:n_chains], r[:n_chains])
    jz, jr = jz.detach().cpu().double(), jr.detach().cpu().double()
    nd = _noise_dim(gz)
    nz = torch.linalg.matrix_norm(jz).detach()
    nr = torch.linalg.matrix_norm(jr[..., :nd]).detach()
    rho = torch.linalg.eigvals(jz).abs().max(dim=-1).values
    out = {"replica_max_abs_err": err, "chains": int(jz.shape[0]),
           "norm_Jz_median": float(nz.median()), "norm_Jr_median": float(nr.median()),
           "noise_gain_raw_median": float((nr / nz).median()),
           "noise_gain_std_weighted_median": float((nr * (_additive_std(gz) or 1 / math.sqrt(12)) / nz).median()),
           "spectral_radius_Jz_median": float(rho.median()),
           "top3_singular_values_mean_Jz": [float(x) for x in torch.linalg.svdvals(jz.mean(0))[:3]]}
    if jr.shape[-1] > nd:
        out["norm_Jeps_median"] = float(torch.linalg.matrix_norm(jr[..., nd:]).median())
    return out


# ============================================================================ positive controls

class _Rotate5(nn.Module):
    """z' = R z (+ optional r term): 72-degree rotations in the planes (0,1), (2,3), (4,5): period 5,
    spectral radius exactly 1."""

    def __init__(self, uses_r: float = 0.0, reseed_noise: bool = False):
        super().__init__()
        self.latent_dim, self.noise_dim, self.noise_width = 6, 4, 4
        self.uses_r, self.reseed_noise = uses_r, reseed_noise
        c, s = math.cos(2 * math.pi / 5), math.sin(2 * math.pi / 5)
        rot = torch.zeros(6, 6)
        for k in (0, 2, 4):
            rot[k:k + 2, k:k + 2] = torch.tensor([[c, -s], [s, c]])
        self.register_buffer("rot", rot)
        self.mix = nn.Parameter(torch.randn(4, 6) * 0.8, requires_grad=False)

    def forward(self, z, r):
        return z @ self.rot.T + self.uses_r * ((r - 0.5) @ self.mix)

    def sample_noise(self, batch, *, device=None, dtype=None, generator=None):
        g = torch.Generator().manual_seed(0) if self.reseed_noise else generator
        return torch.rand(batch, self.noise_dim, dtype=dtype, generator=g).to(device or "cpu")

    def unroll(self, z0, steps, generator=None):
        out, z = [z0], z0
        for _ in range(steps):
            z = self(z, self.sample_noise(z.shape[0], device=z.device, dtype=z.dtype, generator=generator))
            out.append(z)
        return torch.stack(out)


def _manual_driver(gz, z0, steps, seed):
    def run():
        g, z = torch.Generator().manual_seed(seed), z0
        with torch.no_grad():
            for _ in range(steps):
                z = gz(z, gz.sample_noise(z.shape[0], device=z.device, dtype=z.dtype, generator=g))
    return run


def self_test() -> None:
    torch.manual_seed(0)
    z0 = torch.randn(1024, 6)
    ignore, uses, broken = _Rotate5(0.0), _Rotate5(1.0), _Rotate5(0.0, reseed_noise=True)

    c1 = check_logged_noise(ignore, _manual_driver(ignore, z0, 30, seed=3))
    c1_bad = check_logged_noise(broken, _manual_driver(broken, z0, 30, seed=3))
    assert c1["ok"], c1
    assert not c1_bad["ok"] and c1_bad["identical_step_pairs"], "check 1 failed to catch a reseeded stream"
    print(f"check 1 control: clean stream passes (max |corr| {c1['max_abs_corr']:.4f} < {c1['noise_level']:.4f}); "
          f"reseeded stream caught ({len(c1_bad['identical_step_pairs'])} identical step pairs)")

    c2 = check_swapped_stream(ignore, z0, burn_in=5, n_blocks=125)
    assert c2["ok"] and abs(c2["torch_stream"]["period_blocks"] - 5) < 0.2, c2
    print(f"check 2 control: period {c2['torch_stream']['period_blocks']:.2f} (torch) vs "
          f"{c2['numpy_stream']['period_blocks']:.2f} (numpy)")

    s_ign = check_r_sensitivity(ignore, z0, 50)
    s_use = check_r_sensitivity(uses, z0, 50)
    j_ign = check_jacobians(ignore, z0[:512], stationary_steps=10, n_chains=64)
    j_use = check_jacobians(uses, z0[:512], stationary_steps=10, n_chains=64)
    assert max(s_ign.values()) < 1e-9 and j_ign["norm_Jr_median"] < 1e-12, (s_ign, j_ign)
    assert s_use[20] > 0.5 and j_use["norm_Jr_median"] > 0.5, (s_use, j_use)
    assert abs(j_ign["spectral_radius_Jz_median"] - 1.0) < 1e-6
    print(f"check 3 control: ignores r -> trajectory ratio {max(s_ign.values()):.1e}, ||J_r|| {j_ign['norm_Jr_median']:.1e}, "
          f"spectral radius {j_ign['spectral_radius_Jz_median']:.3f}; uses r -> ratio {s_use[20]:.2f} at step 20, "
          f"||J_r|| {j_use['norm_Jr_median']:.2f}")
    print("all positive controls passed")


# ============================================================================ real snapshots

def run_snapshot(path: str, chains: int, device: str) -> dict:
    from mdgan.checkpoint import load_checkpoint
    from mdgan.rollout import rollout
    ck = load_checkpoint(path, map_location="cpu")
    c = ck.config
    gz, gy = ck.gz.to(device).eval(), ck.gy.to(device).eval()
    n_blocks = c.estimation_blocks(10.0)
    z0 = torch.rand(chains, c.latent_dim, generator=torch.Generator().manual_seed(0)).to(device)
    res = {"snapshot": path, "iteration": ck.iteration, "latent_noise_share": float(getattr(gz, "latent_noise_share", 0.0))}
    # 20 blocks = burn_in + 19 G_z calls (>= the 20 steps check 1 analyses)
    res["check1"] = check_logged_noise(gz, lambda: rollout(gz, gy, chains, 20, c.burn_in,
                                                          generator=torch.Generator().manual_seed(5)))
    res["check2"] = check_swapped_stream(gz, z0, c.burn_in, n_blocks)
    res["check3_trajectory_ratio"] = check_r_sensitivity(gz, z0, c.burn_in + n_blocks - 1)
    res["check3_jacobians"] = check_jacobians(gz, z0)
    return res


def print_report(res: dict) -> None:
    c1, c2, tr, jac = res["check1"], res["check2"], res["check3_trajectory_ratio"], res["check3_jacobians"]
    print(f"\n=== {res['snapshot']} (iteration {res['iteration']})")
    print(f"1. logged noise ({c1['steps']} G_z calls in the real rollout, batch {c1['batch']}): "
          f"{'OK' if c1['ok'] else 'PROBLEM'}")
    print(f"   identical step pairs {len(c1['identical_step_pairs'])}, steps with repeated rows "
          f"{len(c1['steps_with_shared_rows'])}, max |corr| between steps {c1['max_abs_corr']:.4f} "
          f"(noise level {c1['noise_level']:.4f}); noise {c1['distribution']}: mean {c1['u_mean']:.4f}, var {c1['u_var']:.4f}")
    t, n = c2["torch_stream"], c2["numpy_stream"]
    print(f"2. swapped stream: period {t['period_blocks']:.2f} (torch) vs {n['period_blocks']:.2f} (numpy); "
          f"ACF 1-6 torch {' '.join(f'{x:+.2f}' for x in t['pooled_acf_1_6'])} | numpy "
          f"{' '.join(f'{x:+.2f}' for x in n['pooled_acf_1_6'])}: {'OK' if c2['ok'] else 'DIFFERENT'}")
    print("3a. same z0, different r: trajectory distance / independent-chain distance at step "
          + ", ".join(f"{k}: {v:.3f}" for k, v in tr.items()) + "   (0 = r ignored, 1 = fully mixed)")
    print(f"3b. Jacobians at a stationary latent ({jac['chains']} chains; replica error {jac['replica_max_abs_err']:.1e}): "
          f"||dz'/dz|| {jac['norm_Jz_median']:.3f}, ||dz'/dr|| {jac['norm_Jr_median']:.4f}, "
          f"gain raw {jac['noise_gain_raw_median']:.4f}, std-weighted {jac['noise_gain_std_weighted_median']:.4f}, "
          f"spectral radius {jac['spectral_radius_Jz_median']:.3f}, top singular values of mean J_z "
          + " ".join(f"{x:.3f}" for x in jac["top3_singular_values_mean_Jz"])
          + (f", ||dz'/d eps|| {jac['norm_Jeps_median']:.3f}" if "norm_Jeps_median" in jac else ""))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("snapshots", nargs="*")
    ap.add_argument("--self-test", action="store_true", help="run the positive controls")
    ap.add_argument("--chains", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/noise_checks")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
    results = [run_snapshot(s, args.chains, args.device) for s in args.snapshots]
    for r in results:
        print_report(r)
    if results:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "noise_checks.json").write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwrote {args.out}/noise_checks.json")


if __name__ == "__main__":
    main()
