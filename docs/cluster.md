# Running on CHTC (HTCondor)

Independent single-GPU jobs (one per seed or setting) through HTCondor's GPU Lab. The
Slurm HPC cluster is for multi-node MPI work and is not needed here.

Files:

| File | Purpose |
| --- | --- |
| `configs/pe_paper.json` | The paper's PE protocol, written out in full (equal to `Config()` today, but explicit so a later default change cannot alter a run). |
| `tools/chtc/run_train.sh` | Job wrapper: prints the environment, runs `mdgan.train` with `--auto-resume --max-hours`, passes the exit code through. |
| `tools/chtc/pe_sizing.sub` | One 2,000-iteration job to measure speed. Submit first. |
| `tools/chtc/pe_paper.sub` | Five seeds of the full 300k-iteration protocol. |

## What a job carries

Only three inputs, a few MB in total: `src/` (the package, put on `PYTHONPATH`; nothing is
installed), `configs/pe_paper.json`, and `data/cache/PE.npz` (the preprocessed
displacements). Training reads the `.npz` directly, so jobs need neither the `.xtc` files
nor mdtraj. The container image provides Python, torch with CUDA, and numpy; the code needs
Python ≥ 3.10.

## One-time setup

On your laptop, if the cache does not exist yet:

```
python -m mdgan.data.preprocess data/raw/wat_in_polymer_mdgan/MD_data/PE --out data/cache/PE.npz
```

Commit your changes, stamp the commit, and copy the repository (without `runs/`, `data/raw/`,
`.venv/`, `.git/`) and the cache to the submit node:

```
bash tools/chtc/stamp_commit.sh
rsync -a --exclude runs --exclude data/raw --exclude .venv --exclude .git ./ <user>@<submit-node>:mdgan-pytorch/
```

The stamp (`src/mdgan/_commit.json`, git-ignored) travels inside `src/`, so each run's
`provenance.json` records the commit even though the cluster copy has no `.git`. If the
stamp says `"git_dirty": true`, the run used uncommitted code: commit first.

On the submit node, from the repository root: `mkdir -p logs`.

## Step 1: sizing run

```
condor_submit tools/chtc/pe_sizing.sub
condor_q                                  # wait until it leaves the queue
cat logs/sizing.out                       # environment line: torch version, "cuda available True"
```

Then read `run_sizing/metrics.jsonl` (`iterations_per_second`) and `run_sizing/events.jsonl`
(the time between a `checkpoint` event and the next metrics line is about one evaluation).
Estimated wall time for the full protocol:

    300,000 / iterations_per_second  +  30 x (one evaluation)

If `logs/sizing.err` shows an import error or `cuda available False`, fix the image or the GPU
request before going further; nothing about the full runs differs from the sizing run except
length and self-checkpointing.

## Step 2: the five-seed protocol

```
condor_submit tools/chtc/pe_paper.sub
condor_q
condor_tail -f <job id>                   # live stdout of one job
```

Each job trains seed `$(Process)` (0-4) into `run_seed$(Process)/`. Every `MAX_HOURS`
(4 by default, set in the submit file) the job saves `latest.pt` and exits with code 85;
HTCondor copies `run_seedN/` back and restarts the job, which resumes from `latest.pt`. An
eviction therefore loses at most `MAX_HOURS` of training. Exit 0 means finished; any other
exit code is a real failure (see `logs/pe_seedN.err`).

## Results

Per run, in `run_seedN/`:

| File | Contents |
| --- | --- |
| `evaluations.jsonl` | err and D (and D_rel against the MD reference) at every 10k checkpoint |
| `metrics.jsonl` | W, gradient penalty, gradient norm, step std, iterations/s every 100 iterations |
| `events.jsonl` | start, checkpoints, time limits, resumes, rollbacks |
| `snap_XXXXXX.pt` | G_z + G_Y at each checkpoint (evaluation needs only these) |
| `latest.pt` | full resume state (models, optimizers, RNG) |
| `config.json`, `dataset.json`, `provenance.json` | exactly what ran, on which data, from which commit |

Re-evaluate any snapshot anywhere, from the cache alone:

```
python -m mdgan.evaluate run_seed0/snap_150000.pt data/cache/PE.npz --seed 0
```

What to look at first: `step std` near 1 (the data-starvation warning sign is a value well
below 1 with W stuck high), err and D_rel across checkpoints, and whether W keeps rising.

## Disk and transfer

At width 2048: `latest.pt` ≈ 375 MB (31M parameters plus two Adam buffers), each snapshot
≈ 72 MB. A finished run holds `latest.pt` + 30 snapshots ≈ 2.6 GB; five seeds ≈ 13 GB.

Each self-checkpoint transfers the whole run directory, so transfers grow from ~0.4 GB to
~2.6 GB over a run. That is the simplest correct setup (an eviction restores everything),
but it may exceed what CHTC wants moved through normal file transfer or stored in `/home`.
Options, to settle with CHTC's guidance:

- keep the setup and raise `MAX_HOURS` so fewer transfers happen;
- write snapshots to `/staging` from the wrapper and checkpoint only `latest.pt` and the logs;
- keep fewer snapshots (for example every 20k) by setting `checkpoint_every`.

## To verify with CHTC before the first submission

These lines in the submit files are CHTC-specific and marked `VERIFY`:

- `universe = container` with a `docker://` image, and the image tag (needs CUDA support
  matching the GPU Lab drivers).
- `+WantGPULab` and `+GPUJobLength` (allowed values and time limits).
- `+is_resumable` together with `checkpoint_exit_code` (CHTC's self-checkpointing guide).
- Size limits for file transfer and `/home`, and when `/staging` is required.

CHTC's research computing facilitators can confirm all four in one conversation.