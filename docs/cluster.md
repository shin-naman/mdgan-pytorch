# Running on CHTC (HTCondor GPU Lab)

Independent single-GPU jobs through HTCondor on the HTC system (access point `ap2002`).
Slurm/HPC is for multi-node MPI work and is not used. Written against CHTC's guides on job
submission, GPUs, checkpointing, large data and Docker (October 2026).

## What gets submitted

| File | Role |
| --- | --- |
| `tools/chtc/pe_sizing.sub` | One 2,000-iteration paper-scale job. Submit first; it measures speed, memory and disk. |
| `tools/chtc/pe_paper.sub` | The paper's protocol: 5 jobs, seeds 0-4, 300k iterations each. |
| `tools/chtc/run_train.sh` | The job's executable: prints the environment, runs `mdgan.train`, passes the exit code through. |
| `configs/pe_paper.json` | The protocol written out in full (n_s = 1, n_z = 32, M = 64, width 2048, lr 1e-4, c = 7, K = 4). |
| `tools/chtc/stamp_commit.sh` | Records the commit in `src/mdgan/_commit.json`; run on the laptop before rsync. |

Each job carries three inputs (a few MB): `src/` (the package, on `PYTHONPATH`; nothing is
installed), `pe_paper.json` and `PE.npz` (preprocessed displacements: no `.xtc`, no mdtraj).
Software comes from the container `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`
(Python 3.11, torch with CUDA, numpy); CHTC recommends PyTorch's `-runtime` images.

## How a job runs

1. `condor_submit` (on the access point) queues the job.
2. HTCondor matches it to a GPU Lab machine with 1 GPU, 2 CPUs, 16 GB RAM, 10 GB disk.
3. That machine pulls the container image and creates an empty working directory holding the
   three inputs and the wrapper.
4. The wrapper trains into `run_<name>/` inside that directory. Progress lines go to stdout,
   which you can watch live with `condor_tail`.
5. On exit 0, HTCondor copies `run_<name>/` back to the directory you submitted from (in
   `/home`), and the `.out`/`.err` files to `logs/`. The `.log` file is written on the access
   point throughout.
6. Safety net: if training is not finished after `MAX_HOURS`, the job saves `latest.pt` and
   exits 85. HTCondor keeps `run_<name>/` in `/spool`, puts the job back in the queue, and the
   restarted job (possibly on another machine) resumes from `latest.pt`.

Run names include `$(Cluster)`, the ID HTCondor assigns per submission, so a later submission
never overwrites earlier results.

## Where things live

| Item | Size | Location | Why |
| --- | --- | --- | --- |
| Code (rsync copy, no `.git`) | < 1 MB | `/home/nkansal/mdgan-pytorch` | Submit from here (CHTC: submit only from `/home`). |
| `data/cache/PE.npz` | ~6 MB | `/home/nkansal/mdgan-pytorch/data/cache/` | < 1 GB, normal file transfer. |
| One finished run | ~2.6 GB, largest file 375 MB | `/home/nkansal/mdgan-pytorch/run_c<cluster>_seedN/` | Every file < 1 GB, so `/home`, not `/staging`. |
| Five runs | ~13 GB | same | Fits the 40 GB `/home` quota. Copy off and delete: CHTC does not back up. |

`/staging` (100 GB) is not needed: CHTC reserves it for individual files over 1 GB.

## Getting code and data onto the access point

One method only: **rsync from your laptop**. It copies the code, the commit stamp and
`data/cache/PE.npz` in one step (the cache is git-ignored, so it never travels through git).
Do not also clone the repository on the access point: the copy has no `.git`, and the stamp
written on your laptop is the only record of which commit ran.

Once, if the cache does not exist yet (on the laptop):

```
python -m mdgan.data.preprocess data/raw/wat_in_polymer_mdgan/MD_data/PE --out data/cache/PE.npz
```

Before every submission, on the laptop, in the repository root:

```
git status                                  # clean: commit first
bash tools/chtc/stamp_commit.sh             # must report git_dirty false
rsync -av \
  --exclude runs --exclude data/raw --exclude .venv --exclude .git \
  --exclude __pycache__ --exclude .pytest_cache --exclude .ruff_cache \
  ./ nkansal@ap2002.chtc.wisc.edu:mdgan-pytorch/
```

Never add `--delete`: finished runs live in `~/mdgan-pytorch/run_c…` on the access point and
do not exist on the laptop, so `--delete` would erase them.

Then on the access point (`ssh nkansal@ap2002.chtc.wisc.edu`):

```
cd ~/mdgan-pytorch
mkdir -p logs
ls data/cache/PE.npz src/mdgan/_commit.json   # both must exist
```

Optional check that the cache arrived intact: `md5sum data/cache/PE.npz` on the access point
equals `md5 data/cache/PE.npz` on the Mac.

## Every submission

### Step 1: sizing run

```
condor_submit tools/chtc/pe_sizing.sub
condor_watch_q                        # idle -> running -> gone; Ctrl+C to stop watching
condor_tail -f <job id>               # live progress lines while it runs
```

When it is done:

```
cat logs/sizing_c<cluster>.out        # environment line: torch version, "cuda available True"
tail -20 logs/sizing_c<cluster>.log   # memory and disk actually used vs requested
cat run_sizing_c<cluster>/metrics.jsonl | tail -3          # iterations_per_second
cat run_sizing_c<cluster>/evaluations.jsonl                # two 8192-chain evaluations ran
```

Full-run time ≈ 300,000 / iterations_per_second + 30 × (time of one evaluation; read it from
the `checkpoint` events in `run_sizing_c<cluster>/events.jsonl`). Pick `+GPUJobLength` and
`MAX_HOURS` in `pe_paper.sub` from that (short 12 h / 11; medium 24 h / 23; long 7 days / 160).
If the log shows memory or disk far below the request, lower the request: smaller requests
match more machines.
Make these edits on the laptop, commit, then stamp and rsync again, so the run's recorded
commit includes the submit file it used.

### Step 2: the five seeds

```
condor_submit tools/chtc/pe_paper.sub
condor_q -nobatch
condor_tail -f <job id>
```

`condor_q -hold` explains any held job; `condor_rm <job id>` removes one.

### Step 3: results

Each finished job leaves `run_c<cluster>_seedN/` in `~/mdgan-pytorch`:

| File | Contents |
| --- | --- |
| `evaluations.jsonl` | err, D, D_rel at every 10k checkpoint |
| `metrics.jsonl` | W, gradient penalty, gradient norm, step std, iterations/s every 100 iterations |
| `events.jsonl` | start, checkpoints, time limits, resumes, rollbacks |
| `snap_XXXXXX.pt` | G_z + G_Y at each checkpoint (all evaluation needs) |
| `latest.pt` | full resume state |
| `config.json`, `dataset.json`, `provenance.json` | exactly what ran, on which data, from which commit |

Copy them to your laptop and then delete them from CHTC:

```
rsync -av nkansal@ap2002.chtc.wisc.edu:~/mdgan-pytorch/run_c<cluster>_seed* runs/chtc/   # on the laptop
```

Re-evaluate any snapshot anywhere:

```
python -m mdgan.evaluate runs/chtc/run_c<cluster>_seed0/snap_150000.pt data/cache/PE.npz --seed 0
```

What to look at first: step std near 1 (well below 1 with W stuck high is the data-starvation
sign), err and D_rel across checkpoints, and whether W keeps rising.

## Notes

- Do not run training on the access point; the sizing job is the test.
- `condor_ssh_to_job` is disabled on shared GPUs; `condor_tail` is the way to look inside.
- The first job on a machine pulls the container image (several GB), so start-up can take a while.
- Evictions: a job restarts from the last checkpoint HTCondor stored (the last exit 85), not
  from `latest.pt` in the lost working directory. Within its GPU Lab length a job is not
  evicted, so in practice this matters only for the MAX_HOURS safety net. To exercise the safety
  net once, submit the sizing job with `environment = "MAX_HOURS=0.1"`: it should exit 85
  after about six minutes, be requeued, and resume from `latest.pt`.