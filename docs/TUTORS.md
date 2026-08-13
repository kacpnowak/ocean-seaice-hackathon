# For the tutors

Kacper and Nils. Participants do not read this file -- they start at
[docs/00_start_here.md](00_start_here.md). This is the operational side: what has
to be true before the room fills up, what only a tutor can do, and the one-line
answer to each failure we have already seen somebody hit.

---

## 1. Before day one

| | | |
|---|---|---|
| ☐ | **Participants are in the `hclimrep` group** | **Only you can do this, and nothing works without it.** |
| ☑ | Prepared data, 92 GB, 1993--2025 | `data/glorys_1deg_prepped/` |
| ☑ | Statistics, masks, climatology | `oceanarches/stats/` |
| ☑ | Four small trained runs | `modelstore/{task6_tiny,ocean_tiny,seaice_tiny,seaice_isolated_tiny}` |
| ☑ | The pre-trained `large` checkpoint | `modelstore/large_pretrained`, 75 000 steps -- [section 3](#3-the-pre-trained-large-model) |
| ☑ | Its scorecard | `evalstore/large_pretrained/report.md` |
| ☑ | Public repository | `git@github.com:kacpnowak/ocean-seaice-hackathon.git` |

### The group is the whole gate

```
/e/scratch/hclimrep   drwxrws---   root:hclimrep
```

Everything below it is world-readable, so **any member of `hclimrep` can read the
kit, the data and the checkpoints, and a non-member can read none of it** -- not
the clone, not the 92 GB of prepared data, not the statistics. There is no
partial state and no useful error: a non-member gets `Permission denied` on the
`git clone` in [docs/01 §1.1](01_setup.md) and stops there.

Check the room against the group before the session, not during it:

```bash
getent group hclimrep
```

---

## 2. The shared store

There is no separate share. **This checkout is the shared store**, and
[docs/01 §1.1](01_setup.md) sends participants straight at it:

```
/e/scratch/hclimrep/nowak2/hackathon-ocean-sea-ice
```

They `git clone` it (about 7 MB -- everything large is git-ignored), run
`make setup` and `make stats` in their own copy, and **symlink the shipped runs
in one by one**:

```bash
SHARED=/e/scratch/hclimrep/nowak2/hackathon-ocean-sea-ice/modelstore
for run in large_pretrained task6_tiny ocean_tiny seaice_tiny seaice_isolated_tiny; do
    ln -s $SHARED/$run modelstore/$run
done
```

One by one, **not** `ln -s $SHARED modelstore`. The rehearsal did the latter
anyway, which makes every run name subsequently trained resolve into this
directory -- shared with the whole room. That is why
`oceanarches/paths.py` refuses to write to or delete anything reached through a
symlink, and why the four names above are in `SHIPPED_RUNS` and cannot be
overwritten even where they are a directory of somebody's own.

Your `modelstore/` is mode `755` and owned by you, so the room reads it and
cannot write it. Nothing enforces that from inside the kit -- it is the
filesystem. **Do not loosen it.**

### Adding a run to the shipped set

Four places, and all four matter:

1. put the run in `modelstore/<name>/` (checkpoint + `config.yaml`),
2. add `"<name>"` to `SHIPPED_RUNS` in [oceanarches/paths.py](../oceanarches/paths.py)
   -- otherwise the kit will happily let a participant delete it,
3. add it to the `for run in ...` loop in [docs/01 §1.1](01_setup.md),
4. `bash scripts/publish.sh "..."` ([section 4](#4-publishing)).

`make test` catches steps 2 and 3 if you forget: `test_run_store.py` asserts that
`SHIPPED_RUNS` and the `for run in ...` loop in docs/01 are the same list, and
that every name in it is really in the store. Steps 1 and 4 are yours.

---

## 3. The pre-trained `large` model

**Done, and shipped as `large_pretrained`.** 75 000 steps of `module=large` on
4 nodes x 4 GH200 (16 ranks), `batch_size 1` per rank, `dataloader=glorys`,
`lr 1e-4` with a 10 000-step warm-up. About 1.3 it/s per rank -- roughly 18 hours
of wall clock across two launches, because that is more than the 12-hour QOS
limit, which is the whole reason
[section 3.2](#32-relaunching-after-the-wall-clock) exists. Final `val_loss`
0.236, still falling at the end of the schedule.

On the held-out test years it beats persistence on **every variable at every
lead time** out to 10 days -- 36% on day-1 SST, 53% on sea surface height, 35%
on sea-ice concentration -- and stays better than climatology for 23 days on SST
and 29 on sea-ice concentration. `evalstore/large_pretrained/report.md` is the
full scorecard, with 10 figures and 6 animations beside it; regenerate it with
`make eval NAME=large_pretrained`.

### 3.1 The learning rate is not a free knob

The first attempt ran at `lr 2e-4` and **diverged between step 15 000 and
20 000**: healthy at 15 000 (loss 1.96), destroyed by 20 000 (loss 1.05e8,
output std 652 against a target of 0.69). Gradient clipping at 1.0 was on the
whole time and did not save it, and the run then trained for eight more hours
producing nothing. `1e-4` with a 10 000-step warm-up is the setting that holds.

Two things came out of that and are now permanent:

* `DivergenceGuard` aborts the run instead of burning the rest of the
  allocation (`++module.module.divergence_guard=False` turns it off).
* Do not raise the learning rate for the `large` preset without watching the
  first 20 000 steps.

### 3.2 Relaunching after the wall clock

The QOS limit is 12:00:00, so a full `large` run needs at least two launches.
The script is built for it -- `sbatch scripts/pretrain_large.slurm` resumes from
the newest checkpoint on its own, and carries `num_training_steps` with it so the
cosine schedule does not climb back toward peak on the second launch.

**A relaunch used to be refused by the previous launch's lock.** SLURM enforces
the wall clock with SIGKILL, so nothing cleans `modelstore/<name>/.training.lock`
up, and the guard believed a lock from another host for a day. Two 4-node
relaunches died an hour in with fifteen copies of
`DistNetworkError: The client socket has timed out`, which is what it looks like
when the guard refuses on rank 0 -- the rendezvous master -- and the other
fifteen ranks wait for a store that is never created.

Both halves are fixed: the lock now asks `squeue` whether the job that wrote it
still exists, and `srun --kill-on-bad-exit=1` tears the step down the moment any
rank exits. **If you see a rendezvous timeout again, read the top of the log
before blaming the network** -- a rank-0 refusal is now the last thing in it.

### 3.3 Staging the finished checkpoint

Already done for `large_pretrained` -- this is the recipe for the next one.
Follow [section 2](#adding-a-run-to-the-shipped-set): the run directory, then
`SHIPPED_RUNS`, then the loop in docs/01, then a release. `load_module` takes
the newest checkpoint in `checkpoints/`, so the run directory is what ships and
the intermediate checkpoints are dead weight rather than a problem
([section 3.4](#34-housekeeping)).

Participants fine-tune it on one GPU with

```bash
sbatch --export=ALL,FROM=large_pretrained,MODULE=large,NAME=my_finetune \
    scripts/finetune.slurm
```

**`MODULE=large` is the part they will forget.** It defaults to `tiny`, and
`load_ckpt` copies weights tensor by tensor, so the default against a `large`
checkpoint is 343 lines of `size mismatch` -- after the queue clears. The script
compares `emb_dim` and refuses before spending the GPU time, which is the whole
reason that guard is in it. See [docs/04](04_scaling_finetuning.md).

### 3.4 Housekeeping

`modelstore/` currently holds ~102 GB, of which two directories are almost all
of it:

* `large_pretrained_diverged_lr2e-4/` -- 47 GB, the failed `2e-4` run. Keep one
  checkpoint if you want the evidence; the other nine are worth nothing.
* `large_pretrained/` -- 78 GB, fifteen checkpoints. Participants symlink the
  directory and `load_module` takes the newest, so the other fourteen cost them
  nothing and cost you 72 GB. Thin it if you want the space; nothing depends on
  them once step 75 000 is written.

Plus a handful of rehearsal runs (`probe`, `t10_*`, `t12_*`, `docs_tiny`) that
are not in `SHIPPED_RUNS` and are nobody's dependency.

---

## 4. Publishing

`main` here is the development history and stays private. The public repository
sees **one squashed commit per release**, built from `main^{tree}`:

```bash
bash scripts/publish.sh --dry-run "Add the large pre-trained checkpoint"
bash scripts/publish.sh          "Add the large pre-trained checkpoint"
git push origin public:main
```

It refuses a dirty tree, the wrong branch and a tainted `public` branch, and
shows you the file list before doing anything. `git diff main public` is empty by
construction, so the public tree is exactly `main`'s tracked tree -- nothing
git-ignored can leak into it.

---

## 5. When somebody is stuck

The kit was rehearsed cold, end to end, before it shipped. Every expensive
failure that turned up was a *silence*, and each one now refuses out loud -- so the fastest move is
almost always **read the message**, which names the override that fixes it. The
ones worth recognising on sight:

| What they say | What happened | Answer |
|---|---|---|
| "it trained but there's no checkpoint" | `max_steps` below `save_step_frequency` | the startup guard now refuses this outright |
| "my override did nothing" | a bare `++max_steps=10` on a `make` command line is parsed as a make variable | `HYDRA_ARGS="++max_steps=10"` |
| "it says it's already training" | a lock, and now only ever a live one | `squeue -j <id>`, then delete it |
| "`make doctor` fails" | `make stats` has not been run, 8 minutes, once per clone | [docs/01 §1.1](01_setup.md) |
| "resume crashes on the optimizer state" | fixed -- torch 2.6 `weights_only` versus OmegaConf | none, but check they are on a current clone |
| "the loss went to NaN" | see [3.1](#31-the-learning-rate-is-not-a-free-knob) | lower the learning rate; the guard already stopped the run |
| "it's training on the login node" | no allocation | the startup guard pauses and prints the `srun` line |
| "shapes don't match after I changed `n_depths`" | the backbone needs exactly 8 latent levels, so every preset uses 13 depths | [docs/02](02_data_and_masking.md) |

`make doctor` answers most environment questions before you have to.
