# Find the Missing Skill, Build the Data: Targeted Data Sets for π0.5 on LIBERO-Plus

Code and per-case results for the paper of that title. Paper: [`paper/paper.pdf`](paper/paper.pdf) (the current
version; the arXiv submission is in moderation, link to follow).

Our main system is the leaderboard system: mxri-va, π0.5 fine-tuned on the LIBERO-Plus demonstrations, our recovery
clips and a camera stage (no language stage), behind an open test-time rewriter (GPT-6-astra) that rewrites each
instruction before the policy acts. It scores 93.42 % [93.03, 93.82] seven-axis mean on LIBERO-Plus, the mean of two
passes, with its 95 % interval over the 10,030 cases; +7.71 [5.82, 9.75] points over base pi0.5, with its interval from
the 40 tasks. On identical prompts (base pi0.5 behind the same rewriter, 88.23 %), our training adds +5.19 [3.77, 6.75]
points. The rewriter's prompt holds the word list and phrase templates of the
40 training instructions, but neither the 40 task instructions nor a mapping onto them.

The main weights are those of mxri-va: https://huggingface.co/alakazamworld/pi05-libero-plus-targeted (under the Gemma
Terms of Use), at the root of that repository. The same repository keeps mxri-lg-va-1, the closed-set model of the
paper's first version (93.23 %), in its folder `mxri-lg-va-1/`, for reference.

This repository lets you check every score in the paper and re-run its method parts:

- `CLAIMS.md`: every number of the paper, with the public evidence for it or a note that it is an internal record.
- `results/`: the outcome of every LIBERO-Plus case (10,030 cases, one trial each) for each evaluation pass that the
  paper reports, 36 passes of 29 weight files (two of them run base π0.5 with changed prompts, three run mxri-va
  behind a rewriter, and two run base π0.5 and mxri-2 behind the GPT rewriter). Every score and paired difference in the paper follows from these files; the intervals from the
  40 tasks are an internal record (see `CLAIMS.md`).
- `camera-render/`: the camera re-render generator: the camera sampler, the camera transform that matches the
  benchmark's camera code, and the render driver.
- `language/`: the open rewriter of the leaderboard system (`language/rewriter/`: its rulebook, its system prompt and
  the GPT and Qwen rewrite tables, with sha256), the distance of each language test case to the training pairs,
  mxri-lg-va-1's decoded prompts, and the cleaned instruction file of the later runs.
- `encoder-stage/`: the training configuration of the encoder stage: the LoRA hook on the SigLIP MLPs, the data-mix
  hook, and the sampling weights of mxri-lg-va-1's camera stage.

## Layout

```
LICENSE                 Apache-2.0 (the code and the results)
NOTICE                  attribution and third-party notes
results/
  README.md             fields and sources of the per-case files
  MANIFEST.json         per file: model, weights sha256, protocol, case count, rows sha256
  recompute.py          recomputes the scores and paired differences (Python standard library only)
  *.jsonl.gz            one file per pass: case_id, suite, axis, level, success, steps
camera-render/
  README.md
  cameras/              transform.py, spec.py, sample.py, test_cameras.py, testset.json
  pairs/driver.py       the render driver
language/
  README.md             the files and their checks
  rewriter/             rules-v1.md, system-prompt.txt, table-gpt.json, table-qwen.json, SHA256SUMS, README.md
  distance.py           the distance to the nearest training pair, and the bands
  case-distances.jsonl  per language case: distance and band
  decoded-prompts.jsonl per case of mxri-lg-va-1's pass: decoded task instruction
  cleaned-train.jsonl, cleaned-val.jsonl   the cleaned instruction file
encoder-stage/
  README.md             the stage's settings
  hooks/                encoder_lora.py, build_mix.py, encft_metrics.py
  encft-fladisj.sampling.json   mxri-lg-va-1's sampling weights and frame counts
```

## Recompute the headline numbers

```
python3 results/recompute.py
python3 results/recompute.py --pair mxri-va-gpt-pass0 main-model
python3 results/recompute.py --pair main-model pi05-libero
python3 results/recompute.py --pair run-a-pass0 run-a-pass1
```

The first command prints each model's 7 axis success rates and their unweighted mean (the 7-axis mean, in percent).
The leaderboard system's two passes are the rows `mxri-va-gpt-pass0` (93.51) and `mxri-va-gpt-pass1` (93.33); their
mean is the headline 93.42 %. The second command prints the leaderboard system's first pass minus mxri-lg-va-1 (the
`main-model` file), paired by case. The third prints mxri-lg-va-1 minus base π0.5 (for example +7.52 percentage points
on the 7-axis mean, 95 % interval [+6.82, +8.22]). The fourth is the noise between two passes of one model (+0.44
percentage points on the 7-axis mean). `recompute.py --help` gives the formulas.

## Evaluation protocol

Every π0.5 pass uses one protocol (id `ee3f0803c23b1ab86ab8f8e76e6e582d6a37d7d1bd8a42eb4524e2a66ef7210e`); the OpenVLA-OFT+
and FLUX-3 passes use their own evaluator settings, named per file in `results/MANIFEST.json`. Compare these
scores only with scores measured under the same settings:

| setting | value |
|---|---|
| LIBERO-Plus commit | `4976dc30028e805ff8094b55501d532c48fec182` |
| LIBERO-Plus assets revision | `dd2bd61b7d9a6fef1abc52d606e983b41886a149` |
| MuJoCo | 3.2.3 |
| robosuite | 1.4.1 |
| cases | 10,030 (LIBERO-Plus case list, sha256 `faa87cce...a999e6`), 1 trial per case |
| environment seed | 42 |
| flow-matching steps | 5 |
| actions executed per query | 5 (of an action chunk of 10) |
| gripper | raw: the policy's continuous gripper command is passed to robosuite unchanged (robosuite uses its sign) |
| episode end | first success, or the suite's step limit (spatial 220, object 280, goal 300, long 520) |
| settle | 10 steps of zero motion with the gripper open before the policy starts; `steps` in the results excludes them |
| images | rendered at 256 px, rotated 180 degrees, resized with padding to 224 px |
| instruction | the task's instruction as shipped; for the mxri-va passes, its rewrite from `language/rewriter/` |
| policy stack | openpi `215abfb2`, torch 2.7.1 |

## What is not released, and why

We release generators, not rendered frames. The frames show LIBERO scenes, and some of the assets in those scenes
carry licences that restrict redistribution: the NVIDIA HOPE objects are under CC BY-NC-SA 4.0, and the TurboSquid
models may be rendered for machine learning only with TurboSquid's permission, which the LIBERO authors hold. Releasing
the generator lets you build the same data from your own copy of the benchmark, under the licences you hold. This
section will change if the asset licences become clearer.

| what | why not | what we release instead |
|---|---|---|
| The re-rendered camera bank (270,685 frames, 4.46 GB) | the frames show the LIBERO assets above; they were rendered with the LIBERO-Plus simulator fork, which has no licence file | `camera-render/`, the generator |
| The recovery clips (6,480 clips) | the same assets | the clip generator, once its licence is set |
| The instruction files of the language stage (13,546 pairs, and the cleaned 10,684) | their licence is not chosen yet | to follow |
| The evaluation harness | too much code to support, and it is not the paper's contribution | the protocol settings above and the per-case results |
| The LIBERO-Plus fork code | it has no licence file; we never redistribute it | its upstream commit, in the protocol table |

## Licence

The code, the results and the rewrite tables in this repository are under the Apache License 2.0 (`LICENSE`). The
model weights, on Hugging Face, are under the Gemma Terms of Use, not under Apache-2.0 alone.
