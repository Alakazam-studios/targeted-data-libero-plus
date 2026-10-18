# Encoder stage: training configuration of the camera stage of mxri-lg-va-1 and mxri-va

The stage that adapts the vision encoder to the rendered camera views. It starts from a π0.5 checkpoint fine-tuned on
the base recipe (run A in the paper), freezes every weight, and trains a low-rank adapter on the MLPs of the SigLIP
vision encoder for 2,000 steps on a mix of the LIBERO-Plus training set, recovery clips and the rendered views.

The hooks are copied from our internal work repository, at the version that trained mxri-lg-va-1. For this release
only their comments, docstrings and messages were edited, and the two descriptive strings that `build_mix.py` writes
into its receipt (`rule`, `known_difference_ruling`); the code is otherwise unchanged. The first sha256 is the one
the training run's specification pins (the files as trained; an internal record), the second the released file's:

| file | sha256 as trained | sha256 as released | what it does |
|---|---|---|---|
| `hooks/encoder_lora.py` | `8d06347489ca48373f2223df1d8941b212a9be55f978b25be54ecde24f3d7177` | `6130d7858e6f591e44b92990559aa6aac7286f5894e426428a97d2e02a3c9506` | the freeze set and the LoRA adapter (rank 16, alpha 16, dropout 0, float32) on `fc1` and `fc2` of every SigLIP encoder layer, frozen-weight checks and a checked save; applied to an RLinf (commit `d5ec669`) FSDP trainer |
| `hooks/build_mix.py` | `e07fe64b2ec1e0cee54ff3186dde0e288f7988341645b59df2f20c0cc6c1af01` | `673697a50b6d5053227140a5e0e997520df57d1b78be246fa341902b53ba6807` | the data hook: appends the rendered-view training set to the base mix and lists the excluded view frames (training weight 0) |
| `hooks/encft_metrics.py` | `9c0fd00403db18cff2b23f404da41fac268808d98a827041e12da37d24e72a47` | `585fc737c0e0b035b7c1b6b96f62b9fa827967867ab45c452327c3b3ca9b6033` | per-sample flow-matching loss split by frame source, camera family and bin, for the training and validation logs |

`encft-fladisj.sampling.json` holds the sampling weights of the three sources on the effective training frames and
the frame counts they come from.

## Settings of mxri-lg-va-1's run

| setting | value |
|---|---|
| start | run A of the base recipe (weights sha256 prefix `854ea3d0`), trainer mode `continue` |
| arm | `fla` (`ENCFT_ARM=fla`): every base weight frozen, LoRA on the SigLIP MLPs |
| steps | 2,000 |
| learning rate | 5e-4, 500 warm-up steps, minimum 5e-6 |
| sampling weights (per frame) | LIBERO-Plus training set 1.0, recovery clips 1.7408, rendered views 11.3408 |
| drawn share | LIBERO-Plus 0.4455, recovery clips 0.2245, rendered views 0.33 |
| trainer override | `actor.fsdp_config.use_orig_params=true` |
| validation split | one held-out split file (sha256 `9d70d969...`), validation loss logged per data group |

mxri-va, the policy of the leaderboard system, ran the same stage with the same data and settings, started from
mxri-2 (run B before its language stage, weights sha256 prefix `6bc90f19`) instead of run A; its run record is internal
(available on request).

The trainer and its data pipeline are not part of this release; the hooks show what the stage changes in it.
