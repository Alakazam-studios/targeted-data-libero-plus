# Claims and how to check them

This file lists every number of the paper's claim appendix, in the appendix's order, with the public evidence that
supports it. Where the evidence is one of our internal records (training logs, data banks, evaluation receipts that
are not in this repository), the row says "internal record (available on request)".

Units: % is a success rate, in percent of the cases named; pt (percentage points) is a difference of two such rates.
The seven-axis mean is the unweighted mean of the 7 LIBERO-Plus factor success rates; the total is the successes over
all 10,030 cases. Paired differences are paired by case. "Intervals from the 40 tasks" are bootstrap intervals that
resample the 40 base tasks; `recompute.py` prints case-level intervals only, so for those intervals the point
estimate is public and the interval is an internal record.

## The results files and the paper's model names

All commands run from the repository root. `results/MANIFEST.json` gives each file's weights hash and protocol.

| File in `results/` | Model in the paper |
|---|---|
| `pi05-libero` | base π0.5 |
| `noclips-nolang` | mx |
| `noclips` | mx-lg |
| `run-a-pass0`, `run-a-pass1` | mxri-lg-1 (run A), passes 0 and 1 |
| `run-b-nolang` | mxri-2 |
| `run-b-control` | mxri-lg-2 (run B, the reference of the ablations) |
| `main-model` | mxri-lg-va-1, the main model |
| `main-model-seed2`, `main-model-seed3` | mxri-lg-va-2, mxri-lg-va-3 |
| `main-model-noviews` | mxri-lg-ad |
| `all-views` | mxri-lg-vaf |
| `cross-view-adapter` | mxri-lg-vacv |
| `cross-view-all-weights` | mxri-lg-vwcv |
| `run-c-nolang` | mxri-3 |
| `run-c-cleaned-lang` | mxri-lg-3 |
| `rewordings-in-mix` | mxrirw |
| `adapter-then-fullft`, `fullft-then-adapter`, `views-all-weights`, `teacher-rollouts`, `teacher-failures`, `views-and-layouts`, `layouts-moved-cameras` | the camera arms of the camera table, by their table labels |
| `layout-rollouts` | the third layout arm of the Limits section |
| `openvla-oft-plus`, `openvla-oft-plus-mix` | OpenVLA-OFT+, and OFT+ further trained on our mix (OpenVLA-OFT protocol) |
| `flux3-plus-only`, `flux3-mix` | FLUX-3 Action on the LIBERO-Plus training set only, and on our mix (FLUX-3 protocol) |

## Commands

`recompute.py` prints each file's 7 axis rates and seven-axis mean (no option), and with `--pair A B` the paired
difference A minus B per axis and on the seven-axis mean. For the values it does not print, these one-line commands
use the same files (standard library only):

- **T**, total of one file, % of 10,030 cases:
  `python3 -c "import gzip,json,sys;r=[json.loads(l) for l in gzip.open(sys.argv[1])];print(round(100*sum(x['success'] for x in r)/len(r),2))" results/main-model.jsonl.gz`
- **L**, one factor at one level, % of its cases, and the case count:
  `python3 -c "import gzip,json,sys;r=[json.loads(l) for l in gzip.open(sys.argv[1])];s=[x['success'] for x in r if x['axis']==sys.argv[2] and x['level']==int(sys.argv[3])];print(round(100*sum(s)/len(s),2),len(s))" results/pi05-libero.jsonl.gz "Camera Viewpoints" 5`
- **P**, paired change A minus B on one factor at one level, pt, and the case count:
  `python3 -c "import gzip,json,sys;L=lambda f:{x['case_id']:x for x in map(json.loads,gzip.open(f))};a,b=L(sys.argv[1]),L(sys.argv[2]);k=[c for c,x in a.items() if x['axis']==sys.argv[3] and x['level']==int(sys.argv[4])];print(round(100*sum(a[c]['success']-b[c]['success'] for c in k)/len(k),2),len(k))" results/main-model.jsonl.gz results/pi05-libero.jsonl.gz "Camera Viewpoints" 5`
- **K**, layout cases of one kind (`added`: the case name holds `_add_`; `moved`: every other layout case), optionally
  only the names that hold a text: %, cases, failures:
  `python3 -c "import gzip,json,sys;r=[x for x in map(json.loads,gzip.open(sys.argv[1])) if x['axis']=='Objects Layout' and ('_add_' in x['case_id'])==(sys.argv[2]=='added') and (len(sys.argv)<4 or sys.argv[3] in x['case_id'])];print(round(100*sum(x['success'] for x in r)/len(r),2),len(r),sum(not x['success'] for x in r))" results/run-b-control.jsonl.gz moved libero_goal`
- **M**, paired change A minus B on the moved-scene layout cases, pt, and the case count:
  `python3 -c "import gzip,json,sys;L=lambda f:{x['case_id']:x for x in map(json.loads,gzip.open(f))};a,b=L(sys.argv[1]),L(sys.argv[2]);k=[c for c,x in a.items() if x['axis']=='Objects Layout' and '_add_' not in c];print(round(100*sum(a[c]['success']-b[c]['success'] for c in k)/len(k),2),len(k))" results/layout-rollouts.jsonl.gz results/run-b-control.jsonl.gz`

## Claims

| Claim | Value | How to check |
|---|---|---|
| 1. Main model: total and seven-axis mean | 93.09 % (total, of 10,030 cases); 93.23 % (seven-axis mean) | `results/main-model.jsonl.gz`. `python3 results/recompute.py` prints 93.23 on the `main-model` line. recompute.py prints the per-axis rates; the total is the successes over 10,030 cases: command T prints 93.09. |
| 2. Main model minus run A, on camera and on the seven-axis mean | +7.75 pt camera; +0.80 pt seven-axis mean | `python3 results/recompute.py --pair main-model run-a-pass0`, lines Camera Viewpoints and 7-axis mean. |
| 3. Task-clustered intervals against case-level intervals | task-clustered standard error 1.4 to 1.8 times the case-level paired standard error | recompute.py prints the case-level paired standard error (`--pair`); the task-clustered bootstrap: internal record (available on request). |
| 4. Camera types, main model minus run A: orbit, distance, turn and tilt | +11.79 pt orbit (992 cases); +1.28 pt distance (313 cases); +1.02 pt turn and tilt (294 cases) | The case counts per camera type are in `camera-render/cameras/sample.py` (`FAMILY_CASES`, from `cameras/spec.py`). The camera type of each case is not a field of the results files, so the changes: internal record (available on request). |
| 5. Exclusion rule: rendered frames removed, and the median distance of the nearest kept view to its test camera | 46.2 % of the rendered frames; 11.04 degrees | `encoder-stage/encft-fladisj.sampling.json`, `exclusion`: `count` 125,079 of `bank_frames` 270,685 = 46.2 %. The median distance: internal record (available on request). |
| 6. Validation loss of the main model | 0.0146 to 0.0135 (held-out flow-matching loss) | internal record (available on request). |
| 7. Camera arms against run B (the control) | adapter arms +6.6 to +8.8 pt camera; all-weights arms +3.0 to +3.8 pt camera | `python3 results/recompute.py --pair <arm> run-b-control`, line Camera Viewpoints. Adapter arms: `main-model` +6.57, `all-views` +8.76, `adapter-then-fullft` +8.13, `fullft-then-adapter` +7.82. All-weights arms: `views-all-weights` +3.13, `teacher-rollouts` +3.75, `teacher-failures` +3.38, `views-and-layouts` +3.00, `layouts-moved-cameras` +3.50. |
| 8. Host shape of each pass | run A's two passes on a mix of 1-GPU and 4-GPU hosts; run B and the main model on 1-GPU hosts | internal record (available on request). |
| 9. Pass-to-pass (A/A) noise | 0.44 pt on the seven-axis mean, between two passes of run A | `python3 results/recompute.py --pair run-a-pass0 run-a-pass1`, line 7-axis mean (+0.44). |
| 10. Camera data set: frames, camera-type shares, ranges | 270,685 frames; type shares 0.620 / 0.196 / 0.184 (orbit / distance / turn and tilt, fractions of the draws); moves up to 1.3 times the largest test move | `encoder-stage/encft-fladisj.sampling.json` (`bank_frames`: 270685); `camera-render/cameras/sample.py`: the shares are the test case counts per type (`FAMILY_CASES`: 992, 313, 294 of 1,599 camera cases), and the ranges are its draw rules. |
| 11. Adapter: rank, parameters, schedule | rank 16; 4.7 million parameters; 2,000 steps | `encoder-stage/README.md` (rank 16, alpha 16, on `fc1` and `fc2` of every SigLIP layer; steps 2,000) and `encoder-stage/hooks/encoder_lora.py` (`LORA_RANK`). The parameter count is 27 layers × rank 16 × (in + out features of `fc1` and of `fc2`, 1,152 + 4,304 each) = 4,713,984. |
| 12. Training cost of the main model | 13.2 USD | internal record (available on request). |
| 13. Without the language stage (run B before it, against run B) | −11.71 pt language; −3.02 pt seven-axis mean | `python3 results/recompute.py --pair run-b-nolang run-b-control`, lines Language Instructions and 7-axis mean. |
| 14. Language stage: instruction pairs; accuracy of run A's language stage on its own check (not the main model's pass, see row 15); training pairs equal to a test prompt | 13,546 pairs; 97.01 % of the benchmark instructions; 0 pairs | internal record (available on request); the instruction files are not released yet (`README.md`, "What is not released"). |
| 15. Instruction mapping of the main model's language stage, in our pass | 1,489 of 1,537 language cases (96.88 %); 8,493 of 8,493 other cases | internal record (available on request): the results files do not hold the instruction the policy acted on. |
| 16. Recovery clips: episodes; steer-back frames; clip frames | 6,480 clips; steer-back frames 7.4 % of the 2,886,036 training frames; clip frames 22.5 % | internal record (available on request). |
| 17. An earlier clip recipe failed its development check | 6 of 29 Robot Start development cases after 300 steps, against 25 of 29 for the base model; three later smokes, best 19 of 29, against a bar of 20 of 29 | internal record (available on request). |
| 18. Self-training (Skill: General Perturbations) | −2.29 pt [−5.28, 0.70] against its continued run, on a development pool of 568 starts (not one of the paper's evaluators) | internal record (available on request). |
| 19. Instruction classifier | 0.603 top-1 accuracy against a bar of 0.70 (offline check) | internal record (available on request). |
| 20. Book task gap | 3.48 mm below the goal region at rest | internal record (available on request). |
| 21. Models scored on the full test set | 30 models in 31 passes on our protocol; 27 in the paper, 3 not reported (a layout arm, two models of a side study); 25 passes on other evaluators | The passes the paper quotes are the files of `results/MANIFEST.json`; the full count of passes: internal record (available on request). |
| 22. Removing the file-name text, earlier model, on vla-eval | +0.55 pt (standard error 0.34 pt, p = 0.12); 12.21 % of the 8,493 cases changed outcome, against 12.13 % for a second pass | internal record (available on request). |
| 23. Main model as served: only the adapter differs from the base recipe | 758 of 812 tensors bitwise equal, 54 adapter tensors | internal record (available on request); the main model's weights are not released yet. |
| 24. Serving rounds parameters kept in float32 at load; the actions stay the same | bit-equal actions (public model) | The rounding: github.com/Physical-Intelligence/openpi, `policy_config.py` line 55 and `gemma_pytorch.py` lines 63-80, at commit 215abfb2 (the openpi commit of our protocol, `README.md`). The action comparison: internal record (available on request). |
| 25. Public π0.5 weights in our pass | bit-exact copy of the public pi05_libero checkpoint | The checkpoint is openpi's public pi05_libero (github.com/Physical-Intelligence/openpi). The tensor comparison: internal record (available on request). |
| 26. Base π0.5 | 85.71 % seven-axis mean; 84.72 % total | `python3 results/recompute.py`, line `pi05-libero` (85.71). recompute.py prints the per-axis rates; the total is the successes over 10,030 cases: command T on `results/pi05-libero.jsonl.gz` prints 84.72. |
| 27. Each step of the recipe against base π0.5 (the ladder table) | for example +7.52 pt seven-axis mean (main model) | `python3 results/recompute.py --pair main-model pi05-libero`, line 7-axis mean (+7.52); each other step: `--pair <file> pi05-libero`. The table's intervals are from the 40 tasks: internal record. |
| 28. Per-level scores and changes (the levels table) | for example base π0.5 camera 44.4 % at level 5; main model change +36.5 pt at level 5 | The `level` field is the benchmark's own level. recompute.py prints the per-axis rates; a cell is that rate on the cases of one level: command L (`pi05-libero`, "Camera Viewpoints", 5) prints 44.35 on 487 cases, and command P (`main-model` minus `pi05-libero`, "Camera Viewpoints", 5) prints +36.55 (printed +36.5 in the table). |
| 29. Clip motion per unit command against the demonstrations | all frames: translation 1.07 to 1.21 times (x, y, z, each at the lag where each source responds), rotation 0.70 to 0.90 times (lags 0 to 2); command-size deciles 5 to 9, lags 0 to 2: translation 0.96 to 1.40 times, rotation 0.70 to 1.23 times; largest decile, rotation 0.72 to 0.88 times | internal record (available on request). |
| 30. Rewording audit: exact matches, near matches (the rewording table) | 0 of 13,546 pairs equal to a test prompt; 23 / 186 / 1,155 pairs within normalised edit distance 0.10 / 0.20 / 0.30 of a test instruction | internal record (available on request). |
| 31. Language gain by distance band, raw and level-adjusted (the bands table): mxri-lg-2 against mxri-2 | far band (distance above 0.30, 1,038 cases): +15.5 pt [12.9, 18.1]; level-adjusted +13.9 pt [11.6, 16.2] | The two models' outcomes are `results/run-b-control.jsonl.gz` and `results/run-b-nolang.jsonl.gz`; the distance band of each case needs the instruction file: internal record (available on request). |
| 32. Cleaned language file for later runs | 10,684 instruction pairs | internal record (available on request); the file is not released yet (`README.md`). |
| 33. Other papers' scores (the other-papers table) | as each paper reports them, % of cases | Each paper's arXiv version: RoboHarness arXiv 2607.18060v3; PatchWAM arXiv 2609.25961v1, its LIBERO-Plus table (PatchWAM row, 91.8 printed); ZimaBlue arXiv 2609.00188, Table 5; Hermite-VLAReg arXiv 2608.01265v2, Table 3; Kairos arXiv 2606.16533, Table 13; CAC-VLA arXiv 2607.04816v2; QuoVLA arXiv 2605.24890v2, Table 2; Anchor-Align VLA arXiv 2607.13429v2, Table 1. Totals and seven-axis means marked as ours are recomputed from the factor scores each paper prints (the total weights each factor by its LIBERO-Plus case count: 1,599, 1,550, 1,537, 1,142, 1,076, 1,601, 1,525). |
| 34. Published OpenVLA-OFT+ scores (the calibration table) | 92.8 % camera; total 79.5 % in Table 2 (79.6 % in the paper's text and in the benchmark README) | LIBERO-Plus paper, arXiv 2510.13626v3, Table 2, row "Ours"; the benchmark README at github.com/sylvestf/LIBERO-plus, commit 4976dc30. Our pass of the same checkpoint: `results/openvla-oft-plus.jsonl.gz`. |
| 35. Second evaluator (Limits paragraph only): totals agree within noise | main model 93.39 % total (second evaluator) against 93.09 % (our protocol) | 93.09 %: command T on `results/main-model.jsonl.gz`. 93.39 %: internal record (available on request). |
| 36. Second evaluator, per task: passes that fail the per-task test | 4 of 8 passes (run A, the fine-tune-first arm, views and layouts, the main model); the main model largest, a statistic of 3.091 against a threshold of 1.792 (the threshold set after the fact) | internal record (available on request). |
| 37. Second evaluator: adapter arm minus all-weights arm, same data and steps | camera +7.25 pt [4.32, 10.42]; seven-axis mean +2.05 pt [1.16, 3.03], intervals from the 40 tasks; totals 93.56 % and 91.47 % (9,384 and 9,174 of 10,030 cases) | internal record (available on request). |
| 38. Published π0.5 row (other-papers table, base model section) | 85.7 % total; camera 75.8 %, robot start 79.4 %, language 83.3 % | Hermite-VLAReg arXiv 2608.01265v2, Table 3; QuoVLA arXiv 2605.24890v2, Table 2 (identical rows). |
| 39. AllenAI leaderboard snapshot (the leaderboard figure) | 239 LIBERO-Plus rows; top seven-factor mean 92.40 % (RoboHarness) | github.com/allenai/vla-evaluation-harness at commit 41bd8a44, its `leaderboard.json` (sha256 a544ee50...). |
| 40. Layout failures by kind and level; layout arms against the control (Limits section) | run B: moved scenes 73.28 % (696 cases) against added objects 95.66 % (829 cases); 186 of its 222 layout failures in moved scenes; moved scenes fall by 18.0 pt from level 1 to level 5; libero_goal moved scenes 34.59 %; the three layout arms within 0.43 pt of run B on moved scenes; one arm's coverage of the failing cells 16.0 % of its episodes | `results/run-b-control.jsonl.gz` with command K: `moved` prints 73.28, 696, 186; `added` prints 95.66, 829, 36; `moved libero_goal` prints 34.59. The levels here are the displacement levels in the case names (command K with `_level1_` prints 84.68, with `_level5_` 66.67: −18.0 pt), not the `level` field. The arms, command M against `run-b-control`: `layout-rollouts` −0.43, `views-and-layouts` +0.14, `layouts-moved-cameras` −0.29. Coverage: internal record (available on request). |
| 41. Main model against the adapter stage without the views (the ablation table) | camera +3.38 pt [1.84, 5.00]; seven-axis mean −0.22 pt [−0.92, 0.42]; without the views against run A: camera +4.38 pt [2.11, 6.79], seven-axis mean +1.02 pt [0.38, 1.64]; intervals from the 40 tasks | `python3 results/recompute.py --pair main-model main-model-noviews` (+3.38 camera, −0.22 seven-axis) and `--pair main-model-noviews run-a-pass0` (+4.38, +1.02). recompute.py prints case-level intervals; the intervals from the 40 tasks: internal record (available on request). |
| 42. Validation loss, main model against the arm without the views | LIBERO-Plus frames 0.00905 against 0.00942; clips 0.01515 against 0.01405 (held-out flow-matching loss at step 2,000) | internal record (available on request). |
| 43. OpenVLA-OFT action head outside the data-parallel wrapper; OFT+ inferred to share it; OFT+ batch 16 | `finetune.py` line 388 calls `action_head.module`, the checkpoint keeps rank 0's head (line 640); CPU proof on 4 ranks: 4.6e-2 largest weight gap; our 8-GPU run keeps the property | github.com/moojink/openvla-oft at commit e4287e94, `finetune.py` lines 388 and 640; issue #160 of that repository ("Action head gradients are not synchronized in multi-GPU training because DDP forward is bypassed", open) and pull request #162 ("Fix DDP synchronization for continuous action heads", open, not merged); OFT+ batch: LIBERO-Plus paper, arXiv 2510.13626v3, Appendix D.4. The CPU proof and our run: internal record (available on request). |
| 44. Runs that use the cleaned rewording file (Limits section) | OpenVLA's language stage; FLUX-3's full mix; mxri-lg-3; mxrirw | internal record (available on request). |
| 45. OpenVLA continuation: training, validation, protocol | 25,000 steps, batch 64, 8 H100 GPUs; held-out L1 action error 0.0600; 8 actions per query, gripper command snapped | internal record (available on request). The passes: `results/openvla-oft-plus-mix.jsonl.gz` and `results/openvla-oft-plus.jsonl.gz` (OpenVLA-OFT protocol, `MANIFEST.json`). |
| 46. The adapter-first arm starts from base π0.5, converted (the configuration table) | 812 of 812 source tensors equal; the trainer's start state equal on 813 of 813 tensors; 114 adaptive-norm tensors rounded to bf16 in the file | internal record (available on request). |
| 47. The fine-tune-first arm's export holds its source's tensors | the export has the size of the other 26; it differs from its source by 1,152 B of pickle and zip metadata only; 813 of 813 tensors equal; strict load equal | internal record (available on request). |
| 48. Run B is one resumed training run | four training-log runs in sequence; weights 6bc90f19 saved at step 10,000 | internal record (available on request). The weights hash of `run-b-nolang` is in `results/MANIFEST.json`. |
| 49. Licences of the data (Data and licences) | LIBERO-Plus training set and assets: MIT (card metadata); its code and test cases: no licence found | Hugging Face dataset cards Sylvest/libero_plus_lerobot at revision 22c57433 and Sylvest/LIBERO-plus at revision dd2bd61b; github.com/sylvestf/LIBERO-plus has no licence file. |
| 50. FLUX-3 baseline divergence and resume | first run: largest gradient norm 67.0 at update 1,755; resumed run on the same batches: largest 2.52, no divergence; 1 of 200 replayed updates bit-equal; no gradient clipping | internal record (available on request). |
| 51. Gripper command: only its sign acts | raw and snapped commands move the fingers the same way, except a command of exactly 0, which holds them (0 of 9,294 smoke steps at exactly 0.0), on our runs (robosuite 1.4.1, MuJoCo 3.2.3) | github.com/ARISE-Initiative/robosuite, tags v1.4.0 and v1.4.1, `robosuite/models/grippers/panda_gripper.py` lines 55-57 (`np.sign(action)`, a step of 0.01 per call); github.com/sylvestf/LIBERO-plus at commit 4976dc30, `libero/libero/envs/robots/mounted_panda.py` lines 27-28 (default gripper PandaGripper); versions: `README.md`, "Evaluation protocol". The smoke step count: internal record (available on request). |
| 52. FLUX-3 validation loss at step 4,000 | on the same 256 windows (168 LIBERO-Plus, 88 clips): our mix 0.342 (LIBERO-Plus 0.411, clips 0.211); baseline 0.571 (LIBERO-Plus 0.388, clips 0.921) | internal record (available on request). |
| 53. Main model on the AllenAI evaluation code with our settings | 9,363 of 10,030 cases, 93.35 % (Wilson 95 % interval 92.85 to 93.82); seven-axis mean 93.62 %; against the same weights at 10 flow steps and 10 actions per query: +1.91 pt (McNemar p = 4.8e-10), the two settings changed together | The code: github.com/allenai/vla-evaluation-harness at commit 6cc3e1bd (vla-eval, arXiv 2603.13966). The passes: internal record (available on request). |
| 54. FLUX-3 pair: recipe and shares | one trainer and settings (4,000 updates, learning rate 1.92e-4, action heads ×5, weight decay 0.05, no clipping); the two recipe files differ only in their data keys; full-mix draws from the build counts: LIBERO-Plus 0.29634, reworded 0.15, clips 0.22385, camera views 0.32981 (fractions of the windows; derived, not sampled) | internal record (available on request). |
| 55. OpenVLA continuation: data shares | LIBERO-Plus 0.4455, clips 0.2245, camera views 0.33 (fractions of the training windows; the main model's sampling file) | `encoder-stage/encft-fladisj.sampling.json`, `drawn_share`: plus 0.4455, clips 0.2245, bank (camera views) 0.33. That the OpenVLA continuation used these shares: internal record (available on request). |
| 56. Test use on other evaluation code | full passes of all 10,030 cases: 8 on a second evaluator, 10 on the AllenAI evaluation code, 3 on an RLinf harness, 4 on a second RLinf-based harness; not counted: a 2,000-case sample and a 568-start self-training pool | internal record (available on request). |
| 57. Test use in the base recipe's own campaign | at least 15 full passes not counted above: 13 leaderboard reports on the RLinf evaluator, 2 DAgger passes on the AllenAI code | internal record (available on request). |
| 58. Provenance of the language data | the 5 writing agents received the 40 task instructions and a style guide and read no file; 0 of the 1,509 language test instructions in their prompts or context; 0 of 13,146 non-identity training rows equal one; the style guides written after their author saw the test instructions | internal record (available on request). |
| 59. Near-test wording in the cleaned file | 1,887 of the 10,684 training rows (17.7 %) within normalised edit distance 0.35 or token Jaccard 0.50 of a language test instruction (identity rows exempt); 786 of 1,537 language cases (51 %) with such a neighbour | internal record (available on request). |
