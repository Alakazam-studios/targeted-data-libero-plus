# Per-case results

One gzipped JSONL file per evaluation pass that the paper reports. Each file has exactly 10,030 rows, one per
LIBERO-Plus case, sorted by `case_id`:

| field | meaning |
|---|---|
| `case_id` | `suite/name` of the LIBERO-Plus case |
| `suite` | `libero_spatial`, `libero_object`, `libero_goal` or `libero_10` |
| `axis` | the case's perturbation factor: Camera Viewpoints, Robot Initial States, Language Instructions, Light Conditions, Background Textures, Sensor Noise or Objects Layout |
| `level` | the case's LIBERO-Plus difficulty level (1 to 5) |
| `success` | `true` when the episode reached the task's success predicate within the suite's step limit |
| `steps` | simulator steps of the episode, the 10 settle steps excluded |

`MANIFEST.json` gives, for each file: the model as the paper names it, the sha256 of its weights, the pass index,
the case count, the sha256 of the uncompressed rows and of the file, and the protocol. The source of the rows is
an internal evaluation record (available on request). Every source file was checked against the sha256 its record
holds, and each file's 7 axis rates were recomputed from the released fields and checked against the recorded
scores before release.

Two files, `run-a-pass0` and `run-a-pass1`, are two passes of one model (run A of the base recipe); their difference
is the pass-to-pass noise that the paper's paired rule is checked against.

The files come from three evaluation protocols. Most hold π0.5 passes under one protocol, the `protocol_id` at
the top of `MANIFEST.json`. `openvla-oft-plus` and `openvla-oft-plus-mix` ran the OpenVLA-OFT evaluation settings (8
actions per query, gripper command snapped to ±1), and `flux3-plus-only` and `flux3-mix` ran FLUX-3 Action's own
evaluator settings. Each file's entry in `MANIFEST.json` names its `protocol_id` and `protocol`. Scores from different
protocols do not compare.

`recompute.py` reads these files and prints the scores (`python3 results/recompute.py --help`).
