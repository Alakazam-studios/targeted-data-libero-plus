# Language data and checks

Files for Section VIII of the paper (Skill: Language), its distance bands table, and the open rewriter of the leaderboard system.

| File | What it holds |
|---|---|
| `distance.py` | the paper's distance from a language test instruction to its nearest training pair, and the bands (its header defines the distance exactly) |
| `case-distances.jsonl` | `distance.py`'s output for the 1,537 language cases: `case_id`, `distance`, `band` |
| `decoded-prompts.jsonl` | the pass of mxri-lg-va-1 (the closed-set model, the main model of the paper's first version), one row per case (10,030): `case_id`, `decoded` (the task instruction its language stage decoded, or null when the decode is not one of the 40), `own_task` (true when it is the case's own task instruction) |
| `rewriter/` | the open rewriter of the leaderboard system: its rulebook, its exact system prompt, and the GPT-6-astra and Qwen3.5-9B rewrite tables that the passes read, with their sha256 (`rewriter/README.md`) |
| `cleaned-train.jsonl`, `cleaned-val.jsonl` | the cleaned instruction file of the later runs: 10,684 training rows (10,284 rewordings and 400 identity rows) and the validation rows, as `instruction`, `task`, `suite`, `source` |

## Checks

```
python3 -c "import json; r=[json.loads(l) for l in open('language/decoded-prompts.jsonl')]; print(sum(x['own_task'] for x in r), len(r))"
```

prints `9982 10030`: 1,489 of the 1,537 language cases and all 8,493 other cases.

`distance.py` needs the benchmark's language prompts, which this repository does not hold: the LIBERO-Plus fork has
no licence. Build `--cases` from your own copy of the benchmark (`{"case_id", "instruction"}` per language case). The
13,546-pair training file of the paper's language stage (sha256 bdb4dd12) is an internal record, available on
request. On it, `distance.py` prints 24, 483 and 1,030 cases per band, and 11.76 % of the captions and 8.37 % of the
rewordings within 0.30.

## How the cleaned file was made

Rule, fixed before any near-match count was printed: drop each non-identity row that equals a language test
instruction, is within an edit distance of 0.30 of one, has a word-set Jaccard of 0.60 or more with one, or contains one
whole; and drop every caption. Identity rows (LIBERO's 40 instructions) stay. Claude Sonnet 5 wrote the rewordings.
