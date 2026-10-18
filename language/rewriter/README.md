# The open rewriter of the leaderboard system

Our leaderboard system is mxri-va behind an open test-time rewriter. Before a pass, a language model rewrites each
benchmark instruction once, and the policy acts on the rewrite. This folder holds the rewriter's rulebook, the exact
system prompt it received, and the two rewrite tables that the paper's passes read (Section VIII of the paper).

| File | What it holds |
|---|---|
| `rules-v1.md` | the rulebook: its provenance, the vocabulary and the 9 rules |
| `system-prompt.txt` | the exact system prompt built from `rules-v1.md` (prompt `RB`, text input); no final newline, the bytes as sent |
| `table-gpt.json` | the GPT-6-astra rewrite of every case (10,030), read by both passes of the leaderboard system |
| `table-qwen.json` | the Qwen3.5-9B rewrite of every case (10,030), read by the open-weight comparison pass |
| `SHA256SUMS` | the sha256 of each file above |

Each table is `{"_doc": ..., "rewrites": {"<suite>/<case name>": "<rewrite>"}}` (the GPT table also has `_sources`,
the three partial tables it was merged from). The keys are the `case_id` values of `results/`.

## What the rewriter sees

- The system prompt: the rules, and a vocabulary that is the word list and phrase templates of the 40 LIBERO training
  instructions (their objects, their action templates such as "pick up ... and place it on ...", and their position
  words).
- The user message: the case's own instruction as the benchmark ships it (its `task.language`, the file-name text
  included). No image, no history: one fresh call per case.

The prompt holds neither the 40 task instructions nor a mapping onto them. At test time the rewriter reads each case's own instruction, which on the six non-language factors is a task instruction with the file-name text. The author of the rulebook had not
read the language test instructions, and its rules were tuned on 860 held-out paraphrases only. None of the 20
quoted strings of the rules and none of the 49 vocabulary items equals a language test instruction or comes within
the cleaned-file rule of one (`../README.md`, "How the cleaned file was made"). This check was run on the benchmark's
test instructions, which this repository does not hold.

## How to run the leaderboard system

1. **With the table (what the paper's scores used).** Load mxri-va (huggingface.co/alakazamworld/pi05-libero-plus-targeted,
   the root of the repository) with openpi's `pi05_libero` config. For each case, send `rewrites["<suite>/<case name>"]`
   from `table-gpt.json` to the policy in place of the case's instruction. Everything else follows the protocol of the
   top-level README (5 flow steps, 5 actions per query, seed 42).
2. **With a call to the rewriter (new instructions).** Send `system-prompt.txt` as the system prompt and the raw
   instruction as the only user message, one fresh call per instruction. Clean the reply: strip surrounding quotes,
   then a final period, then lowercase. Nothing else.
   - GPT: `gpt-6-astra`, its default reasoning effort, text only.
   - Qwen: `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, thinking off
     (`enable_thinking: false`), greedy (temperature 0), at most 64 new tokens, bf16.

   A new GPT call can return another rewrite than the table holds. The scores were measured with the tables, not with
   new calls.

## Checks

```
sha256sum -c language/rewriter/SHA256SUMS
python3 -c "import gzip,json,sys;t=json.load(open(sys.argv[1]))['rewrites'];d={x['case_id']:x['decoded'] for x in map(json.loads,open('language/decoded-prompts.jsonl'))};a={x['case_id']:x['axis'] for x in map(json.loads,gzip.open('results/main-model.jsonl.gz'))};k=[c for c in t if a[c]!='Language Instructions'];print(sum(t[c]==d[c] for c in k),len(k))" language/rewriter/table-gpt.json
```

The second command prints `8473 8493`: on the cases outside the language factor, the GPT rewrite equals the case's own
task instruction for 8,473 of 8,493 cases (99.76 %). With `table-qwen.json` it prints `7978 8493` (93.94 %).

On the 860 held-out paraphrases, judged by Claude Sonnet 5.5, GPT keeps the task of 99.5 % and Qwen of 87.6 %
(internal record, available on request).

## Licence

The tables are our outputs, under the Apache License 2.0 with the rest of this repository. The rewrites of the 1,537
language cases are derived from the benchmark's test instructions, which are not included; `NOTICE` says more.
