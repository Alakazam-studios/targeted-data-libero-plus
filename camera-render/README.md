# Camera re-render generator

The code that draws the training camera views and renders the LIBERO demonstrations from them (the paper's camera
data). Copied from our internal work repository at the version that rendered the paper's camera data; for this
release only comments, docstrings, messages and the default paths of `spec.py` were changed.

| file | what it does |
|---|---|
| `cameras/transform.py` | the LIBERO-Plus fork's five-step agentview camera transform (vertical, horizontal, scale, roll, end tilt), reimplemented for real-valued poses; integer poses give the fork's exact values |
| `cameras/spec.py` | re-derives the 451 test camera poses from the LIBERO-Plus case list and the fork, refuses on any mismatch, and writes `testset.json` |
| `cameras/testset.json` | the 451 test poses, as the benchmark's case names write them, and each task's scene class |
| `cameras/sample.py` | the training camera draws (one family per view, seeded by episode and frame) and the 24-view validation and test ladder; a draw that sets a test camera is drawn again |
| `cameras/test_cameras.py` | checks `transform.camera` against the fork's own `_setup_camera` code, the re-derivation, and the sampler's shares, uniformity, seeding and redraws |
| `pairs/driver.py` | renders each demonstration state at the nominal camera and at its drawn views, checks that the wrist image and the state are bit-identical, and writes one npz and receipt per job |

## Requirements

- Python with numpy and scipy (`transform.py`, `sample.py`); h5py and MuJoCo 3.2.3 for the driver; pytest and the
  same for the tests. Run from this folder so that `cameras` and `pairs` import (`PYTHONPATH=.`).
- Your own checkout of the LIBERO-Plus simulator fork at commit `4976dc30028e805ff8094b55501d532c48fec182` and its
  assets, and the LIBERO demonstrations, from their sources. We do not redistribute them.
- `spec.py` takes the fork with `--fork <checkout>` (default: `$LIBERO_PLUS_FORK`, else `./LIBERO-plus`) and the
  case list with `--cases <task_classification.json>` (default: the fork's own
  `libero/libero/benchmark/task_classification.json`, sha256 `faa87cce...a999e6`). `test_cameras.py` reads the fork
  from `$LIBERO_PLUS_FORK` the same way.
- `pairs/driver.py` imports a `render` module (scene construction, state restore and frame reading on the fork) that
  is not part of this release.

## Example

```
cd camera-render
python3 -c "from cameras import sample; print(sample.train_view('libero_spatial/task:demo_0', 0, 'libero_tabletop_manipulation'))"
```
