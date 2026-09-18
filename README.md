### Project title: Kahikatea Tree Classification Using Deep Learning



• Student name and ID: Niger Sultana and 1135036

• Brief project overview: 

Accurate mapping of native vegetation is essential for biodiversity conservation,
ecosystem restoration, and sustainable land management in New Zealand.
Traditionally, identifying tree species from aerial photographs requires significant time
and expert knowledge. Recent advances in deep learning and computer vision have
made it possible to automatically identify vegetation types from high-resolution aerial
imagery, supporting conservation agencies in monitoring native forests over large
geographical areas.
In this project, it was investigate the Kahikatea Image Dataset, which was
developed as part of the TAIAO Environmental Data Science (Jia, 2021). The dataset
contains aerial images extracted from orthophotographs captured over New Zealand
landscapes. Each image has been annotated to indicate whether it contains Kahikatea
(Dacrycarpus dacrydioides), one of New Zealand's tallest native tree species, which
commonly grows in lowland swamp forests.
Primary objective is to develop and evaluate deep learning
models with different configurations and architectures for binary image classification to
determine whether a given aerial image contains Kahikatea.
You are expected to investigate the dataset, perform data preparation and exploratory
analysis, design and evaluate deep learning models, and critically assess the strengths
and limitations of your approach.

• Dataset description:
Each image in the Kahikatea dataset represents a small geographic area and has been
labelled based on the presence or absence of Kahikatea trees.


• Instructions to download the dataset:
You can download the dataset with this link:
https://datasets.cms.waikato.ac.nz/taiao/kahikatea/

• Installation instructions:

This project uses Python 3.12 and the following packages (also listed in requirements.txt):

| Package | Used for |
|---|---|
| `jupyter`, `ipykernel` | running the `.ipynb` notebook |
| `numpy`, `pandas` | data handling and tabular analysis |
| `pillow` | image loading, resizing, letterbox preprocessing |
| `opencv-python-headless` | brightness/sharpness (Laplacian variance) metrics |
| `imagehash` | perceptual-hash duplicate detection (Task 1) |
| `matplotlib`, `seaborn` | all charts and figures |
| `scikit-learn` | `StratifiedGroupKFold` split, evaluation metrics |
| `tensorflow-cpu` | model building, training, and evaluation (Task 3) |

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd <repo-folder>
```

**2. Create and activate a virtual environment** (recommended, keeps this
project's packages isolated from the rest of system)
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

**3. Install the required packages**
```bash
pip install -r requirements.txt
```

> **GPU users:** `tensorflow-cpu` runs everything on CPU only. If you have
> a compatible NVIDIA GPU and CUDA/cuDNN set up, replace `tensorflow-cpu`
> with `tensorflow` in `requirements.txt` for significantly faster training
> in Task 3, particularly the hyperparameter sweeps and fine-tuning
> comparisons.

**4. Add the dataset**

Place the Kahikatea image folders under `data/` at the project root:
```
project/
├── data/
│   ├── ve_positive/
│   └── ve_negative/
├── src/
│   └── Kahikatea_Combined_Task1-3.ipynb
├── figures/
└── models/
```
The notebook expects this exact layout: `DATA_ROOT = "../data"` in the
setup cell assumes the notebook lives in `src/` and `data/` is a sibling
folder one level up. If your layout differs, update `DATA_ROOT` (and the
`FIG_DIR`/`MODEL_DIR` paths in each Part) accordingly.

---

### Instructions for running the notebook

**1. Launch Jupyter**
```bash
jupyter notebook
# or, if using JupyterLab:
jupyter lab
```
Then open `src/Kahikatea_Combined_Task1-3.ipynb` in the browser tab that
opens (or open the same file directly in VS Code with the Jupyter
extension installed, if you prefer that instead).

**2. Select the kernel**

Make sure the notebook's kernel points at the virtual environment created
above (in VS Code: click the kernel picker in the top-right of the
notebook; in Jupyter: **Kernel → Change Kernel**).

**3. Run all cells, top to bottom**

Use **Restart Kernel and Run All** rather than running cells individually
or out of order, later cells depend on variables defined earlier in the
same session (e.g. `df`, `df_train`/`df_val`/`df_test`, `class_weight`),
and running cells out of sequence is the most common source of
`NameError`s in this notebook.

**4. Expect these run times** (CPU, approximate):
| Section | Time |
|---|---|
| Part 1 — full image scan (Task 1) | 1–2 minutes |
| Part 2 — preprocessing pipeline (Task 2) | a few seconds |
| Part 3 — Custom CNN hyperparameter sweep | ~12–15 minutes |
| Part 3 — VGG16 / EfficientNetB0 bottleneck extraction | ~2–3 minutes each |
| Part 3 — frozen-head sweeps (cached features) | seconds |
| Part 3 — fine-tuning comparison runs | ~2–3 minutes each |
| Part 3 — final model training (all three) | a few minutes total |

**5. Outputs**

Running the notebook creates/populates:
- `figures/Task1/`, `figures/Task2/`, `figures/Task3/` — all generated charts
- `models/` — the saved final model (`.keras` files)

Both are created automatically (`os.makedirs(..., exist_ok=True)`) — no
need to create them by hand first.

**6. Pretrained weights**

VGG16 and EfficientNetB0 download ImageNet-pretrained weights automatically
via `tf.keras.applications` on first run (requires an internet connection).
If `weights="imagenet"` fails in your environment for any reason, the VGG16
path falls back to a GitHub-mirrored copy automatically; there is currently
no fallback wired in for EfficientNetB0, so a failed download there will
raise an explicit error rather than silently using random weights.

• Summary of key findings:

**EfficientNetB0, with a frozen ImageNet-pretrained backbone and a small
trained classification head, is the recommended model.**

**Why, using multiple forms of evidence rather than accuracy alone:**

1. **It wins on every test-set metric measured**, not just one: accuracy
   (0.808 vs. VGG16's 0.779, Custom CNN's 0.442), precision (0.760 vs.
   0.725 vs. 0.442), recall (0.826 vs. 0.804 vs. 1.000*), F1 (0.792 vs.
   0.763 vs. 0.613), and AUC (0.900 vs. 0.879 vs. 0.691). *Custom CNN's
   recall of 1.000 is not a genuine strength — see point 4.
2. **It achieves this with 3.6× fewer parameters than VGG16** (4.13M vs.
   14.75M total) — this is not a performance-for-complexity trade-off
   where a marker has to weigh competing priorities; EfficientNetB0 is
   simultaneously smaller and better, which is the strongest possible
   form of evidence for a recommendation.
3. **The RQ2 hyperparameter sweep independently supported this before the
   test set was even touched**: EfficientNetB0's frozen-head sweep (§4.2)
   reached validation AUC 0.90–0.96 across all five configurations tested,
   compared to VGG16's 0.90–0.92 band (§3.2) — the same ranking the test
   set later confirmed.
4. **The RQ3 fine-tuning evidence supports the "frozen, not fine-tuned"
   design choice for both transfer models, reinforcing the reliability of
   the frozen-backbone approach generally**: unfreezing the last block hurt
   both VGG16 (AUC 0.927→0.595) and EfficientNetB0 (AUC 0.960→0.428) at
   this dataset size — EfficientNetB0's collapse was the more severe of the
   two, which is worth monitoring if this dataset grows and fine-tuning is
   revisited, but does not change the recommendation for the frozen
   configuration evaluated here.
5. **The custom CNN is not recommended, and its result requires a specific
   caveat, not just a "weakest model" ranking.** Its 100% recall and
   AUC 0.691 could easily be misread as "high-recall, acceptable" if the
   confusion matrix weren't inspected directly — it in fact predicted
   *every single test image as positive*, a degenerate outcome rather than
   a genuine ecological-conservatism trade-off. This is a caution for
   interpreting any single small-validation-set threshold search: recall
   alone, or F1 alone, can look reasonable while masking a collapsed
   classifier.
