# Experiments & Benchmarking Guide

This guide documents every experiment suite in the Darshan framework: what it does, how to run it, what it outputs, and how to interpret the results.

---

## 1. Overview

Darshan provides 7 experiment types, accessible via the `/test` command or by running scripts directly:

| Suite | Interactive Command | Direct Script | Purpose |
|:---|:---|:---|:---|
| Compare | `/test compare` | `python experiments/run_comparison.py` | Multi-seed model comparison across all 9 model variants |
| Sweep | `/test sweep` | (uses `run_comparison` in loop) | Compare across multiple datasets in one run |
| Scaling | `/test scaling` | `python experiments/run_scaling.py` | Accuracy vs training sample size |
| Noise | `/test noise` | `python experiments/run_noise.py` | Accuracy vs depolarizing noise probability |
| Ablation | `/test ablation` | `python experiments/run_ablation.py` | Component removal study |
| Stats | `/test stats` | `python experiments/stats_engine.py` | Statistical report generation |
| Quantum Advantage | `/test quantum_advantage` | `python experiments/run_quantum_advantage.py` | Theoretical Hilbert space scaling |

Additionally, two standalone Matplotlib-based scripts exist:
- `experiments/n_scaling.py` — Learning curve plots with N=200 hypothesis line
- `experiments/noise_study.py` — Noise degradation plots with classical baseline

---

## 2. Comparison Experiment (`/test compare`)

### Purpose
Benchmark all 9 model variants head-to-head on a single dataset across multiple random seeds.

### Models Evaluated

| Model | Class | Source |
|:---|:---|:---|
| Parampara Legacy | `ParamparaLegacy` | `models/parampara_svm.py` |
| Parampara Pro | `ParamparaPro(mode='fair')` | `models/parampara_pro.py` |
| Parampara Pro+ | `ParamparaPro(mode='industry')` | `models/parampara_pro.py` |
| Ananta Legacy | `AnantaVQC` | `models/ananta_vqc.py` |
| Ananta Pro | `AnantaPro` | `models/ananta_pro.py` |
| Samyoga Legacy | `SamyogaLegacySVM` | `models/samyoga_svm.py` |
| Samyoga Pro | `SamyogaPro` | `models/samyoga_pro.py` |
| Samyoga Go | `SamyogaGo` | `models/samyoga_go.py` |
| Samyoga Shadow | `SamyogaShadow` (only in Fair Mode) | `models/samyoga_shadow.py` |

### How to Run

```
# From the interactive CLI:
 ❯ /dataset wine
 ❯ /test compare

# Direct script execution:
python experiments/run_comparison.py
```

### Interactive Prompts

1. **Execution Profile:** Selects epochs, sample cap, and qubit count (see profiles table below)
2. **Fair Mode:** Whether to include Samyoga Shadow for parameter-fair quantum evaluation
3. **Retrain Selection:** If previous results exist, select which models to retrain

### Execution Profiles

| Profile | Epochs | Max Samples | Qubits | Binary |
|:---|:---|:---|:---|:---|
| `smoke` | 1 | 30 | 4 | Yes |
| `demo` / `fast` | 3 | 100 | 4 | No |
| `low_data` | 10 | 200 | 8 | No |
| `quantum_safe` | 5 | 200 | 4 | Yes |
| `research` | 10 | 200 | 8 | No |
| `large` | (user) | 1000 | 8 | No |
| `full` | (user) | unlimited | 8 | No |

### Seeds

Default seeds: `[42, 43, 44]`. Each model is trained and evaluated independently per seed.

### Output Files

| File | Path | Contents |
|:---|:---|:---|
| Persistent comparison | `results/metrics/model_comparison.csv` | All results, deduped by (Dataset, Seed, Model) |
| Latest leaderboard | `results/{dataset}_benchmark_latest.csv` | Per-run leaderboard with Efficiency column |
| Session snapshot | `results/metrics/sessions/compare_{YYYYMMDD}_{HHMMSS}_{duration}s.csv` | Timestamped copy of this run's results |
| Comparison charts | `results/figures/comparison_{metric}_{DATASET}.png` | Bar charts for each metric |
| Accuracy vs time | `results/figures/accuracy_vs_time_{DATASET}.png` | Scatter plot |

### Output Columns

`Dataset, Seed, Model, Accuracy, F1_Macro, AUC_ROC, Brier_Score, Log_Loss, Train_Time_s, Predict_Time_s`

### Console Output

1. **Seed-wise summary table** — Ranked results per seed
2. **Combined averages table** — Mean across seeds, sorted by accuracy
3. **Championship standings** — Points-based ranking (25/18/15/12/10/8/6/4/2/1)
4. **Winner podium** — Top-3 display
5. **AI interpretation** — Automated analysis text
6. **Categorical comparisons** — Legacy vs Pro vs Advanced breakdowns
7. **Terminal bar charts** — Rich-rendered metric bars
8. **Full leaderboard table** — Complete results

### Interpreting Results

- **Accuracy/F1/AUC** — Higher is better. Compare Fair-Track models (same dimensions) against each other.
- **Efficiency** = Accuracy / Train_Time. High efficiency means the model converges quickly.
- **Parampara Pro vs Quantum models** — The core comparison. If Parampara Pro matches or beats quantum models under Fair-Track, quantum advantage is not demonstrated.
- **SamyogaPro vs SamyogaGo** — If Go (classical mock) matches Pro (quantum circuits), the quantum overhead provides no benefit.
- **SamyogaPro vs SamyogaShadow** — If Shadow (parameter-matched MLP) matches Pro, quantum structure provides no advantage beyond parameter count.

---

## 3. Sweep Experiment (`/test sweep`)

### Purpose
Run the comparison experiment across multiple datasets sequentially.

### How to Run

```
 ❯ /test sweep
# Select datasets via checkbox prompt
```

### Behavior
- Presents a checkbox list of all datasets (large datasets unchecked by default)
- Runs `run_comparison()` with `datasets=[selected list]`
- Results are saved to `results/all_datasets_benchmark_latest.csv`
- Charts include `SWEEP (N DATASETS)` in the title

---

## 4. Scaling Experiment (`/test scaling`)

### Purpose
Test how model accuracy changes as training sample size increases. This directly investigates the "200-sample crossover" hypothesis: whether quantum models excel in low-data regimes.

### Sample Sizes
`[10, 20, 50, 100, 200]` — sizes larger than the dataset are skipped.

### How to Run

```
 ❯ /dataset breast_cancer
 ❯ /test scaling
```

### Output Files

| File | Path |
|:---|:---|
| CSV results | `results/metrics/scaling_analysis.csv` |
| Scaling curves | `results/figures/n_scaling_{dataset}.png` (from `n_scaling.py`) |

### Output Columns
`Dataset, Model, Train_Size, Accuracy, AUC_ROC`

### Interpreting Results
- Look for the **crossover point** where quantum/hybrid models surpass classical baselines
- Models that improve faster with fewer samples have superior **sample efficiency**
- The N=200 hypothesis line in standalone plots marks the theorized quantum advantage threshold

---

## 5. Noise Experiment (`/test noise`)

### Purpose
Evaluate model robustness under simulated quantum hardware noise. Depolarizing noise is injected into quantum circuits via PennyLane's `DepolarizingChannel(p)`.

### Noise Levels
`[0.0, 0.01, 0.05, 0.1, 0.2]` — probability that each qubit gate is followed by a depolarizing channel.

### How to Run

```
 ❯ /dataset iris
 ❯ /test noise
```

### Which Models Are Affected

| Model | Noise Injected? | Mechanism |
|:---|:---|:---|
| Parampara Legacy/Pro/Pro+ | No | Classical models are noise-invariant |
| Ananta VQC | Yes | `DepolarizingChannel(noise_prob)` per qubit |
| Ananta Pro | Yes | `noise_aware=True, noise_prob=p` in quantum feature extractor |
| Samyoga Legacy | Yes | `noise_prob=p` in VQC pre-training circuit |
| Samyoga Pro/Go | No | Not parameterized for noise injection |
| Samyoga Shadow | No | Classical model |

### Output Files

| File | Path |
|:---|:---|
| CSV results | `results/metrics/noise_analysis.csv` |
| Noise curves | `results/figures/noise_study_{dataset}.png` (from `noise_study.py`) |

### Output Columns
`Dataset, Model, Noise, Accuracy, AUC_ROC`

### Interpreting Results
- Classical models should show flat accuracy across noise levels (noise-invariant)
- Pure VQCs (Ananta) typically degrade significantly at $p ≥ 0.05$
- Hybrid models (Samyoga) may show more graceful degradation due to classical head compensation
- The **decoherence threshold** is the noise level where quantum models drop below the classical baseline

---

## 6. Ablation Experiment (`/test ablation`)

### Purpose
Systematically remove components from hybrid models to measure each component's contribution.

### Ablation Variants

**Samyoga Legacy variants:**

| Variant | `use_interactions` | `feature_selection` |
|:---|:---|:---|
| Full | True | True |
| No Interactions | False | True |
| No Selection | True | False |

**Ananta Pro variants:**

| Variant | `ablation_mode` | `use_interactions` | What It Tests |
|:---|:---|:---|:---|
| Full Hybrid | None | True | Complete pipeline |
| Classical Only | `classical_only` | — | Only PCA features, no quantum |
| Quantum Only | `quantum_only` | — | Only quantum features, no classical |
| Shuffled Quantum | `shuffled_quantum` | — | Quantum features randomly shuffled (destroys signal) |
| No Interactions | None | False | Hybrid without polynomial interactions |

### How to Run

```
 ❯ /dataset iris
 ❯ /test ablation
```

### Output Files

| File | Path |
|:---|:---|
| Full results | `results/metrics/ablation_analysis.csv` |
| Ananta Pro subset | `results/metrics/ananta_pro_ablation.csv` |

### Output Columns
`Dataset, Model, Accuracy, AUC_ROC`

### Interpreting Results
- **Classical Only vs Full Hybrid:** If full hybrid barely exceeds classical only, quantum features add little value
- **Quantum Only vs Full Hybrid:** If quantum only is much worse, classical features are essential for the pipeline
- **Shuffled Quantum vs Full Hybrid:** Shuffling destroys quantum signal. If shuffled matches full, quantum features are not meaningful
- **No Interactions:** Measures the value of polynomial feature interactions in the pipeline

---

## 7. Statistics Report (`/test stats`)

### Purpose
Generate aggregate statistical summaries and LaTeX tables from existing result CSVs.

### How to Run

```
 ❯ /test stats
```

### Implementation
Located in `experiments/stats_engine.py`:
1. Reads `results/metrics/model_comparison.csv`
2. Groups by (Dataset, Model) and computes:
   - `Accuracy_Mean`, `Accuracy_Std`
   - `AUC_ROC_Mean`, `AUC_ROC_Std`
3. Exports LaTeX table to `*_table.tex`
4. Prints formatted summary to console

### Output Files
- `results/metrics/model_comparison_table.tex` — LaTeX summary table
- (Also processes any other CSVs in `results/metrics/` when run as standalone script)

> **Note:** The stats engine imports `wilcoxon` and `ttest_rel` from scipy.stats but the current `generate_statistics_report()` function only performs mean/std aggregation. Formal hypothesis testing (p-values, effect sizes) would require extending this function.

---

## 8. Quantum Advantage Analysis (`/test quantum_advantage`)

### Purpose
Display a theoretical scaling table showing how Hilbert space dimensionality ($2^N$) grows exponentially compared to classical feature dimensionality ($N$).

### How to Run

```
 ❯ /test quantum_advantage
```

### Output
Console table and terminal curve showing:
- Qubits: [2, 4, 6, 8, 10, 12, 16]
- Classical Features: N
- Hilbert Space Dim: $2^N$
- Advantage Multiplier: $2^N / N$
- Classical Simulation Complexity: $N^2 \log_2(2^N + 1)$

This is a theoretical demonstration, not an empirical benchmark.

---

## 9. Standalone Scripts

### `experiments/n_scaling.py`

Runs a complete N-scaling experiment with Matplotlib plot output.

```powershell
python experiments/n_scaling.py
```

- **Function:** `run_n_scaling(dataset_name, train_sizes, epochs_vqc)`
- **Default:** breast_cancer, sizes [20, 50, 100, 200, 400], 50 epochs
- **Output:** `results/metrics/n_scaling_{dataset}.csv`, `results/figures/n_scaling_{dataset}.png`
- **Special:** Draws a red dashed line at N=200 (the hypothesis line)

### `experiments/noise_study.py`

Runs a noise degradation study with Matplotlib plot output.

```powershell
python experiments/noise_study.py
```

- **Function:** `run_noise_study(dataset_name, noise_levels, epochs_vqc)`
- **Default:** iris, levels [0.0, 0.01, 0.05, 0.1], 50 epochs
- **Output:** `results/metrics/noise_study_{dataset}.csv`, `results/figures/noise_study_{dataset}.png`
- **Special:** Draws a black dashed horizontal line at the classical SVM noiseless baseline accuracy

---

## 10. Incremental Results System

All experiment scripts implement an incremental training system:

1. On startup, check if the target CSV already exists
2. If it does, load existing results
3. Present a checkbox prompt listing already-trained models grouped by family (Parampara, Ananta, Samyoga)
4. Only retrain models explicitly selected for retraining
5. Merge new results with existing results, deduplicating by key columns
6. Save the merged dataframe back to the CSV

**To force a clean run:** Delete the relevant CSV in `results/metrics/` before running.

---

## 11. Runtime Considerations

| Model | Approximate Time per Seed (Wine, 4 qubits) |
|:---|:---|
| Parampara Legacy | < 1 second |
| Parampara Pro / Pro+ | 3–5 seconds |
| Ananta VQC (10 epochs) | 30–70 seconds |
| Ananta Pro | 4–400 seconds (depends on quantum seeds and dataset size) |
| Samyoga Legacy (10 epochs) | 60–170 seconds |
| Samyoga Go | < 1 second |
| Samyoga Shadow | < 1 second |
| Samyoga Pro | 80–1600 seconds (extremely variable; uses PennyLane quantum circuits) |

**Total comparison run (Wine, 3 seeds, all models):** 30 minutes to 3+ hours depending on mode and hardware.

**Recommendation for rapid iteration:** Use `quantum_safe` or `fast` mode, or skip SamyogaPro by pressing Ctrl+C when prompted.
