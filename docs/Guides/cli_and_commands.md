# CLI Reference & Command Guide

This guide documents every interactive command available in the Darshan CLI, the state machine, tab-completion, and advanced features.

---

## 1. Launching the CLI

```powershell
# Recommended on Windows: sets UTF-8 encoding and activates .venv/venv if present
.\start_darshan.bat

# Direct launch
python darshan.py
```

### Boot Sequence

On launch, Darshan displays:
1. Gradient-colored ASCII logo
2. Animated "system status" panel (simulated quantum-themed boot messages)
3. System information display (Python version, PennyLane version, cores, OS)
4. Prompt with status bar showing: active dataset, epoch count, theme, quiet mode

---

## 2. State Management

The CLI maintains global state in `ui/theme.py:STATE`:

| Key | Type | Default | Description |
|:---|:---|:---|:---|
| `dataset` | str\|None | `None` | Currently loaded dataset name |
| `epochs` | int | `10` | Number of VQC training epochs |
| `theme` | str | `'solar'` | Active color theme |
| `working_mode` | bool | `False` | Whether to capture raw training logs |
| `quiet_mode` | bool | `False` | Suppress progress bars |
| `last_run` | str\|None | `None` | Description of the last experiment run |
| `last_winner` | str\|None | `None` | Name of the last comparison winner |
| `working_log` | str | `''` | Captured raw training output |
| `last_metrics_path` | str\|None | `None` | Path to the last saved metrics CSV |
| `last_figures_path` | str\|None | `None` | Path to the last saved figures directory |

---

## 3. Command Reference

### Dataset Commands

| Command | Description | Example |
|:---|:---|:---|
| `/dataset [name]` | Load a dataset by name. Displays Rich panel with stats (samples, features, classes, preprocessing). | `/dataset wine` |
| `/dataset` (no args) | Open interactive dataset selector via `questionary.select()` | `/dataset` |

Available dataset names: `moons`, `iris`, `wine`, `breast_cancer`, `complexity_wall`, `digits`, `pendigits`

### Model Commands

| Command | Description | Example |
|:---|:---|:---|
| `/model [name]` | Display model architecture logo, then run a benchmark with validation strategy selection | `/model samyoga_pro` |

Available model names: `parampara_legacy`, `parampara` (Pro Fair), `parampara_pro_industry`, `ananta` (VQC Legacy), `ananta_pro`, `samyoga_legacy`, `samyoga_pro`, `samyoga_go`, `samyoga_shadow`

After selecting a model, the CLI prompts for a validation strategy:
1. **Standard Split (80/20)** — Single train/test evaluation
2. **5-Fold Cross-Validation** — StratifiedKFold
3. **Monte Carlo (5 iterations)** — ShuffleSplit
4. **Leave-One-Out (LOOCV)** — Capped at 30 folds for efficiency

### Experiment Commands

| Command | Description |
|:---|:---|
| `/test compare` | Run multi-seed comparison (9 models × 3 seeds) on the loaded dataset |
| `/test sweep` | Run comparison across multiple user-selected datasets |
| `/test scaling` | Run sample-size scaling analysis |
| `/test noise` | Run depolarizing noise degradation study |
| `/test ablation` | Run component ablation study (Samyoga Legacy + Ananta Pro) |
| `/test stats` | Generate statistical report + LaTeX tables from existing CSVs |
| `/test quantum_advantage` | Display theoretical Hilbert space scaling table |

### Training Commands

| Command | Description | Example |
|:---|:---|:---|
| `/epochs [N]` | Set VQC training epoch count | `/epochs 20` |
| `/train continue [model] [N]` | Resume training on a previously trained model for N additional epochs. Only works for Ananta VQC and Samyoga Legacy. | `/train continue ananta 5` |
| `/checkpoint save [model]` | Save model weights to `.npz` file | `/checkpoint save ananta` |
| `/checkpoint load [model]` | Load model weights from `.npz` file | `/checkpoint load samyoga_legacy` |

### Results Commands

| Command | Description |
|:---|:---|
| `/results` | Interactive session browser: lists all result CSVs in `results/metrics/sessions/`, select one to view averaged results table, podium, and AI interpretation |
| `/history` | Display last 10 entries from `results/history.json` |
| `/report` | Auto-generate a markdown research report from experiment history |

### Visualization Commands

| Command | Description |
|:---|:---|
| `/graphs list` | List all saved PNG charts in `results/figures/` |
| `/graphs latest` | Regenerate the latest comparison bar charts |
| `/graphs summary` | Generate aggregate summary charts |

### Interface Commands

| Command | Description | Example |
|:---|:---|:---|
| `/theme [name]` | Change color theme (see UI guide for available themes) | `/theme cyberpunk` |
| `/theme` (no args) | Open interactive grid-based theme selector | `/theme` |
| `/menu` | Open the interactive command center via `questionary.select()` | `/menu` |
| `/working on` | Enable raw training log capture | `/working on` |
| `/working off` | Disable raw training log capture | `/working off` |
| `/working show` | Display the captured training log | `/working show` |
| `/working clear` | Clear the captured training log | `/working clear` |
| `/quiet on` | Enable minimal progress view | `/quiet on` |
| `/quiet off` | Disable minimal progress view | `/quiet off` |
| `/about` | Display research background panel | `/about` |
| `/clear` | Clear terminal screen | `/clear` |
| `/reset` | Factory reset: deletes all caches, metrics, graphs, and history | `/reset` |
| `/restart` | Reset state and replay boot sequence | `/restart` |
| `/exit` or `/quit` | Cleanly terminate the session | `/exit` |

---

## 4. Tab Completion

The CLI uses `prompt_toolkit` to provide real-time tab completion:

### Completions Available

| Prefix | Completes To |
|:---|:---|
| `/d` | `/dataset` |
| `/m` | `/model`, `/menu` |
| `/t` | `/test`, `/theme`, `/train` |
| `/te` | `/test` |
| `/th` | `/theme` |
| `/r` | `/results`, `/report`, `/restart` |
| `/h` | `/history` |
| `/g` | `/graphs` |
| `/c` | `/clear`, `/checkpoint` |
| `/e` | `/epochs`, `/exit` |
| `/w` | `/working` |
| `/q` | `/quiet`, `/quit` |
| `/a` | `/about` |

### Nested Completion

Some commands provide second-level completion:
- `/model ` → lists all model names
- `/dataset ` → lists all dataset names
- `/test ` → lists all test suites
- `/theme ` → lists all theme names
- `/train continue ` → lists compatible model names

---

## 5. Prompt Design

The CLI prompt shows a colored status bar:

```
 WINE │ 30ep │ solar │ quiet: off 
 ❯ 
```

The status bar is gradient-colored using the active theme's primary and secondary colors.

---

## 6. History System

Every experiment run appends an entry to `results/history.json`:

```json
{
    "timestamp": "2026-07-26 01:10:37",
    "type": "COMPARE_ALL",
    "dataset": "wine",
    "model": "Parampara Legacy",
    "accuracy": 0.9722,
    "train_time": 0.35,
    "notes": "Seed 42 | Accuracy: 0.9722 | Time: 0.35s"
}
```

Access history via `/history` (last 10 entries) or view the raw JSON file.

---

## 7. Error Handling

The CLI provides structured error panels via `print_error_panel()`:

```
┌── SYSTEM ALERT ──┐
│ Issue:  No dataset loaded          │
│ Cause:  You haven't selected a dataset yet │
│ Fix:    Use /dataset wine          │
│ Suggested: /dataset                │
└──────────────────┘
```

### Common Error Triggers

| Error | Cause | Recovery |
|:---|:---|:---|
| "No dataset loaded" | Running `/test` or `/model` without `/dataset` first | Run `/dataset [name]` |
| "Unknown command" | Typo in command name | Check `/menu` for available commands |
| "Model not found" | Invalid model name | Check available models in the model table |
| Ctrl+C during training | User interruption | Model training is skipped; next model continues |

---

## 8. Working Mode

Working mode captures all raw stdout/stderr output during model training:

```
 ❯ /working on     # Start capturing
 ❯ /model ananta    # All print statements are captured
 ❯ /working show   # Display captured output
 ❯ /working clear  # Clear the buffer
 ❯ /working off    # Stop capturing
```

This is implemented via `utils/logger.py:WorkingLog` which redirects stdout/stderr to a `StringIO` buffer.

---

## 9. Keyboard Shortcuts

| Key | Action |
|:---|:---|
| `Tab` | Autocomplete current command |
| `Ctrl+C` | Cancel current operation / Skip current model |
| `Ctrl+D` | Exit the CLI |
| `Up/Down` | Navigate command history |
