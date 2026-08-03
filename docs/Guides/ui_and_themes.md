# UI Styling & Visualization Guide

This guide documents the Darshan framework's visual system: the 18 color themes, model-specific color mapping, chart generation pipeline, terminal visualizations, and the Rich-based UI component library.

---

## 1. Theme System

### Architecture

The theme system is implemented in `ui/theme.py` and consists of:
- `THEMES` dict: 18 theme definitions, each with 9 color keys
- `STATE` dict: Global application state including the active theme
- `get_color(key)`: Resolves a semantic color key against the active theme
- `get_model_color(model_name)`: Returns a hex color specific to a model family
- `gradient_text(text, start, end)`: Renders character-by-character gradient text

### Available Themes (18)

| Theme | Primary | Secondary | Accent | Aesthetic |
|:---|:---|:---|:---|:---|
| `neon` | #FF107A (Hot Pink) | #00E5FF (Cyan) | #7000FF (Purple) | High-energy nightclub |
| `cyber` | #E6FF00 (Electric Yellow) | #FF00E6 (Magenta) | #00FFFF (Cyan) | Cyberpunk dystopia |
| `matrix` | #00FF41 (Bright Green) | #008F11 (Dark Green) | #003B00 (Forest) | Phosphor console |
| `aurora` | #00FA9A (Spring Green) | #9370DB (Medium Purple) | #87CEFA (Light Sky Blue) | Northern lights |
| `royal` | #FFD700 (Gold) | #4B0082 (Indigo) | #8A2BE2 (Blue Violet) | Regal luxury |
| `ocean` | #0077BE (Ocean Blue) | #00A86B (Jade) | #20B2AA (Light Sea Green) | Deep sea |
| `forest` | #8FBC8F (Dark Sea Green) | #556B2F (Dark Olive) | #DEB887 (Burlywood) | Forest canopy |
| `ember` | #FF4500 (Orange Red) | #FF8C00 (Dark Orange) | #FFD700 (Gold) | Thermal energy |
| `glacier` | #F0F8FF (Alice Blue) | #ADD8E6 (Light Blue) | #B0E0E6 (Powder Blue) | Ice and frost |
| `monochrome` | #FFFFFF (White) | #A9A9A9 (Dark Gray) | #D3D3D3 (Light Gray) | Clean grayscale |
| `amethyst` | #DA70D6 (Orchid) | #800080 (Purple) | #DDA0DD (Plum) | Crystalline purple |
| `crimson` | #DC143C (Crimson) | #800000 (Maroon) | #B22222 (Firebrick) | Blood red |
| `quantum` | #C0C0C0 (Silver) | #0055FF (Royal Blue) | #00FFFF (Cyan) | Lab instrument |
| `paper` | #F5F5DC (Beige) | #8B4513 (Saddle Brown) | #CD853F (Peru) | Aged parchment |
| `solar` *(default)* | #FFDD00 (Gold) | #FF7700 (Orange) | #FF0066 (Rose) | Solar energy |
| `gemini` | #FF33A1 (Deep Pink) | #C27CFF (Lavender) | #33D1FF (Sky Blue) | Gemini AI |
| `accessible` | #FFFF00 (Yellow) | #00FFFF (Cyan) | #FF00FF (Magenta) | High-contrast accessibility |
| `vibgyor` | #FF007F (Rose) | #00E5FF (Cyan) | #FFEA00 (Yellow) | Rainbow spectrum |

### Color Keys

Each theme defines 9 semantic color keys:

| Key | Usage |
|:---|:---|
| `primary` | Main UI elements, prompts, borders |
| `secondary` | Supporting text, column headers |
| `accent` | Highlighted items, special elements |
| `success` | Positive outcomes, accuracy indicators |
| `warning` | Caution messages, moderate values |
| `error` | Error messages, failure indicators |
| `muted` | Dimmed/background text |
| `border` | Panel and table borders |
| `header` | Table headers, section titles |

### Switching Themes

```
# Direct command
 ❯ /theme matrix

# Interactive selector (grid display)
 ❯ /theme
```

The theme selector (`ui/theme_selector.py`) displays a visual grid of all themes with sample text in each theme's primary color.

---

## 2. Model Color Mapping

Each model family has a distinct color assignment for consistent chart rendering:

| Model | Color Source | Brightness Modifier |
|:---|:---|:---|
| Ananta Legacy / Ananta | `#8B4513` (Saddle Brown) | None |
| Parampara Legacy | `#D2B48C` (Tan) | None |
| Samyoga Legacy | `#CD7F32` (Bronze) | None |
| Ananta Pro+ | Theme `accent` | 1.4× brightness |
| Parampara Pro/Pro+ | Theme `secondary` | 1.4× (Pro+), normal (Pro) |
| Samyoga Pro/Go | Theme `primary` | 1.8× (Go), normal (Pro) |

This ensures consistent color coding across charts, bars, and terminal displays.

---

## 3. Chart Generation Pipeline

### Matplotlib/Seaborn Charts

All Matplotlib charts use a dark background theme applied by `apply_neon_theme()`:

```python
plt.style.use("dark_background")
sns.set_theme(
    style="darkgrid",
    rc={"axes.facecolor": "#121212", "figure.facecolor": "#121212", "grid.color": "#2a2a2a", "text.color": "white"},
)
```

Charts are saved at **300 DPI** to `results/figures/`:

```python
fig.savefig(path, dpi=300, bbox_inches="tight")
```

### Chart Types

| Chart | Function | Triggered By | Output File |
|:---|:---|:---|:---|
| Metric comparison bars | `save_all_comparison_charts()` | `/test compare` | `comparison_{metric}_{dataset}.png` |
| Accuracy vs time scatter | `save_accuracy_vs_time_scatter()` | `/test compare` | `accuracy_vs_time_{dataset}.png` |
| Learning curves | `save_learning_curve()` | `/train continue` | `learning_curve.png` |
| N-scaling curves | (in `n_scaling.py`) | `python experiments/n_scaling.py` | `n_scaling_{dataset}.png` |
| Noise degradation | (in `noise_study.py`) | `python experiments/noise_study.py` | `noise_study_{dataset}.png` |
| Confusion matrix | `save_confusion_matrix()` | `/model [name]` | `confusion_matrix_{model}.png` |
| Scaling curves | `save_scaling_curve()` | `/test scaling` | `scaling_curve_{dataset}.png` |
| Noise curves | `save_noise_curve()` | `/test noise` | `noise_curve_{dataset}.png` |

### Bar Chart Features

- Horizontal bar charts with model-specific gradient coloring
- Each bar uses a two-color gradient (start → end) based on the model family
- Model names are left-aligned with value labels on bars
- Title includes dataset name and metric type

---

## 4. Terminal Visualizations

### Rich Metric Bars (`render_rich_metric_bars`)

Renders horizontal bar charts directly in the terminal using Rich `Table`:

```
Accuracy Comparison
━━━━━━━━━━━━━━━━━━━━━
Parampara Pro    ████████████████████████ 94.4%
Samyoga Go       ██████████████████████   93.3%
Ananta Pro       ██████████████████       90.0%
```

Each bar is colored using the model's assigned color and rendered with Unicode block characters.

### Plotext ASCII Curves (`render_terminal_curve`)

Renders line graphs in the terminal using the `plotext` library:

```python
import plotext as plt

plt.plot(x_values, y_values)
plt.title("Scaling Curve")
plt.show()
```

Used for quantum advantage scaling and other continuous data visualizations.

### Winner Podium (`print_winner_podium`)

Displays top-3 models with trophy styling:

```
🥇 1st Place: Parampara Legacy    Accuracy: 100.0%
🥈 2nd Place: Samyoga Shadow      Accuracy: 96.7%
🥉 3rd Place: Samyoga Pro         Accuracy: 96.7%
```

### AI Interpretation (`print_interpretation`)

Generates an automated textual analysis of comparison results:
- Identifies the winner and runner-up
- Calculates accuracy gap between quantum and classical models
- Comments on efficiency (accuracy/time ratio)
- Notes any quantum advantage or disadvantage

---

## 5. UI Components (`ui/components.py`)

### Core Functions

| Function | Description |
|:---|:---|
| `print_panel(text, title, style_key)` | Renders a styled Rich panel with borders |
| `print_error_panel(issue, cause, fix)` | Renders a structured error message panel |
| `print_dashboard()` | Renders the current state dashboard (dataset, epochs, theme, modes) |
| `df_to_table(df, title)` | Converts a pandas DataFrame to a Rich Table |
| `print_winner_podium(results)` | Renders the top-3 winners with trophies |
| `print_interpretation(results)` | Renders automated results interpretation |
| `render_rich_metric_bars(data, metric)` | Renders horizontal bar chart in terminal |

### Panel Styling

Panels use `rich.rule.Rule` for top/bottom borders and `rich.padding.Padding` for content indentation. Border colors are derived from the active theme.

### Table Styling

Tables use `box.SIMPLE` style with headers colored by the theme's `header` key. Model names in table cells are colored using `get_model_color()`.

---

## 6. Model ASCII Logos

Each model has an ASCII art logo defined in `RAW_LOGOS` (in `ui/theme.py`):

```
 ___   _   __  ____   _____   ___   _
/ __| /_\ |  \/  \ \ / / _ \ / __| /_\
\__ \/ _ \| |\/| |\ V / (_) | (_ |/ _ \  PRO
|___/_/ \_\_|  |_| | | \___/ \___/_/ \_\

Enterprise-Grade Quantum Transfer Learning
```

Logos are gradient-colored per-character when rendered, with Legacy models using bronze tones and Pro models using theme colors.

### Available Logo Keys

- `parampara_legacy`, `parampara`, `parampara_pro_industry`
- `ananta`, `ananta_pro`
- `samyoga_legacy`, `samyoga_pro`, `samyoga_go`, `samyoga_shadow`

---

## 7. Animated Panels (`animate_panel`)

The boot sequence and model loading screens use character-by-character animation:

```python
animate_panel(
    text="Initializing quantum subsystem...",
    title="SYSTEM STATUS",
    border_style="#00FF41",
    delay=0.005,  # 5ms per character
)
```

Uses `rich.live.Live` for flicker-free terminal updates at 60 FPS.

---

## 8. Gradient Text

The `gradient_text()` function renders text with per-character color interpolation:

```python
gradient_text("DARSHAN", start_hex="#FF107A", end_hex="#00E5FF")
```

Each character gets a color computed by linear interpolation between the start and end RGB values based on its position in the string.

---

## 9. Prompt Toolkit Integration

The CLI prompt uses `prompt_toolkit` for:
- **Gradient prompt bar:** Shows current state (dataset, epochs, theme, quiet mode)
- **Tab completion:** Nested command completion via `NestedCompleter`
- **Auto-suggest:** History-based suggestions via `AutoSuggestFromHistory`
- **Key bindings:** Standard readline bindings (Ctrl+C, Ctrl+D, arrow keys)

### Questionary Integration

Interactive selection menus use `questionary` with theme-aware styling:

```python
from ui.theme import get_q_style

answer = questionary.select(
    "Choose a dataset:",
    choices=dataset_list,
    style=get_q_style(),  # Matches current Darshan theme
).ask()
```
