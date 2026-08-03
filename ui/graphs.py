import os
import plotext as plt
import matplotlib.pyplot as plt_mpl
import seaborn as sns
import numpy as np
from rich.table import Table
from rich.console import Console
from rich import box
from ui.theme import STATE, get_color, model_color
console = Console()

def _get_hex_color(model_name):
    return model_color(model_name)

def _get_rich_color(model_name):
    return model_color(model_name)

def _get_plotext_color(model_name):
    from ui.theme import get_model_color, hex_to_rgb
    return hex_to_rgb(get_model_color(model_name))

def apply_neon_theme():
    plt_mpl.style.use('dark_background')
    sns.set_theme(style='darkgrid', rc={'axes.facecolor': '#121212', 'figure.facecolor': '#121212', 'grid.color': '#2a2a2a', 'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'})

def save_figure(fig, filename):
    os.makedirs('results/figures', exist_ok=True)
    path = os.path.join('results/figures', filename)
    try:
        fig.savefig(path, dpi=300, bbox_inches='tight')
    except Exception as e:
        console.print(f'[dim error]Failed to save {filename}: {e}[/]')
    finally:
        plt_mpl.close(fig)
    STATE['last_figures_path'] = 'results/figures'
    return path


def _get_bar_gradient(model_name):
    from ui.theme import get_color
    name = model_name.lower()
    c1 = get_color('primary').replace('bold ', '').replace('dim ', '')
    c2 = get_color('secondary').replace('bold ', '').replace('dim ', '')
    c3 = get_color('accent').replace('bold ', '').replace('dim ', '')
    c4 = get_color('success').replace('bold ', '').replace('dim ', '')
    c5 = get_color('warning').replace('bold ', '').replace('dim ', '')
    c6 = get_color('error').replace('bold ', '').replace('dim ', '')

    if 'ananta legacy' in name or name == 'ananta':
        return ('#8B4513', '#A0522D')
    elif 'parampara legacy' in name:
        return ('#D2B48C', '#DEB887')
    elif 'samyoga legacy' in name:
        return ('#CD7F32', '#8C5622')

    if 'go' in name or 'fast' in name or 'mock' in name:
        return (c1, c4)
    elif 'shadow' in name:
        return (c2, c5)
    elif 'samyoga' in name:
        return (c1, c2)

    elif 'pro+' in name or 'industry' in name:
        return (c5, c6)
    elif 'parampara' in name:
        return (c2, c3)

    elif 'ananta' in name:
        return (c3, c1)

    elif 'advantage' in name or 'multiplier' in name or 'quantum' in name:
        return ('#A2FF00', '#FFDD00')
    elif 'complexity' in name or 'sim' in name:
        return ('#CC0000', '#FFAA00')
    elif 'features' in name or 'classical' in name:
        return ('#0077FF', '#00E5FF')

    palette = [(c1, c2), (c2, c3), (c3, c4), (c4, c5), (c5, c6), (c6, c1)]
    return palette[abs(hash(name)) % len(palette)]

def render_rich_metric_bars(title, labels, values, metric_name='Value', higher_is_better=True, datasets=None, seeds=None, compact_legend=False, return_table_only=False):
    if not labels or not values:
        return
    from rich.table import Table
    from rich import box
    from ui.theme import get_color
    import shutil

    table = Table(show_header=True, header_style=get_color('header'), box=box.HEAVY_EDGE, border_style=get_color('primary'))

    if datasets is not None:
        table.add_column('Dataset', justify='left', style='dim')
    if seeds is not None:
        table.add_column('Seed', justify='center', style='dim')

    table.add_column('Model', justify='left', style='bold')
    table.add_column('Bar', justify='left')
    table.add_column(metric_name, justify='right')

    if metric_name in ['Accuracy', 'F1_Macro', 'AUC_ROC']:
        max_val = 1.0
    else:
        max_val = max([v for v in values if v is not None and (not np.isnan(v))] + [1e-09])

    from ui.theme import gradient_text

    term_width = shutil.get_terminal_size((80, 24)).columns
    reserved = 40
    if datasets is not None: reserved += 15
    if seeds is not None: reserved += 10
    bar_max_len = max(15, min(50, term_width - reserved))

    for idx, (label, val) in enumerate(zip(labels, values)):
        if val is None or np.isnan(val):
            continue
        c = _get_rich_color(label)
        filled = min(bar_max_len, max(0, int(val / max_val * bar_max_len)))
        empty = bar_max_len - filled

        start_hex, end_hex = _get_bar_gradient(label)
        filled_str = '█' * filled
        empty_str = '░' * empty

        grad_filled = gradient_text(filled_str, start_hex, end_hex) if filled > 0 else ""
        border_left = f"[{start_hex}]│[/]"
        border_right = f"[{end_hex}]│[/]"
        bar_str = f"{border_left}{grad_filled}[dim {end_hex}]{empty_str}[/]{border_right}"

        row_args = []
        if datasets is not None:
            ds_val = str(datasets[idx]).upper() if idx < len(datasets) else '—'
            row_args.append(ds_val)
        if seeds is not None:
            seed_val = str(seeds[idx]) if idx < len(seeds) else '—'
            row_args.append(seed_val)

        row_args.extend([f"{label:<18}", bar_str, f'[{c}]{val:.4f}[/{c}]'])
        table.add_row(*row_args)

    if return_table_only:
        return table

    from rich.panel import Panel
    from rich.columns import Columns

    def gt(name):
        sh, eh = _get_bar_gradient(name)
        return gradient_text("██████", sh, eh)

    if compact_legend:
        legend_panel = generate_horizontal_legend(labels)
    else:
        samyoga_present = any('samyoga' in l.lower() for l in labels)
        parampara_present = any('parampara' in l.lower() for l in labels)
        ananta_present = any('ananta' in l.lower() for l in labels)

        legend_text = ""
        if samyoga_present:
            legend_text += f"[bold]Samyoga Family[/bold]\n"
            if any('go' in l.lower() or 'fast' in l.lower() for l in labels): legend_text += f" Go/Fast:   {gt('samyoga go')}\n"
            if any('shadow' in l.lower() for l in labels): legend_text += f" Shadow:    {gt('samyoga shadow')}\n"
            if any('pro' in l.lower() and 'pro+' not in l.lower() for l in labels): legend_text += f" Pro:       {gt('samyoga pro')}\n"
            if any('legacy' in l.lower() for l in labels): legend_text += f" Legacy:    {gt('samyoga legacy')}\n"
            legend_text += "\n"
        if parampara_present:
            legend_text += f"[bold]Parampara Family[/bold]\n"
            if any('pro+' in l.lower() for l in labels): legend_text += f" Pro+:      {gt('parampara pro+')}\n"
            if any('pro' in l.lower() and 'pro+' not in l.lower() for l in labels): legend_text += f" Pro:       {gt('parampara pro')}\n"
            if any('legacy' in l.lower() for l in labels): legend_text += f" Legacy:    {gt('parampara legacy')}\n"
            legend_text += "\n"
        if ananta_present:
            legend_text += f"[bold]Ananta Family[/bold]\n"
            if any('pro' in l.lower() for l in labels): legend_text += f" Pro:       {gt('ananta pro')}\n"
            if any('legacy' in l.lower() for l in labels): legend_text += f" Legacy:    {gt('ananta legacy')}\n"
        legend_text = legend_text.strip()
        legend_panel = Panel(legend_text, title="Color Key", border_style="dim", box=box.ROUNDED)

    from rich.padding import Padding
    from rich.align import Align
    padded_legend = Padding(legend_panel, (0, 0, 0, 2))

    main_panel = Panel(Columns([table, padded_legend], align="center", expand=False), title=f"[bold white]{title}[/]", border_style=get_color('accent'), box=box.ROUNDED, padding=(1, 2))
    console.print(Align.center(main_panel))
    console.print()

def generate_horizontal_legend(labels):
    from rich.panel import Panel
    from ui.theme import gradient_text

    def gt(name):
        sh, eh = _get_bar_gradient(name)
        return gradient_text("██████", sh, eh)

    legend_parts = []
    for l in list(dict.fromkeys(labels)):
        legend_parts.append(f"[bold]{l}[/bold]: {gt(l)}")
    if len(legend_parts) > 3:
        rows = []
        for i in range(0, len(legend_parts), 3):
            rows.append("   ".join(legend_parts[i:i+3]))
        legend_text = "\n".join(rows)
    else:
        legend_text = "  ".join(legend_parts)
    return Panel(legend_text, title="Color Key", border_style="dim")

def render_terminal_curve(title, x_values, y_dict, x_title='X', y_title='Y'):
    import shutil
    import asciichartpy
    from rich.panel import Panel
    from rich.align import Align
    from rich.text import Text
    from rich.columns import Columns
    from rich import box
    from ui.theme import get_color
    import re

    term_w, term_h = shutil.get_terminal_size((80, 24))
    chart_width = max(40, term_w - 45)
    chart_height = min(20, term_h - 10)

    series_list = []
    colors_list = []
    labels = []

    for label, y_vals in y_dict.items():
        if not y_vals:
            continue
        labels.append(label)

        y_arr = np.array(y_vals, dtype=float)
        valid_mask = ~np.isnan(y_arr)

        if x_values and len(x_values) == len(y_vals) and isinstance(x_values[0], (int, float)):
            x_arr = np.array(x_values, dtype=float)
            if np.sum(valid_mask) > 1 and (x_arr[-1] != x_arr[0]):
                x_valid = x_arr[valid_mask]
                y_valid = y_arr[valid_mask]
                new_x = np.linspace(x_arr[0], x_arr[-1], chart_width)
                stretched_y = np.interp(new_x, x_valid, y_valid).tolist()
            else:
                stretched_y = [y_vals[0] if len(y_vals) > 0 else 0.0] * chart_width
        elif np.sum(valid_mask) > 1:
            y_valid = y_arr[valid_mask]
            old_x = np.linspace(0, 1, len(y_valid))
            new_x = np.linspace(0, 1, chart_width)
            stretched_y = np.interp(new_x, old_x, y_valid).tolist()
        else:
            stretched_y = [y_vals[0] if len(y_vals) > 0 else 0.0] * chart_width

        series_list.append(stretched_y)
        r, g, b = _get_plotext_color(label)
        ansi_color = f"\x1b[38;2;{r};{g};{b}m"
        colors_list.append(ansi_color)

    if not series_list:
        return

    all_y = [v for s in series_list for v in s if not np.isnan(v)]
    min_y = min(all_y) if all_y else 0
    max_y = max(all_y) if all_y else 1
    y_range = max_y - min_y

    if y_range < 0.15:
        fmt_str = "{:7.3f}"
    elif y_range < 0.015:
        fmt_str = "{:8.4f}"
    else:
        fmt_str = "{:6.2f}"

    cfg = {
        'height': chart_height,
        'colors': colors_list,
        'format': fmt_str
    }

    raw_plot_str = asciichartpy.plot(series_list, cfg)

                                                                 
    lines = raw_plot_str.split('\n')
    cleaned_lines = []
    last_label = None
    for line in lines:
        match = re.match(r'^(\s*[\d\.\-]+)(\s*[┤┼┬┴].*)', line)
        if match:
            lbl_val = match.group(1).strip()
            rest = match.group(2)
            if lbl_val == last_label:
                blank_spaces = ' ' * len(match.group(1))
                cleaned_lines.append(blank_spaces + rest)
            else:
                last_label = lbl_val
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    plot_str = '\n'.join(cleaned_lines)

    last_line = plot_str.split('\n')[-1]
    match = re.search(r'[┤┼┬┴]', last_line)
    y_axis_width = match.start() if match else 10

    num_labels = min(5, len(x_values)) if x_values else 5
    if x_values:
        indices = [int(i * (len(x_values) - 1) / (num_labels - 1)) for i in range(num_labels)]
        lbls = [str(x_values[i]) for i in indices]
    else:
        lbls = ["0", "25", "50", "75", "100"]

    label_line_chars = [' '] * chart_width
    axis_line_chars = ['─'] * chart_width
    axis_line_chars[0] = '└'

    for i, lbl in enumerate(lbls):
        if i == 0:
            pos = 0
            center_pos = 0
        elif i == len(lbls) - 1:
            pos = chart_width - len(lbl)
            center_pos = chart_width - 1
        else:
            if x_values and isinstance(x_values[0], (int, float)) and x_values[-1] != x_values[0]:
                x_val = float(x_values[indices[i]])
                x_min = float(x_values[0])
                x_max = float(x_values[-1])
                center_pos = int((x_val - x_min) / (x_max - x_min) * (chart_width - 1))
            else:
                center_pos = int(i * (chart_width - 1) / (len(lbls) - 1))
            pos = center_pos - len(lbl) // 2

        if 0 <= center_pos < chart_width:
            axis_line_chars[center_pos] = '┴'

        for j, c in enumerate(lbl):
            if 0 <= pos + j < chart_width:
                label_line_chars[pos + j] = c

    axis_line = " " * y_axis_width + "".join(axis_line_chars)
    label_line = " " * (y_axis_width + 1) + "".join(label_line_chars)
    title_line = " " * y_axis_width + f"{x_title:^{chart_width}}"

    plot_str += "\n" + axis_line + "\n" + label_line + "\n" + title_line

    plot_panel = Panel(
        Text.from_ansi(plot_str),
        title=f"[bold white]{title}[/]",
        border_style=get_color('accent'),
        box=box.ROUNDED,
        padding=(1, 2)
    )

    legend_panel = generate_horizontal_legend(labels)
    console.print(Align.center(plot_panel))
    console.print(Align.center(legend_panel))
    console.print()

def render_terminal_scatter(title, x_values, y_values, labels=None, x_title='X', y_title='Y'):
    if not x_values or not y_values:
        return
    import shutil
    from rich.panel import Panel
    from rich.align import Align
    from rich.text import Text
    from rich import box

    term_w, term_h = shutil.get_terminal_size((80, 24))
    plt.clf()
    plt.plotsize(max(40, term_w - 6), min(30, term_h - 4))
    plt.theme('clear')
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.grid(1, 1)

    if labels:
        for x, y, label in zip(x_values, y_values, labels):
            c_rgb = _get_plotext_color(label)
            plt.scatter([x], [y], label=label, marker='fhd', color=c_rgb)
    else:
        plt.scatter(x_values, y_values, marker='fhd', color=(0, 255, 255))

    plot_str = plt.build()
    panel = Panel(
        Text.from_ansi(plot_str),
        title=f"[bold white]{title}[/]",
        border_style=get_color('secondary'),
        box=box.ROUNDED,
        padding=(0, 1)
    )
    console.print(Align.center(panel))
    console.print()

def save_comparison_bar_chart(df, metric, dataset_name):
    if df.empty or metric not in df.columns:
        return None
    apply_neon_theme()
    fig, ax = plt_mpl.subplots(figsize=(10, 6))
    palette = {m: _get_hex_color(m) for m in df['Model'].unique()}
    sns.barplot(data=df, x='Model', y=metric, hue='Model', ax=ax, palette=palette, legend=False, capsize=0.1)
    ax.set_title(f'Comparison: {metric} ({dataset_name.upper()})', fontsize=14, pad=15)
    plt_mpl.xticks(rotation=45, ha='right')
    if metric in ['Accuracy', 'F1_Macro', 'AUC_ROC']:
        ax.set_ylim(0, 1.05)
    else:
        max_val = df[metric].max()
        ax.set_ylim(0, max_val * 1.2 if max_val > 0 else 1)

    for i, p in enumerate(ax.patches):
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f'{height:.4f}', (p.get_x() + p.get_width() / 2.0, height), ha='center', va='bottom', color='white', fontsize=9, xytext=(0, 5), textcoords='offset points', rotation=90)

    fig.tight_layout()
    return save_figure(fig, f'comparison_{metric.lower()}_{dataset_name}.png')

def save_all_comparison_charts(df, dataset_name):
    saved_paths = []
    metrics = ['Accuracy', 'F1_Macro', 'AUC_ROC', 'Train_Time_s', 'Predict_Time_s', 'Efficiency', 'Brier_Score', 'Log_Loss']
    for m in metrics:
        if m in df.columns:
            p = save_comparison_bar_chart(df, m, dataset_name)
            if p:
                saved_paths.append(p)
    return saved_paths

def save_accuracy_vs_time_scatter(df, dataset_name):
    if df.empty or 'Accuracy' not in df.columns or 'Train_Time_s' not in df.columns:
        return None
    apply_neon_theme()
    fig, ax = plt_mpl.subplots(figsize=(10, 6))
    for _, row in df.iterrows():
        c = _get_hex_color(row['Model'])
        ax.scatter(row['Train_Time_s'], row['Accuracy'], color=c, s=150, label=row['Model'], edgecolors='white')
    ax.set_title(f'Accuracy vs Train Time ({dataset_name.upper()})', fontsize=14, pad=15)
    ax.set_xlabel('Train Time (s)')
    ax.set_ylabel('Accuracy')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    return save_figure(fig, f'accuracy_vs_time_{dataset_name}.png')

def save_learning_curve(loss_history, model_name, dataset_name):
    if not loss_history:
        return None
    apply_neon_theme()
    fig, ax = plt_mpl.subplots(figsize=(10, 6))
    c = _get_hex_color(model_name)
    epochs = list(range(1, len(loss_history) + 1))
    ax.plot(epochs, loss_history, color=c, marker='o', linewidth=2, markersize=6)
    ax.set_title(f'Learning Curve: {model_name} ({dataset_name.upper()})', fontsize=14, pad=15)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, f'learning_curve_{model_name.lower()}_{dataset_name}.png')

def save_scaling_curve(df, dataset_name):
    if df.empty or 'Train_Size' not in df.columns:
        return None
    apply_neon_theme()
    fig, ax = plt_mpl.subplots(figsize=(10, 6))
    palette = {m: _get_hex_color(m) for m in df['Model'].unique()}
    sns.lineplot(data=df, x='Train_Size', y='Accuracy', hue='Model', marker='o', ax=ax, palette=palette)
    ax.set_title(f'N-Scaling on {dataset_name.upper()}', fontsize=14, pad=15)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    return save_figure(fig, f'n_scaling_{dataset_name}.png')

def save_noise_curve(df, dataset_name):
    if df.empty or 'Noise' not in df.columns:
        return None
    apply_neon_theme()
    fig, ax = plt_mpl.subplots(figsize=(10, 6))
    palette = {m: _get_hex_color(m) for m in df['Model'].unique()}
    sns.lineplot(data=df, x='Noise', y='Accuracy', hue='Model', marker='o', ax=ax, palette=palette)
    ax.set_title(f'Noise Degradation on {dataset_name.upper()}', fontsize=14, pad=15)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    return save_figure(fig, f'noise_study_{dataset_name}.png')

def save_efficiency_chart(df, dataset_name):
    return save_comparison_bar_chart(df, 'Efficiency', dataset_name)

def save_confusion_matrix(conf_mat, classes, model_name, dataset_name):
    apply_neon_theme()
    fig, ax = plt_mpl.subplots(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Purples', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(f'Confusion Matrix: {model_name.upper()} ({dataset_name.upper()})', fontsize=14, pad=15)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return save_figure(fig, f'confusion_matrix_{model_name.lower()}_{dataset_name}.png')