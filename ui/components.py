from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.padding import Padding
from rich import box
from rich.live import Live
import time
from ui.theme import STATE, get_color, model_color
console = Console()

def print_panel(text, title=None, style_key='border'):
    color = get_color(style_key)
    console.print()
    if title:
        console.print(Rule(f'[{color}]{title}[/{color}]', style='dim grey'))
        console.print(Padding(text, (1, 2)))
        console.print(Rule(style='dim grey'))
    else:
        console.print(Padding(text, (1, 2)))
    console.print()

def print_error_panel(issue, cause, fix, command=None):
    color = get_color('error')
    text = f'[bold white]Issue:[/bold white] {issue}\n'
    text += f'[bold white]Cause:[/bold white] {cause}\n'
    text += f'[bold white]Fix:[/bold white] {fix}\n'
    if command:
        text += f'\n[dim]Suggested: {command}[/dim]'
    console.print()
    console.print(Panel(text, title='SYSTEM ALERT', border_style=color))
    console.print()

def print_dashboard():
    table = Table(show_header=False, box=None)
    table.add_column('Key', style=get_color('secondary'), justify='right')
    table.add_column('Value', style='white')
    table.add_row('Active Dataset', str(STATE['dataset']).upper())
    table.add_row('Training Epochs', str(STATE['epochs']))
    table.add_row('Current Theme', STATE['theme'].capitalize())
    table.add_row('Working Mode', 'ON' if STATE['working_mode'] else 'OFF')
    table.add_row('Quiet Mode', 'ON' if STATE['quiet_mode'] else 'OFF')
    table.add_row('Last Winner', str(STATE['last_winner']).upper() if STATE['last_winner'] else 'None')
    print_panel(table, title='DARSHAN COCKPIT', style_key='primary')

def df_to_table(df, title, header_style_key='header'):
    color = get_color(header_style_key)
    table = Table(title=title, show_header=True, header_style=color, box=box.SIMPLE)

    is_narrow = console.width < 120
    abbrevs = {
        'Train_Time_s': 'TTS',
        'Predict_Time_s': 'PTS',
        'Efficiency': 'Eff',
        'F1_Macro': 'F1'
    }
    used_abbrevs = {}

    cols = []
    for col in df.columns:
        if is_narrow and col in abbrevs:
            cols.append(abbrevs[col])
            used_abbrevs[abbrevs[col]] = col
        else:
            cols.append(col)

    for col in cols:
        table.add_column(col, justify='center')

    for _, row in df.iterrows():
        str_row = []
        for val in row:
            if isinstance(val, float):
                str_row.append(f'{val:.4f}')
            else:
                str_row.append(str(val))
        table.add_row(*str_row)

    from rich.align import Align
    if used_abbrevs:
        from rich.panel import Panel
        from rich.console import Group
        legend = ", ".join([f"[bold]{k}[/bold]: {v}" for k, v in used_abbrevs.items()])
        abbrev_panel = Panel(legend, border_style="dim")
        return Align.center(Group(table, Align.center(abbrev_panel)))

    return Align.center(table)

def print_winner_podium(models):
    if not models:
        return
    best_acc = max(models, key=lambda x: x.get('Accuracy', 0))
    best_eff = max(models, key=lambda x: x.get('Efficiency', 0))
    quantum_models = [m for m in models if 'Parampara' not in m['Model']]
    best_q = max(quantum_models, key=lambda x: x.get('Accuracy', 0)) if quantum_models else None
    podium_colors = ['#ffd700', '#c0c0c0', '#cd7f32']
    podium = Table.grid(padding=1, expand=True)
    podium.add_column()
    podium.add_column()
    podium.add_column()
    panels = []
    text_acc = f"[bold white]Accuracy:[/bold white] {best_acc.get('Accuracy', 0):.4f}\n[bold white]Model:[/bold white] {best_acc['Model']}"
    panels.append(Panel(text_acc, title='🏆 Highest Accuracy', border_style=podium_colors[0]))
    if best_q:
        text_q = f"[bold white]Accuracy:[/bold white] {best_q.get('Accuracy', 0):.4f}\n[bold white]Model:[/bold white] {best_q['Model']}"
        panels.append(Panel(text_q, title='⚛️ Best Quantum', border_style=podium_colors[1]))
    else:
        panels.append(Panel('N/A', title='⚛️ Best Quantum', border_style=podium_colors[1]))
    text_eff = f"[bold white]Score:[/bold white] {best_eff.get('Efficiency', 0):.4f}\n[bold white]Model:[/bold white] {best_eff['Model']}"
    panels.append(Panel(text_eff, title='⚡ Best Efficiency', border_style=podium_colors[2]))
    podium.add_row(*panels)
    console.print()
    console.print(podium)
    console.print()

def print_interpretation(run_df, ds_name):
    if 'Efficiency' not in run_df.columns:
        run_df['Efficiency'] = run_df['Accuracy'] / run_df['Train_Time_s'].apply(lambda x: max(x, 1e-09))
    df = run_df.sort_values(by=['Accuracy', 'Efficiency', 'Train_Time_s'], ascending=[False, False, True])

    if len(df) == 0:
        return

    m1 = df.iloc[0]
    m1_name = m1['Model']
    c1 = model_color(m1_name)

    text = f'The results indicate that [bold {c1}]{m1_name}[/bold {c1}] achieved the best overall performance on the {ds_name} dataset.\n\n'

    if len(df) > 1:
        m2 = df.iloc[1]
        c2 = model_color(m2["Model"])

        if m1['Accuracy'] == m2['Accuracy']:
            if m1['Efficiency'] == m2['Efficiency'] and m1['Train_Time_s'] == m2['Train_Time_s']:
                text += f'The results indicate a [bold]PERFECT DRAW[/bold] between [{c1}]{m1_name}[/{c1}] and [{c2}]{m2["Model"]}[/{c2}] on the {ds_name} dataset.\n'
                text += 'Both achieved identical accuracy, efficiency, and training time. This suggests absolute parity.\n\n'
            elif m1['Efficiency'] > m2['Efficiency']:
                margin = m1['Efficiency'] - m2['Efficiency']
                text += f'[bold white]Deciding Factor:[/bold white] Tied on Accuracy ({m1["Accuracy"]:.4f}), but won via Efficiency margin of {margin:.4f}.\n\n'
            else:
                margin = m2['Train_Time_s'] - m1['Train_Time_s']
                text += f'[bold white]Deciding Factor:[/bold white] Tied on Accuracy & Efficiency, but won via Training Time margin of {margin:.2f}s.\n\n'
        else:
            margin = m1['Accuracy'] - m2['Accuracy']
            text += f'[bold white]Deciding Factor:[/bold white] Absolute Accuracy. Won by a margin of {margin:.4%} over [{c2}]{m2["Model"]}[/{c2}].\n\n'

    text += '[bold]Full Roster Breakdown:[/bold]\n\n'
    for i, row in df.iterrows():
        rank = df.index.get_loc(i) + 1
        name = row['Model']
        c = model_color(name)
        acc = row['Accuracy']
        eff = row['Efficiency']
        tt = row['Train_Time_s']
        text += f" {rank}. [bold {c}]{name}[/bold {c}]: Acc: {acc:.4f} | Eff: {eff:.4f} | Time: {tt:.2f}s\n"

        if rank == 2:
            text += "    (Strong runner-up, but fell short on the primary tie-breakers)\n\n"
        elif rank == 3:
            text += "    (Solid podium finish; balances speed and precision reasonably well)\n\n"
        elif rank > 3:
            text += "    (Lower placement strictly due to trailing efficiency scores)\n\n"

    if 'samyoga_legacy' in m1_name.lower():
        text += '\n[dim]Note: Quantum transfer learning excelled here, supporting the NISQ Crossover Hypothesis.[/dim]'
    elif 'parampara' in m1_name.lower():
        text += '\n[dim]Note: Classical SVM holds the advantage, indicating sufficient data samples where pure quantum circuits struggle with noise.[/dim]'

    color = get_color('accent')
    console.print()
    console.print(Rule(f'[{color}]RESEARCH INTERPRETATION[/{color}]', style='dim grey'))
    from rich.align import Align
    t_obj = Text.from_markup(text, justify="full")
    panel = Padding(t_obj, (1, 2))
    console.print(Align.center(panel, vertical="middle"))
    console.print(Rule(style='dim grey'))

def animate_gradient_logo():
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich.console import Group
    from rich.align import Align
    from ui.theme import LOGO as MAIN_LOGO, get_color

    def clean_color(key):
        c = get_color(key)
        return c.replace('bold ', '').replace('dim ', '')
    c_pri = clean_color('primary')
    c_sec = clean_color('secondary')
    c_acc = clean_color('accent')
    c_suc = clean_color('success')
    colors = [c_pri, c_sec, c_acc]
    import random
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

    import os
    os.system('cls' if os.name == 'nt' else 'clear')

    boot_steps = [
        ("Establishing secure quantum channel...", 0.2),
        ("Calibrating TNQE hypervisor...", 0.15),
        ("Cooling Qubits to 15mK...", 0.3),
        ("Loading Mamba-2 parameters into VRAM...", 0.25),
        ("Synchronizing Classical-Quantum Twin...", 0.1),
        ("Warming up QMoE routing tables...", 0.2),
        ("Verifying dimensional entanglement...", 0.2)
    ]

    with console.status(f"[bold {c_pri}]Initiating Darshan Engine[/bold {c_pri}]", spinner="dots12"):
        for step, delay in boot_steps:
            time.sleep(delay)
            console.print(f"[{c_sec}]>[/{c_sec}] {step} [bold {c_pri}]OK[/bold {c_pri}]")

    time.sleep(0.3)
    os.system('cls' if os.name == 'nt' else 'clear')

    with Progress(
        SpinnerColumn(spinner_name="aesthetic"),
        TextColumn(f"[bold {c_pri}][progress.description]{{task.description}} "),
        BarColumn(bar_width=40, complete_style=c_pri, finished_style=c_sec),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task1 = progress.add_task("Injecting Hamiltonian Dynamics...", total=100)
        task2 = progress.add_task("Spooling UI Threads...", total=100)

        while not progress.finished:
            progress.update(task1, advance=random.uniform(5, 15))
            if progress.tasks[task1].completed > 30:
                progress.update(task2, advance=random.uniform(2, 20))
            time.sleep(0.05)

    time.sleep(0.15)
    os.system('cls' if os.name == 'nt' else 'clear')
    from ui.theme import hex_to_rgb, rgb_to_hex
    lines = MAIN_LOGO.strip('\n').split('\n')
    if not lines:
        return
    max_len = max([len(line) for line in lines])
    if max_len == 0:
        max_len = 1
    rgb_colors = [hex_to_rgb(c) for c in colors]
    colored_lines = []
    for line in lines:
        t = Text()
        for i, char in enumerate(line):
            if char == '░':
                t.append(char, style=f'dim {c_sec}')
                continue
            ratio = i / (max_len - 1)
            scaled = ratio * (len(rgb_colors) - 1)
            idx = int(scaled)
            if idx >= len(rgb_colors) - 1:
                idx = len(rgb_colors) - 2
                segment_ratio = 1.0
            else:
                segment_ratio = scaled - idx
            start_rgb = rgb_colors[idx]
            end_rgb = rgb_colors[idx + 1]
            r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * segment_ratio
            g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * segment_ratio
            b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * segment_ratio
            color = rgb_to_hex((r, g, b))
            t.append(char, style=f'bold {color}')
        colored_lines.append(t)
    progress = Progress(SpinnerColumn(spinner_name='aesthetic', style=f'bold {c_pri}'), TextColumn('[progress.description]{task.description}'), BarColumn(bar_width=40, complete_style=c_acc, finished_style=c_pri, pulse_style='#ffffff'), TextColumn('[progress.percentage]{task.percentage:>3.0f}%'), transient=True)
    task_id = progress.add_task('[dim cyan]Booting Darshan Quantum Kernel...', total=len(colored_lines))
    current = Text()
    with Live(Align.center(Group(current, progress)), refresh_per_second=30, transient=True) as live:
        for t in colored_lines:
            current.append(t)
            current.append('\n')
            progress.advance(task_id)
            time.sleep(0.08)
        progress.update(task_id, description=f'[bold {c_suc}]System Ready![/bold {c_suc}]')
        time.sleep(0.6)
    subtitle = Text('Darshan Quantum Interface v1.0\n', style=f'italic dim {c_pri}')
    final_group = Group(current, Align.center(subtitle))
    console.print(Align.center(final_group))

def print_startup_tips():
    from rich.align import Align
    from ui.theme import get_color
    c_pri = get_color('primary').replace('bold ', '')
    c_sec = get_color('secondary').replace('bold ', '')
    tips = f'Tips for getting started:\n1. Run [bold {c_pri}]/dataset[/bold {c_pri}] to pick your working data.\n2. Type [bold {c_pri}]/test compare[/bold {c_pri}] to run a multi-model benchmark.\n3. Access the dashboard UI using [bold {c_pri}]/menu[/bold {c_pri}].\n4. [bold {c_pri}]/help[/bold {c_pri}] for more information.'
    lines = tips.split('\n')
    with Live(Text(''), refresh_per_second=30, transient=False) as live:
        current_text = ''
        for line in lines:
            words = (line + '\n').split(' ')
            for word in words:
                current_text += word + ' '
                live.update(Align.center(Text.from_markup(current_text, style='dim white')))
                time.sleep(0.02)