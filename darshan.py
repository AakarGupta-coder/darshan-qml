import os
import matplotlib
matplotlib.use('Agg')
import sys
import json
import datetime
import time
import pandas as pd
import numpy as np
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
import questionary
from ui.theme import get_q_style
from ui.theme import STATE, THEMES, RAW_LOGOS, LOGO as MAIN_LOGO, get_color, model_color, apply_theme_to_prompt
from ui.components import print_panel, print_error_panel, print_dashboard, df_to_table, print_winner_podium, print_interpretation
from ui.graphs import save_confusion_matrix
from utils.logger import WorkingLog
warnings.filterwarnings('ignore')
console = Console()
working_logger = WorkingLog()
ACTIVE_MODELS = {}
HISTORY_FILE = 'results/history.json'


def log_experiment(exp_type, dataset, model, accuracy, time_s, notes=''):
    os.makedirs('results', exist_ok=True)
    record = {'timestamp': datetime.datetime.now().isoformat(), 'type': exp_type, 'dataset': dataset, 'model': model, 'accuracy': accuracy, 'time_s': time_s, 'notes': notes}
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                history = json.load(f)
            except:
                pass
    history.append(record)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def print_help():
    c_p = get_color('primary')
    c_s = get_color('secondary')
    c_a = get_color('accent')
    c_h = get_color('header')
    c_w = get_color('warning')
    c_suc = get_color('success')
    table = Table(show_header=True, header_style=c_h, expand=True)
    table.add_column('Command', style=c_p)
    table.add_column('Description', style='white')
    table.add_row(f'[{c_a}]CORE SYSTEMS[/{c_a}]', '')
    table.add_section()
    table.add_row('/dataset [name]', 'Select dataset')
    table.add_row('/model [name]', 'Inspect model, run CV/Benchmarks, view Conf Matrix')
    table.add_row('/test [name]', 'Run a comparative experiment (compare, scaling, noise)')
    table.add_row('/epochs [num]', 'Set VQC training epochs')
    table.add_row('/train continue [model] [epochs]', 'Progressively train a loaded model')
    table.add_row('/checkpoint [save|load] [model]', 'Save/Load model state')
    table.add_section()
    table.add_row(f'[{c_w}]RESEARCH OPS[/{c_w}]', '')
    table.add_section()
    table.add_row('/results', 'Dashboard summarizing historical experiment win rates')
    table.add_row('/history', 'View recent detailed experiment logs')
    table.add_row('/report', 'Auto-generate a markdown research report')
    table.add_row('/graphs [arg]', 'Manage and view terminal graphs')
    table.add_section()
    table.add_row(f'[{c_suc}]INTERFACE[/{c_suc}]', '')
    table.add_section()
    table.add_row('/theme [name]', 'Change visual theme (forest, ember, quantum, etc)')
    table.add_row('/menu', 'Open the interactive Command Center')
    table.add_row('/working [on|off|show|clear]', 'Manage noisy raw logs')
    table.add_row('/quiet [on|off]', 'Toggle minimal progress view')
    table.add_row('/about', 'Read details about Darshan and the research study')
    table.add_row('/clear', 'Clear the terminal screen')
    table.add_row('/reset', 'Factory reset: deletes all caches, metrics, graphs, and history')
    table.add_row('/restart', 'Reset state and replay boot sequence')
    table.add_row('/exit', 'Exit Darshan')
    print_panel(table, title='Darshan Commands', style_key='border')

def _ensure_darshan_spinner():
    from rich.spinner import SPINNERS
    if 'darshan_arrow' not in SPINNERS:
        SPINNERS['darshan_arrow'] = {'interval': 150, 'frames': [' ▸ ', '▸▸ ', '▸▸▸', ' ▸▸', '  ▸']}

def _make_progress():
    _ensure_darshan_spinner()
    return Progress(SpinnerColumn(spinner_name='darshan_arrow', style=get_color('primary')), TextColumn(f"[{get_color('primary')}]{{task.description}} "), BarColumn(bar_width=30, complete_style=get_color('success'), finished_style=get_color('accent')), TextColumn(f"[{get_color('accent')}]" + '{task.percentage:>3.0f}%'), TimeElapsedColumn(), console=console, transient=False)

class ModelProgressCallback:

    def __init__(self, progress, task_id):
        self.progress = progress
        self.task_id = task_id

    def on_stage_start(self, stage_name, total_steps=100):
        self.progress.update(self.task_id, description=stage_name, total=total_steps, completed=0)

    def on_stage_end(self, stage_name):
        self.progress.update(self.task_id, description=f'✓ {stage_name}', completed=self.progress.tasks[self.task_id].total)

    def on_epoch_end(self, epoch, loss, model_name, total_epochs):
        desc = f'{model_name}: Epoch {epoch + 1}/{total_epochs} (Loss: {loss:.4f})'
        self.progress.update(self.task_id, description=desc, completed=epoch + 1, total=total_epochs)

    def on_batch_end(self, batch_index, total_batches):
        self.progress.update(self.task_id, description=f'Batch {batch_index}/{total_batches}')

def run_experiment_staged(experiment_func, func_args, description):
    if not STATE['quiet_mode']:
        with _make_progress() as progress:
            task = progress.add_task(description, total=100)
            cb = ModelProgressCallback(progress, task)
            if 'callbacks' not in func_args:
                func_args['callbacks'] = []
            func_args['callbacks'].append(cb)
            try:
                result_df = experiment_func(**func_args)
            except Exception as e:
                print_error_panel('Experiment Failed', str(e), 'Check dataset and epochs.', '/test compare')
                return None
            progress.update(task, description=f'[bold]✓ Complete![/bold]', completed=100, total=100)
        return result_df
    else:
        try:
            return experiment_func(**func_args)
        except Exception as e:
            print_error_panel('Experiment Failed', str(e), 'Check dataset and epochs.', '/test compare')
            return None

def cmd_dataset(args):
    from data.loader import DATASET_CONFIGS, load_dataset
    if not args:
        choices = [f"{config.get('size_label', '')} {name}".strip() for name, config in DATASET_CONFIGS.items()]
        selection = questionary.select('Select a dataset:', choices=choices, style=get_q_style()).ask()
        if not selection:
            return
        name = selection.split(' ')[-1].lower()
    else:
        name = args[0].lower()
        if name not in DATASET_CONFIGS:
            print_error_panel('Unknown dataset', f"'{name}' not in registry.", 'Choose a valid dataset.', '/dataset')
            return
    STATE['dataset'] = name
    with _make_progress() as progress:
        task = progress.add_task(f'Loading {name}...', total=100)
        progress.update(task, advance=30, description=f'Fetching {name} data...')
        data = load_dataset(name)
        progress.update(task, advance=70, description=f'[bold]✓ {name.upper()} Loaded![/bold]')
        time.sleep(0.2)
    table = Table(show_header=True, header_style=get_color('success'), box=box.ROUNDED)
    table.add_column('Property', style=get_color('secondary'), justify='right')
    table.add_column('Value', style='white')
    table.add_row('Total Samples', str(data['X_train'].shape[0] + data['X_test'].shape[0]))
    table.add_row('Encoded Qubits (Features)', str(data['n_features']))
    table.add_row('Number of Classes', str(len(np.unique(data['y_train']))))
    table.add_row('Train Samples', str(data['X_train'].shape[0]))
    table.add_row('Test Samples', str(data['X_test'].shape[0]))
    from ui.theme import animate_panel
    from rich.console import Group
    from rich.align import Align
    animate_panel(data['description'], title=f'{name.upper()} DATASET PROFILE', border_style=get_color('primary'), delay=0.01, footer=Group(' ', Align.center(table)))

def get_model_instance(name, n_features):
    if name not in ACTIVE_MODELS:
        if name == 'samyoga_legacy':
            from models.samyoga_svm import SamyogaLegacySVM
            ACTIVE_MODELS[name] = SamyogaLegacySVM(n_qubits=n_features, epochs_pretrain=STATE['epochs'])
        elif name == 'samyoga_pro':
            from models.samyoga_pro import SamyogaPro
            ACTIVE_MODELS[name] = SamyogaPro(d_model=n_features, d_state=8, qubits=4, num_experts=3)
        elif name == 'samyoga_go':
            from models.samyoga_go import SamyogaGo
            ACTIVE_MODELS[name] = SamyogaGo(d_model=n_features, d_state=8, qubits=4, num_experts=3)
        elif name == 'samyoga_shadow':
            from models.samyoga_shadow import SamyogaShadow
            ACTIVE_MODELS[name] = SamyogaShadow(d_model=n_features, d_state=8, num_experts=3)
        elif name == 'parampara_legacy':
            from models.parampara_svm import ParamparaLegacy
            ACTIVE_MODELS[name] = ParamparaLegacy()
        elif name == 'parampara':
            from models.parampara_pro import ParamparaPro
            ACTIVE_MODELS[name] = ParamparaPro(mode='fair', n_qubits=n_features)
        elif name == 'parampara_pro_industry':
            from models.parampara_pro import ParamparaPro
            ACTIVE_MODELS[name] = ParamparaPro(mode='industry', n_qubits=n_features)
        elif name == 'ananta':
            from models.ananta_vqc import AnantaVQC
            ACTIVE_MODELS[name] = AnantaVQC(n_qubits=n_features, epochs=STATE['epochs'])
        elif name == 'ananta_pro':
            from models.ananta_pro import AnantaPro
            ACTIVE_MODELS[name] = AnantaPro(n_qubits=n_features)
    return ACTIVE_MODELS.get(name)

def cmd_train(args):
    if not STATE['dataset']:
        console.print(f"[{get_color('warning')}]No dataset loaded. Let's pick one first...[/{get_color('warning')}]")
        cmd_dataset([])
        if not STATE['dataset']:
            return
    if len(args) < 3 or args[0] != 'continue':
        print_error_panel('Invalid syntax', 'Usage: /train continue <model> <epochs>', 'Example: /train continue ananta 5', '/help')
        return
    model_name = args[1].lower()
    if model_name not in ['ananta', 'samyoga_legacy']:
        print_error_panel('Invalid model', 'Only Ananta and Samyoga Legacy support progressive training.', 'Check spelling.', '/help')
        return
    try:
        additional_epochs = int(args[2])
    except ValueError:
        print_error_panel('Invalid epochs', 'Must be an integer.', 'Example: /train continue ananta 5', '/help')
        return
    from data.loader import load_dataset
    data = load_dataset(STATE['dataset'])
    model = get_model_instance(model_name, data['n_features'])
    print_panel(f'Resuming training for [bold]{model_name}[/bold] for {additional_epochs} epochs...', style_key='primary')

    def _train(callbacks=None):
        model.continue_training(data['X_train'], data['y_train'], additional_epochs=additional_epochs)
        if hasattr(model, 'loss_history_') and model.loss_history_:
            from ui.graphs import save_learning_curve
            p = save_learning_curve(model.loss_history_, model_name, STATE['dataset'])
            if p:
                print_panel(f'Learning curve saved to {p}', style_key='secondary')
        return pd.DataFrame([model.evaluate(data['X_test'], data['y_test'])])
    df = run_experiment_staged(_train, {}, f'Continuing {model_name} training')
    if df is not None:
        console.print(df_to_table(df, f'CONTINUED TRAINING RESULT'))

def cmd_checkpoint(args):
    if len(args) < 2 or args[0] not in ['save', 'load']:
        print_error_panel('Invalid syntax', 'Usage: /checkpoint <save|load> <model>', 'Example: /checkpoint save ananta', '/help')
        return
    action = args[0]
    model_name = args[1].lower()
    if model_name not in ['ananta', 'samyoga_legacy']:
        print_error_panel('Invalid model', 'Only Ananta and Samyoga Legacy support checkpointing.', 'Check spelling.', '/help')
        return
    if action == 'save':
        if model_name not in ACTIVE_MODELS:
            print_error_panel('Model not active', f'{model_name} has not been initialized or trained yet.', 'Train it first or load it.', '/test compare')
            return
        path = f'results/checkpoints/{model_name}.npz'
        ACTIVE_MODELS[model_name].save_checkpoint(path)
        print_panel(f'Saved {model_name} checkpoint to {path}', style_key='success')
    elif action == 'load':
        path = f'results/checkpoints/{model_name}.npz'
        if not os.path.exists(path):
            print_error_panel('Checkpoint not found', f'No checkpoint exists at {path}', 'Save one first.', '/checkpoint save')
            return
        if not STATE['dataset']:
            console.print(f"[{get_color('warning')}]No dataset loaded. Let's pick one first...[/{get_color('warning')}]")
            cmd_dataset([])
            if not STATE['dataset']:
                return
        from data.loader import load_dataset
        data = load_dataset(STATE['dataset'])
        model = get_model_instance(model_name, data['n_features'])
        model.load_checkpoint(path)
        print_panel(f'Loaded {model_name} checkpoint from {path}', style_key='success')

def cmd_model(args):
    if not args:
        choices = ['Samyoga (Hybrid Quantum Transfer Learning)', 'Parampara (Classical RBF Support Vector Machine)', 'Ananta (Quantum Classifiers)']
        choice_str = questionary.select('Which model would you like to inspect?', choices=choices, style=get_q_style()).ask()
        if not choice_str:
            return
        if choice_str.startswith('Samyoga'):
            name = 'samyoga'
        elif choice_str.startswith('Parampara'):
            name = 'parampara'
        elif choice_str.startswith('Ananta'):
            name = 'ananta'
        else:
            name = choice_str.split()[0].lower()

        if name == 'samyoga':
            sub_choice = questionary.select('Which version of Samyoga?', choices=['Samyoga Pro (Enterprise-Grade QTL)', 'Samyoga Go (Instant Classical Mock)', 'Samyoga Shadow (Trainable Classical Twin)', 'Samyoga Legacy (Legacy Hybrid QTL + SVM)'], style=get_q_style()).ask()
            if not sub_choice:
                return
            if 'Pro' in sub_choice:
                name = 'samyoga_pro'
            elif 'Go' in sub_choice:
                name = 'samyoga_go'
            elif 'Shadow' in sub_choice:
                name = 'samyoga_shadow'
            else:
                name = 'samyoga_legacy'
        if name == 'parampara':
            sub_choice = questionary.select('Which version of Parampara?', choices=['Parampara Pro (Tuned & Scalable)', 'Parampara Legacy (Untuned)'], style=get_q_style()).ask()
            if not sub_choice:
                return
            if 'Legacy' in sub_choice:
                name = 'parampara_legacy'
            else:
                sub_choice_2 = questionary.select('Which Parampara Pro Track?', choices=['Parampara Pro (Fair Track - PCA Bounded)', 'Parampara Pro Industry (Industry Track - Unbounded)'], style=get_q_style()).ask()
                if not sub_choice_2:
                    return
                if 'industry' in sub_choice_2.lower():
                    name = 'parampara_pro_industry'
                else:
                    name = 'parampara'
        elif name == 'ananta':
            sub_choice = questionary.select('Which version of Ananta?', choices=['Ananta Pro (Enterprise-Grade Hybrid Extractor)', 'Ananta Legacy (Uncalibrated VQC Strawman)'], style=get_q_style()).ask()
            if not sub_choice:
                return
            if 'pro' in sub_choice.lower():
                name = 'ananta_pro'
            else:
                name = 'ananta'
    else:
        name = args[0].lower()
    base_name = name.split('_')[0] if 'parampara' in name else name
    if base_name not in RAW_LOGOS:
        print_error_panel('Unknown model', f"'{name}' not recognized.", 'Choose a valid model.', '/model')
        return
    stats = {}
    about_text = ''
    if name == 'samyoga_pro':
        about_text = 'Samyoga Pro is a massive, full-fledged enterprise-grade hybrid architecture. It maps data through Tensor Network Quantum Encodings (TNQE), Spiking-Quantum CNNs, and a Mamba-2 State Space Model before routing through a Quantum Mixture-of-Experts (QMoE) and Quantized Self-Attention.'
        stats = {'Architecture': 'TNQE + Mamba2 + QMoE', 'Optimization': 'QNG + DS-ZNE Error Mitigation', 'NISQ Viability': '[bold red]Theoretical / Low[/bold red] (Requires Fault Tolerance)', 'Complexity': 'O(n^2 log d)'}
    elif name == 'samyoga_go':
        about_text = 'Samyoga Go is the pure classical mock of Samyoga Pro. It implements mathematically exact O(1) vectorized NumPy probability overlaps in place of PennyLane quantum circuits to allow for instant execution and rapid prototyping without sacrificing theoretical accuracy.'
        stats = {'Architecture': 'Pure Classical Mock', 'Equivalent': 'Samyoga Pro (Quantum)', 'Training Time': '[bold green]Instant[/bold green]', 'NISQ Viability': 'N/A (Classical)'}
    elif name == 'samyoga_shadow':
        about_text = 'Samyoga Shadow is a Classical Twin mock. It replaces all Quantum modules (TNQE, QMoE, Mamba-2) with classical MLP equivalents of the exact same parameter size. It is used exclusively in Fair-Track benchmarking to test true Quantum Advantage.'
        stats = {'Architecture': 'Classical MLP Mock', 'Equivalent': 'Samyoga Pro (Classical)', 'Parameters': 'Matched to Quantum', 'NISQ Viability': 'N/A (Classical)'}
    elif name == 'samyoga_legacy':
        about_text = 'Samyoga Legacy is an advanced hybrid model that combines the high-dimensional feature mapping capabilities of PennyLane Quantum Variational Circuits (VQC) with the robust margin maximization of Classical RBF SVMs. It is heavily optimized for Near-Term Intermediate Scale Quantum (NISQ) devices.'
        stats = {'Architecture': 'Quantum Transfer Learning (Hybrid)', 'Stage 1': 'PennyLane VQC', 'Stage 2': 'Classical RBF SVM', 'Qubits Used': 'Depends on PCA', 'NISQ Viability': '[bold green]Extremely High[/bold green]'}
    elif 'parampara_legacy' in name:
        about_text = "Parampara Legacy is the old, untuned baseline. It serves as a 'strawman' to show how weak a naive classical SVM can be if hyperparameters are not aggressively tuned using GridSearchCV."
        stats = {'Architecture': 'Untuned SVM', 'Kernel': 'RBF (Default)', 'Features': 'Standard', 'NISQ Viability': 'N/A'}
    elif 'parampara_pro_industry' in name:
        about_text = 'Parampara Pro (Industry) is the true, rigorous classical champion. It uses full feature dimensions (no PCA limit) and scales to HistGradientBoosting for N>10k, representing the maximum classical performance barrier.'
        stats = {'Architecture': 'Rigorous Classical Pipeline', 'Tuning': 'RandomizedSearchCV', 'Features': 'Unbounded (No PCA)', 'NISQ Viability': 'N/A'}
    elif name == 'parampara':
        about_text = 'Parampara Pro (Fair Track) is the rigorous classical champion restricted to the exact same PCA bounded dimensionality as the quantum circuits (e.g. 4 qubits). It uses aggressive RandomizedSearchCV to maximize performance.'
        stats = {'Architecture': 'Rigorous Classical Pipeline', 'Tuning': 'RandomizedSearchCV', 'Features': 'PCA Bounded (Fair)', 'NISQ Viability': 'N/A'}
    elif name == 'ananta':
        about_text = 'Ananta is a pure Quantum Variational Classifier (VQC) implemented via PennyLane. It constructs a heavily parameterized quantum circuit using Rx, Ry, Rz and entangling CNOT gates to learn non-linear boundaries entirely within the Hilbert space.'
        stats = {'Architecture': 'Pure Quantum VQC', 'Quantum Gates': 'Rx, Ry, Rz, CNOT', 'Layers': '3 to 5', 'NISQ Viability': '[bold yellow]Moderate[/bold yellow]'}
    elif name == 'ananta_pro':
        about_text = 'Ananta Pro is the Enterprise-Grade Hybrid Extractor. It extracts multi-seed quantum projected features and classically concatenates them for a highly calibrated robust classical head selection via RandomizedSearchCV.'
        stats = {'Architecture': 'Hybrid Extractor + RBF', 'Features': 'PCA + Quantum Z/Y/X', 'Tuning': 'RandomizedSearchCV', 'NISQ Viability': '[bold green]Extremely High[/bold green]'}
    stats_table = Table(title=None, box=box.ROUNDED, show_header=False, border_style=model_color(name), padding=(0, 3))
    stats_table.add_column('Property', style=get_color('secondary'))
    stats_table.add_column('Value', style='white')
    for k, v in stats.items():
        stats_table.add_row(k + ':', v)
    from ui.theme import animate_panel, get_model_logo
    from rich.console import Group
    from rich.text import Text as RichText
    from rich.align import Align
    from rich.table import Table as RichTable
    logo_str = get_model_logo(name)
    logo_table = RichTable(show_header=False, box=None, padding=(0, 0))
    logo_table.add_column(justify='left')
    logo_table.add_row(logo_str)

    profile_title = RichText(f'{name.upper()} Profile', style=f'bold {model_color(name)}', justify='center')
    footer_group = Group(' ', profile_title, Align.center(stats_table))
    animate_panel(about_text, title=f'{name.upper()} DETAILS', border_style=model_color(name), delay=0.01, header=Align.center(logo_table), footer=footer_group)
    if not STATE.get('dataset'):
        console.print(f'[dim]Select a dataset with /dataset to run benchmarks on {name.upper()}.[/dim]')
        return
    run_bench = questionary.confirm(f"Run a benchmark on {name.upper()} using '{STATE['dataset']}'?", style=get_q_style()).ask()
    if run_bench:
        val_strategy = questionary.select('Select Validation Strategy:', choices=['Standard Split (80/20) - Fast baseline testing', '5-Fold CV - Robust statistical validation', 'Monte Carlo (5 iterations) - Randomized resampling', 'Leave-One-Out (LOOCV) - Maximum data utilization'], style=get_q_style()).ask()
        val_strategy = val_strategy.split(' - ')[0]
        from data.loader import load_dataset
        from sklearn.metrics import accuracy_score, confusion_matrix
        from sklearn.model_selection import KFold, ShuffleSplit, LeaveOneOut
        data = load_dataset(STATE['dataset'])
        acc, train_t, conf_mat = (0, 0, None)
        notes = val_strategy
        X_full = np.vstack((data['X_train'], data['X_test']))
        y_full = np.concatenate((data['y_train'], data['y_test']))
        import threading
        with _make_progress() as progress:
            if 'Standard' in val_strategy:
                task = progress.add_task('Preparing model...', total=100)
                progress.update(task, advance=10, description=f'Loading {name.upper()} architecture...')
                model_obj = get_model_instance(name, data['n_features'])
                time.sleep(0.15)
                progress.update(task, advance=10, description='Fitting training data...')
                error_h = [None]

                def _fit():
                    try:
                        model_obj.fit(data['X_train'], data['y_train'])
                    except Exception as e:
                        error_h[0] = e
                start_t = time.time()
                t = threading.Thread(target=_fit)
                t.start()
                while t.is_alive():
                    if progress.tasks[task].completed < 65:
                        progress.update(task, advance=0.5)
                    time.sleep(0.1)
                t.join()
                train_t = time.time() - start_t
                if error_h[0]:
                    progress.update(task, advance=100, description='Failed!')
                    progress.stop()
                    print_error_panel('Training Failed', str(error_h[0]), 'Check parameters.', '/model')
                    return
                progress.update(task, advance=max(0, 70 - progress.tasks[task].completed), description='Running predictions...')
                preds = model_obj.predict(data['X_test'])
                progress.update(task, advance=15, description='Computing metrics...')
                acc = accuracy_score(data['y_test'], preds)
                conf_mat = confusion_matrix(data['y_test'], preds)
                progress.update(task, advance=10, description='Building results...')
                time.sleep(0.15)
                progress.update(task, advance=100 - progress.tasks[task].completed, description=f'[bold]✓ {name.upper()} Done![/bold]')
                time.sleep(0.3)
                res_table = Table(title=f'{name.upper()} Benchmark Results', show_header=True, header_style=get_color('success'))
                res_table.add_column('Dataset', justify='center')
                res_table.add_column('Accuracy', justify='center')
                res_table.add_column('Train Time (s)', justify='center')
                res_table.add_row(STATE['dataset'], f'{acc:.4f}', f'{train_t:.2f}')
            else:
                if '5-Fold' in val_strategy:
                    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
                    splits = 5
                elif 'Monte Carlo' in val_strategy:
                    splitter = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
                    splits = 5
                elif 'LOOCV' in val_strategy:
                    splitter = LeaveOneOut()
                    splits = len(X_full)
                    if splits > 100:
                        splits = 30
                scores, times = ([], [])
                cv_task = progress.add_task(f'Fold 1/{splits}: Preparing...', total=splits)
                for i, (train_idx, val_idx) in enumerate(splitter.split(X_full)):
                    if i >= splits:
                        break
                    progress.update(cv_task, description=f'Fold {i + 1}/{splits}: Training {name.upper()}...')
                    model_obj = get_model_instance(name, data['n_features'])
                    start_t = time.time()
                    model_obj.fit(X_full[train_idx], y_full[train_idx])
                    times.append(time.time() - start_t)
                    progress.update(cv_task, description=f'Fold {i + 1}/{splits}: Predicting...')
                    preds = model_obj.predict(X_full[val_idx])
                    scores.append(accuracy_score(y_full[val_idx], preds))
                    progress.advance(cv_task)
                progress.update(cv_task, description=f'[bold]✓ {val_strategy} Complete![/bold]')
                time.sleep(0.3)
                acc, train_t = (np.mean(scores), np.mean(times))
                res_table = Table(title=f'{name.upper()} {val_strategy} Results', show_header=True, header_style=get_color('success'))
                res_table.add_column('Dataset', justify='center')
                res_table.add_column('Mean Accuracy', justify='center')
                res_table.add_column('Avg Train Time (s)', justify='center')
                res_table.add_row(STATE['dataset'], f'{acc:.4f} ± {np.std(scores):.4f}', f'{train_t:.2f}')
        console.print()
        console.print(res_table)
        console.print()
        if conf_mat is not None:
            cm_table = Table(title='Confusion Matrix', show_header=True, header_style=get_color('accent'))
            cm_table.add_column('Actual \\ Predicted')
            for i in range(conf_mat.shape[1]):
                cm_table.add_column(f'Pred Class {i}')
            for i in range(conf_mat.shape[0]):
                row = [f'True Class {i}'] + [str(v) for v in conf_mat[i]]
                cm_table.add_row(*row)
            console.print(cm_table)
            try:
                classes = [str(c) for c in np.unique(data['y_train'])]
                p = save_confusion_matrix(conf_mat, classes, name, STATE['dataset'])
                console.print(f'[dim]Saved matrix plot to {p}[/dim]')
            except:
                pass
        log_experiment('Model Benchmark', STATE['dataset'], name, float(acc), float(train_t), notes)
        console.print('[dim]Result logged to history.json[/dim]')

def cmd_test(args):
    is_sweep = args and args[0].lower() == 'sweep'
    if not is_sweep and not STATE['dataset']:
        console.print(f"[{get_color('warning')}]No dataset loaded. Let's pick one first...[/{get_color('warning')}]")
        cmd_dataset([])
        if not STATE['dataset']:
            return
    mode = 'full'
    if '--mode' in args:
        idx = args.index('--mode')
        if idx + 1 < len(args):
            mode = args[idx + 1]
            args.pop(idx + 1)
        args.pop(idx)
    if not args:
        choices = ['Compare (Benchmark models against each other)', 'Scaling (Test qubit scaling efficiency)', 'Noise (Simulate quantum noise resistance)', 'Ablation (Test Samyoga Legacy components)', 'Stats (Generate Statistical Report)', 'Quantum_advantage (Measure TNQE mathematical scaling)']
        test_choice = questionary.select('Select an experiment:', choices=choices, style=get_q_style()).ask()
        if not test_choice:
            return
        test_type = test_choice.split()[0].lower()
    else:
        test_type = args[0].lower()
    ds = STATE['dataset']
    ep = int(STATE['epochs'])
    if test_type in ['compare', 'sweep']:
        from experiments.run_comparison import run_comparison
        PROFILES = {'smoke': {'epochs': 1, 'samples': 30, 'qubits': 4, 'binary': True}, 'demo': {'epochs': 3, 'samples': 100, 'qubits': 4, 'binary': False}, 'fast': {'epochs': 3, 'samples': 100, 'qubits': 4, 'binary': False}, 'low_data': {'epochs': 10, 'samples': 200, 'qubits': 8, 'binary': False}, 'quantum_safe': {'epochs': 5, 'samples': 200, 'qubits': 4, 'binary': True}, 'research': {'epochs': 10, 'samples': 200, 'qubits': 8, 'binary': False}, 'large': {'epochs': ep, 'samples': 1000, 'qubits': 8, 'binary': False}, 'full': {'epochs': ep, 'samples': 999999, 'qubits': 8, 'binary': False}}
        config = PROFILES.get(mode, PROFILES['full'])
        if config['qubits'] > 8 or config['samples'] > 300 or config['epochs'] > 10:
            console.print('[warning]This run may take a long time.[/warning]')
            ans = questionary.confirm('Suggested quantum-safe config: 4 qubits, N <= 200, epochs <= 5. Continue anyway?', style=get_q_style()).ask()
            if not ans:
                console.print('Switching to quantum_safe mode...')
                mode = 'quantum_safe'
                config = PROFILES['quantum_safe']
        fair_choices = [
            "Let them loose (Raw Unbridled Execution)",
            "Fair Experiment (Control for Structural Overheads)"
        ]
        fair_choice = questionary.select(
            "Samyoga Pro detected in benchmark pool. Select Experiment Paradigm:",
            choices=fair_choices, style=get_q_style()
        ).ask()
        fair_mode = False
        if fair_choice and fair_choice.startswith("Let them"):
            console.print("[warning]WARNING: Samyoga Pro uses Mamba-2 and QMoE. Pitting it against classical SVMs is structurally unfair. It will dominate accuracy but fail efficiency.[/warning]")
        elif fair_choice:
            console.print("[success]Fair Mode Activated: Using Classical Shadow equivalent for parameter-fair comparison.[/success]")
            fair_mode = True
            if test_type == 'sweep' or ds != 'complexity_wall':
                console.print("[bold yellow]Recommendation: For a truly fair evaluation of quantum entanglement, test against the 'complexity_wall' dataset using /dataset.[/bold yellow]")

        from data.loader import DATASET_CONFIGS
        if test_type == 'sweep':
            choices = [questionary.Choice(f"{config.get('size_label', '')} {name}".strip(), value=name, checked=not config.get('is_large', False)) for name, config in DATASET_CONFIGS.items()]
            datasets_to_run = questionary.checkbox("Select datasets to include in the sweep (Space to toggle, Enter to confirm):", choices=choices, style=get_q_style()).ask()
            if not datasets_to_run:
                return
            target_name = f"SWEEP ({len(datasets_to_run)} DATASETS)"
        else:
            datasets_to_run = [ds]
            target_name = ds.upper()

        print_panel(f'Running Main Comparison on [bold]{target_name}[/bold] (Mode: {mode})...', style_key='primary')
        start_t = time.time()
        df = run_experiment_staged(run_comparison, {'datasets': datasets_to_run, 'mode': mode, 'fair_mode': fair_mode, 'config': config}, f'Benchmarking models...')
        end_t = time.time()

        if df is not None:
            total_t = end_t - start_t
            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            sess_path = f"results/metrics/sessions/{test_type}_{stamp}_{total_t:.1f}s.csv"
            os.makedirs(os.path.dirname(sess_path), exist_ok=True)
            df.to_csv(sess_path, index=False)

            df['Efficiency'] = df['Accuracy'] / df['Train_Time_s'].apply(lambda x: max(x, 1e-09))
            df = df.sort_values(by=['Dataset', 'Accuracy', 'Efficiency', 'Train_Time_s'], ascending=[True, False, False, True])

            from rich.table import Table
            summary_table = Table(title=f"Experiment Summary: {target_name} ({mode}) [Seed-Wise]", show_header=True, header_style=get_color('header'))
            summary_table.add_column("Dataset")
            summary_table.add_column("Seed")
            summary_table.add_column("Rank")
            summary_table.add_column("Model")
            summary_table.add_column("Accuracy", justify="right")
            summary_table.add_column("Time", justify="right")
            summary_table.add_column("Efficiency", justify="right")

            point_dist = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
            aggregate_points = {}

            for dataset_name, grp in df.groupby('Dataset', sort=False):
                for seed_val, s_grp in grp.groupby('Seed', sort=False):
                    first_in_ds = True
                    for i, (_, row) in enumerate(s_grp.iterrows()):
                        m_name = row['Model']
                        pts = point_dist[i] if i < len(point_dist) else 0
                        aggregate_points[m_name] = aggregate_points.get(m_name, 0) + pts

                        ds_str = f"[bold]{dataset_name.upper()}[/bold]" if first_in_ds else ""
                        seed_str = str(row.get('Seed', '—'))
                        rank_str = f"[bold #ffd700]#{i+1}[/]" if i == 0 else str(i+1)
                        model_c = model_color(m_name)
                        model_str = f"[bold {model_c}]{m_name}[/]"
                        acc_str = f"{row['Accuracy']:.2%}"
                        time_str = f"{row.get('Train_Time_s', 0):.2f}s"
                        eff_str = f"{row.get('Efficiency', 0):.4f}"

                        summary_table.add_row(ds_str, seed_str, rank_str, model_str, acc_str, time_str, eff_str)
                        first_in_ds = False
                    summary_table.add_section()

            console.print()
            console.print(summary_table)

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Seed' in numeric_cols: numeric_cols.remove('Seed')
            avg_df = df.groupby('Model')[numeric_cols].mean().reset_index()
            avg_df['Efficiency'] = avg_df['Accuracy'] / avg_df['Train_Time_s'].apply(lambda x: max(x, 1e-09))
            avg_df = avg_df.sort_values(by=['Accuracy', 'Efficiency'], ascending=[False, False])

            avg_table = Table(title=f"Combined Mathematical Averages (All Seeds)", show_header=True, header_style=get_color('secondary'))
            avg_table.add_column("Rank")
            avg_table.add_column("Model")
            avg_table.add_column("Avg Accuracy", justify="right")
            avg_table.add_column("Avg Time", justify="right")
            avg_table.add_column("Avg Efficiency", justify="right")
            for i, (_, row) in enumerate(avg_df.iterrows()):
                m_name = row['Model']
                rank_str = f"[bold #ffd700]#{i+1}[/]" if i == 0 else str(i+1)
                avg_table.add_row(rank_str, f"[bold {model_color(m_name)}]{m_name}[/]", f"{row['Accuracy']:.2%}", f"{row['Train_Time_s']:.2f}s", f"{row['Efficiency']:.4f}")

            console.print()
            console.print(avg_table)

            pts_df = pd.DataFrame(list(aggregate_points.items()), columns=['Model', 'Points'])
            pts_df = pts_df.sort_values(by='Points', ascending=False)

            pts_table = Table(title=f"Aggregate Championship Standings", show_header=True, header_style=get_color('accent'))
            pts_table.add_column("Rank")
            pts_table.add_column("Model")
            pts_table.add_column("Total Points", justify="right", style="bold yellow")
            for i, (_, row) in enumerate(pts_df.iterrows()):
                m_name = row['Model']
                rank_str = f"[bold #ffd700]#{i+1}[/]" if i == 0 else str(i+1)
                pts_table.add_row(rank_str, f"[bold {model_color(m_name)}]{m_name}[/]", str(row['Points']))

            console.print()
            console.print(pts_table)
            console.print()

            STATE['last_winner'] = pts_df.iloc[0]['Model'] if len(pts_df) > 0 else 'N/A'

            if test_type != 'sweep':
                print_winner_podium(avg_df.to_dict('records'))
                print_interpretation(avg_df, ds)
                print_categorical_comparisons(avg_df)

            console.print()
            console.print(df_to_table(df, f'FULL LEADERBOARD (ALL RUNS): {target_name.upper()}'))

            csv_path = f'results/{"all_datasets" if test_type == "sweep" else ds}_benchmark_latest.csv'
            df.to_csv(csv_path, index=False)
            console.print(f"[bold {get_color('success')}]✓ Leaderboard automatically exported to {csv_path} for your research paper.[/bold {get_color('success')}]")
            console.print()
            try:
                from ui.graphs import render_rich_metric_bars, save_all_comparison_charts, save_accuracy_vs_time_scatter
                console.print()

                sort_cols = [c for c in ['Dataset', 'Seed', 'Model'] if c in df.columns]
                if sort_cols:
                    df = df.sort_values(by=sort_cols)

                for m in ['Accuracy', 'F1_Macro', 'AUC_ROC', 'Train_Time_s', 'Efficiency']:
                    if m in df.columns:
                        ds_list = df['Dataset'].tolist() if 'Dataset' in df.columns else None
                        seed_list = df['Seed'].tolist() if 'Seed' in df.columns else None
                        render_rich_metric_bars(f'{m} Comparison', df['Model'].tolist(), df[m].tolist(), m, datasets=ds_list, seeds=seed_list)
                saved_paths = save_all_comparison_charts(df, target_name)
                scatter_path = save_accuracy_vs_time_scatter(df, target_name)
                if scatter_path:
                    saved_paths.append(scatter_path)
                if saved_paths:
                    console.print(f'[[dim]Saved {len(saved_paths)} charts to {os.path.dirname(saved_paths[0])}[/dim]]')
            except Exception as e:
                console.print(f'[dim error]Failed to render graphs: {e}[/]')
            for _, row in df.iterrows():
                ds_val = str(row.get('Dataset', ds))
                log_experiment('Test: Compare', ds_val, str(row['Model']), float(row['Accuracy']), float(row.get('Train_Time_s', 0)))
    elif test_type == 'scaling':
        from experiments.run_scaling import run_scaling
        print_panel(f'Running N-Scaling Hypothesis on [bold]{ds.upper()}[/bold]...', style_key='warning')
        df = run_experiment_staged(run_scaling, {'datasets': [ds], 'config': {'epochs': ep, 'qubits': 4, 'binary': False}}, f'Training across sizes...')
        if df is not None:
            console.print(df_to_table(df, 'N-Scaling Results', 'warning'))
            curve_dict = {}
            for model in df['Model'].unique():
                m_df = df[df['Model'] == model]
                curve_dict[model] = m_df['Accuracy'].tolist()
            x_vals = df[df['Model'] == df['Model'].iloc[0]]['Train_Size'].tolist()
            try:
                from ui.graphs import render_terminal_curve, save_scaling_curve
                render_terminal_curve(f'N-Scaling on {ds.upper()}', x_vals, curve_dict, 'Train Size (N)', 'Accuracy')
                p = save_scaling_curve(df, ds)
                if p:
                    console.print(f'[[dim]Saved curve to {p}[/dim]]')
            except Exception as e:
                console.print(f'[dim error]Failed to render graphs: {e}[/]')
    elif test_type == 'noise':
        from experiments.run_noise import run_noise
        print_panel(f'Running Noise Study on [bold]{ds.upper()}[/bold]...', style_key='error')
        df = run_experiment_staged(run_noise, {'datasets': [ds], 'config': {'epochs': ep, 'samples': 100, 'qubits': 4, 'binary': False}}, f'Simulating noise...')
        if df is not None:
            console.print(df_to_table(df, 'Noise Degradation Results', 'error'))
            noise_col = 'Noise' if 'Noise' in df.columns else 'Noise_Prob'
            x_vals = sorted(df[noise_col].unique().tolist()) if noise_col in df.columns else [0.0, 0.01, 0.05, 0.1, 0.2]
            curve_dict = {}
            for model in df['Model'].unique():
                m_df = df[df['Model'] == model]
                if not m_df.empty:
                    if noise_col in m_df.columns:
                        acc_map = dict(zip(m_df[noise_col], m_df['Accuracy']))
                        curve_dict[model] = [acc_map.get(x, np.nan) for x in x_vals]
                    else:
                        curve_dict[model] = m_df['Accuracy'].tolist()
            if curve_dict:
                try:
                    from ui.graphs import render_terminal_curve, save_noise_curve
                    render_terminal_curve(f'Noise Study on {ds.upper()}', x_vals, curve_dict, 'Noise (p)', 'Accuracy')
                    p = save_noise_curve(df, ds)
                    if p:
                        console.print(f'[[dim]Saved curve to {p}[/dim]]')
                except Exception as e:
                    console.print(f'[dim error]Failed to render graphs: {e}[/]')
    elif test_type == 'ablation':
        from experiments.run_ablation import run_ablation
        print_panel(f'Running Ablation Study on [bold]{ds.upper()}[/bold]...', style_key='warning')
        df = run_experiment_staged(run_ablation, {'datasets': [ds], 'config': {'epochs': ep, 'samples': 100, 'qubits': 4, 'binary': False}}, 'Testing Samyoga Legacy components...')
        if df is not None:
            console.print(df_to_table(df, 'Ablation Results', 'warning'))
    elif test_type == 'stats':
        from experiments.stats_engine import generate_statistics_report
        csv_path = 'results/metrics/model_comparison.csv'
        print_panel(f'Generating Statistics Report...', style_key='success')
        try:
            generate_statistics_report(csv_path)
        except Exception as e:
            console.print(f'[dim error]Stats generation failed: {e}[/]')
    elif test_type == 'quantum_advantage':
        from experiments.run_quantum_advantage import run_quantum_advantage
        run_quantum_advantage()
    else:
        print_error_panel('Unknown test', f"'{test_type}' is invalid.", 'Available: compare, scaling, noise, ablation, stats.', '/test compare')

def print_categorical_comparisons(df):
    from ui.graphs import render_rich_metric_bars
    from ui.theme import get_color

    categories = {
        "Legacy Shootout": df[df['Model'].str.lower().str.contains('legacy')].copy(),
        "Pro/Pro+ Heavyweights": df[df['Model'].str.lower().str.contains(r'pro(\+)?\b', regex=True) & ~df['Model'].str.lower().str.contains('legacy')].copy(),
        "Advanced Architectures": df[df['Model'].str.lower().str.contains(r'pro\+|go|shadow', regex=True)].copy()
    }

    for title, cat_df in categories.items():
        if cat_df.empty:
            continue

        cat_df = cat_df.sort_values(by=['Accuracy', 'Efficiency'], ascending=[False, False])

        console.print()
        console.print()

        from ui.theme import gradient_text
        header_txt = gradient_text("--- " + title + " ---", get_color('secondary').replace('bold ', ''), get_color('accent').replace('bold ', ''))
        from rich.align import Align
        from rich.text import Text
        console.print(Align.center(Text.from_markup(header_txt)))

        console.print()

        from ui.components import df_to_table
        console.print(df_to_table(cat_df, f'{title} Table'))

        console.print()

        metrics_to_plot = [('Accuracy', True), ('F1_Macro', True), ('AUC_ROC', True), ('Train_Time_s', False), ('Efficiency', True)]
        tables = []
        for m_name, hib in metrics_to_plot:
            if m_name in cat_df.columns:
                t = render_rich_metric_bars(
                    title=f"{title} ({m_name})",
                    labels=cat_df['Model'].tolist(),
                    values=cat_df[m_name].tolist(),
                    metric_name=m_name,
                    higher_is_better=hib,
                    compact_legend=True,
                    return_table_only=True
                )
                if t:
                    tables.append(t)

        if tables:
            from rich.columns import Columns
            from rich.align import Align

            if len(tables) >= 2:
                console.print(Align.center(Columns([tables[0], tables[1]], expand=False, padding=(0, 4))))
            elif len(tables) == 1:
                console.print(Align.center(tables[0]))

            console.print()

            if len(tables) >= 4:
                console.print(Align.center(Columns([tables[2], tables[3]], expand=False, padding=(0, 4))))
            elif len(tables) == 3:
                console.print(Align.center(tables[2]))

            console.print()

            if len(tables) == 5:
                console.print(Align.center(tables[4]))
                console.print()

            from ui.graphs import generate_horizontal_legend
            legend = generate_horizontal_legend(cat_df['Model'].tolist())
            console.print(Align.center(legend))
            console.print()


def cmd_results():
    import glob, questionary
    from ui.theme import get_q_style

    sess_dir = 'results/metrics/sessions'
    if not os.path.exists(sess_dir):
        print_error_panel('No sessions', f'{sess_dir} not found.', 'Run experiments first.', '/test compare')
        return

    session_files = sorted(glob.glob(f"{sess_dir}/*.csv"), reverse=True)
    if not session_files:
        print_error_panel('Empty history', f'No CSVs found in {sess_dir}.', 'Run experiments first.', '/test compare')
        return

    choices = []
    for fp in session_files:
        fname = os.path.basename(fp)
        parts = fname.replace('.csv', '').split('_')
        if len(parts) >= 4:
            ttype = parts[0]
            date_str = f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:]}"
            time_str = f"{parts[2][:2]}:{parts[2][2:4]}:{parts[2][4:]}"
            dur = parts[3]
            display = f"{ttype.upper()} - {date_str} {time_str} - {dur}"
            choices.append(questionary.Choice(title=display, value=fp))
        else:
            choices.append(questionary.Choice(title=fname, value=fp))

    selected_fp = questionary.select(
        "Select an experiment session to view detailed results (averaged across seeds):",
        choices=choices,
        style=get_q_style()
    ).ask()

    if not selected_fp:
        return

    df = pd.read_csv(selected_fp)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Seed' in numeric_cols:
        numeric_cols.remove('Seed')

    agg_df = df.groupby(['Dataset', 'Model'])[numeric_cols].mean().reset_index()
    agg_df['Efficiency'] = agg_df['Accuracy'] / agg_df['Train_Time_s'].apply(lambda x: max(x, 1e-09))
    agg_df = agg_df.sort_values(by=['Dataset', 'Accuracy', 'Efficiency', 'Train_Time_s'], ascending=[True, False, False, True])

    from ui.components import print_winner_podium, print_interpretation, df_to_table
    from ui.graphs import render_rich_metric_bars

    for ds_name, ds_df in agg_df.groupby('Dataset', sort=False):
        console.print(f"\n[bold {get_color('primary')}]Detailed Averaged Results for {ds_name.upper()}[/bold {get_color('primary')}]")
        console.print(df_to_table(ds_df, f'LEADERBOARD: {ds_name.upper()} (Averaged)'))
        print_winner_podium(ds_df.to_dict('records'))
        print_interpretation(ds_df, ds_name)
        print_categorical_comparisons(ds_df)


def cmd_history():
    if not os.path.exists(HISTORY_FILE):
        return
    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)
    df = pd.DataFrame(history)
    if df.empty:
        return
    console.print(df_to_table(df.tail(10), 'RECENT EXPERIMENTS', 'secondary'))

def cmd_report():
    if not os.path.exists(HISTORY_FILE):
        print_error_panel('No history', 'HISTORY_FILE not found.', 'Run experiments first.', '/test compare')
        return
    os.makedirs('docs/guides', exist_ok=True)
    report_path = 'docs/guides/automated_research_report.md'
    with console.status(f"[{get_color('primary')}]Generating Report..."):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
        df = pd.DataFrame(history)
        with open(report_path, 'w') as f:
            f.write('# QML Benchmark Research Report\n\n')
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write('## 1. Abstract\nThis report automatically aggregates the benchmarking results of the 200-Sample NISQ Crossover hypothesis comparing Classical, Pure Quantum, and Hybrid Transfer Learning models.\n\n')
    print_panel(f'Report successfully written to {report_path}', style_key='success')

def cmd_graphs(args=None):
    if not os.path.exists(HISTORY_FILE):
        print_error_panel('No history', 'HISTORY_FILE not found.', 'Run experiments first.', '/test compare')
        return
    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)
    df = pd.DataFrame(history)
    if df.empty:
        print_error_panel('Empty history', 'No data to plot.', 'Run experiments first.', '/test compare')
        return
    choice = None
    if args:
        action = args[0].lower()
        if action == 'list':
            choice = 'List saved figures'
        elif action == 'latest':
            choice = 'Show latest comparison bars'
        elif action == 'summary':
            choice = 'Show graph summary'
        else:
            print_error_panel('Unknown argument', f"'{action}' not valid.", 'Available: list, latest, summary', '/graphs list')
            return
    if not choice:
        choice = questionary.select('Select graph action:', choices=['Show latest comparison bars', 'Show accuracy trend', 'List saved figures', 'Show graph summary', 'Cancel'], style=get_q_style()).ask()
    if not choice or choice == 'Cancel':
        return
    from ui.graphs import render_rich_metric_bars, render_terminal_curve, render_terminal_scatter
    if choice == 'Show latest comparison bars':
        df_comp = df[df['type'] == 'Test: Compare'].copy()
        if df_comp.empty:
            console.print('[dim]No compare tests run yet.[/dim]')
            return

        df_comp['ts'] = pd.to_datetime(df_comp['timestamp'])
        df_comp = df_comp.sort_values('ts')


        df_comp['time_diff'] = df_comp['ts'].diff()
        run_ids = (df_comp['time_diff'] > pd.Timedelta(hours=2)).cumsum()
        latest_run_id = run_ids.iloc[-1]

        df_latest = df_comp[run_ids == latest_run_id]

        sort_cols = [c for c in ['dataset', 'seed', 'model'] if c in df_latest.columns]
        if sort_cols:
            df_latest = df_latest.sort_values(by=sort_cols)

        models = df_latest['model'].tolist()
        ds_list_latest = df_latest['dataset'].tolist() if 'dataset' in df_latest.columns else None
        seed_list_latest = df_latest['seed'].tolist() if 'seed' in df_latest.columns else None
        render_rich_metric_bars('Latest Accuracy Comparison', models, df_latest['accuracy'].tolist(), 'Accuracy', datasets=ds_list_latest, seeds=seed_list_latest)
    elif choice == 'Show accuracy trend':
        curve_dict = {}
        for m in df['model'].unique():
            accs = df[df['model'] == m]['accuracy'].tolist()
            if accs:
                curve_dict[m] = accs
        if curve_dict:
            render_terminal_curve('Terminal Line Graph: Accuracy Trend', None, curve_dict, 'Experiment #', 'Accuracy')
    elif choice == 'List saved figures':
        figs_dir = 'results/figures'
        if os.path.exists(figs_dir):
            files = os.listdir(figs_dir)
            if files:
                console.print(f"[bold {get_color('primary')}]Saved Figures in {figs_dir}:[/bold {get_color('primary')}]")
                for f in files:
                    console.print(f'  - {f}')
            else:
                console.print('[dim]No saved figures found.[/dim]')
        else:
            console.print('[dim]No saved figures directory found.[/dim]')
    elif choice == 'Show graph summary':
        grouped = df.groupby('model').mean(numeric_only=True).reset_index()
        models = grouped['model'].tolist()
        render_rich_metric_bars('Average Accuracy Summary', models, grouped['accuracy'].tolist(), 'Accuracy')
        render_rich_metric_bars('Average Train Time Summary', models, grouped['time_s'].tolist(), 'Time (s)', higher_is_better=False)
        curve_dict = {}
        for m in df['model'].unique():
            accs = df[df['model'] == m]['accuracy'].tolist()
            if accs:
                curve_dict[m] = accs
        if curve_dict:
            render_terminal_curve('Accuracy Trend', None, curve_dict, 'Experiment #', 'Accuracy')

def cmd_theme(args):
    if not args:
        from ui.theme_selector import interactive_theme_grid
        name = interactive_theme_grid(list(THEMES.keys()), STATE.get('theme'), THEMES)
        if not name:
            return
    else:
        name = args[0].lower()
        if name not in THEMES:
            print_error_panel('Unknown theme', f"'{name}' is invalid.", f'Available: {list(THEMES.keys())}', '/theme')
            return
    STATE['theme'] = name
    print_panel(f'Theme changed to [bold]{name}[/bold]', style_key='success')
    restart = questionary.confirm('Would you like to restart Darshan to fully apply this theme?', style=get_q_style()).ask()
    if restart:
        console.clear()
        boot_sequence()

def cmd_working(args):
    action = args[0].lower() if args else 'show'
    if action == 'on':
        STATE['working_mode'] = True
        print_panel('Working mode ON. Raw logs will be shown.', style_key='warning')
    elif action == 'off':
        STATE['working_mode'] = False
        print_panel('Working mode OFF. Interface clean.', style_key='success')
    elif action == 'show':
        console.print(Panel(STATE['working_log'][-2000:] if STATE['working_log'] else 'No logs captured yet.', title='Working Logs (Last 2000 chars)', border_style='dim'))
    elif action == 'clear':
        STATE['working_log'] = ''
        print_panel('Working logs cleared.', style_key='success')

def cmd_quiet(args):
    action = args[0].lower() if args else 'on'
    if action == 'on':
        STATE['quiet_mode'] = True
        print_panel('Quiet mode ON.', style_key='dim')
    else:
        STATE['quiet_mode'] = False
        print_panel('Quiet mode OFF.', style_key='success')

def cmd_menu():
    choice = questionary.select('What would you like to do?', choices=['Select Dataset', 'Inspect Model', 'Run Experiment', 'View Results Dashboard', 'View History', 'Generate Report', 'Change Theme', 'Toggle Working Logs', 'Toggle Quiet Mode', 'Reset Data', 'Exit'], style=get_q_style()).ask()
    if choice == 'Select Dataset':
        cmd_dataset([])
    elif choice == 'Inspect Model':
        cmd_model([])
    elif choice == 'Run Experiment':
        cmd_test([])
    elif choice == 'View Results Dashboard':
        cmd_results()
    elif choice == 'View History':
        cmd_history()
    elif choice == 'Generate Report':
        cmd_report()
    elif choice == 'Change Theme':
        cmd_theme([])
    elif choice == 'Toggle Working Logs':
        cmd_working(['on' if not STATE['working_mode'] else 'off'])
    elif choice == 'Toggle Quiet Mode':
        cmd_quiet(['on' if not STATE['quiet_mode'] else 'off'])
    elif choice == 'Reset Data':
        cmd_reset()
    elif choice == 'Exit':
        sys.exit(0)

def cmd_reset():
    import shutil
    import glob
    from ui.components import print_panel, print_error_panel
    from ui.theme import get_q_style

    confirm = questionary.text("Type RESET to confirm deletion of all data: ", style=get_q_style()).ask()
    if confirm != 'RESET':
        print_panel("Reset cancelled.", style_key="warning")
        return

    try:
        deleted_pycache = 0
        for root, dirs, files in os.walk('.', topdown=False):
            for d in dirs:
                if d == '__pycache__':
                    shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                    deleted_pycache += 1
        
        deleted_files = 0
        def clear_dir(path):
            nonlocal deleted_files
            if os.path.exists(path):
                for f in os.listdir(path):
                    fp = os.path.join(path, f)
                    if os.path.isfile(fp):
                        try:
                            os.remove(fp)
                            deleted_files += 1
                        except Exception:
                            pass

        clear_dir('results')
        clear_dir('results/metrics')
        clear_dir('results/metrics/sessions')
        clear_dir('results/figures')
        clear_dir('results/cache')
        clear_dir('results/checkpoints')
        
        os.makedirs('results/metrics/sessions', exist_ok=True)
        os.makedirs('results/figures', exist_ok=True)
        os.makedirs('results/cache', exist_ok=True)
        os.makedirs('results/checkpoints', exist_ok=True)
        
        STATE['dataset'] = None
        STATE['epochs'] = 10
        STATE['theme'] = 'solar'
        STATE['working_mode'] = False
        STATE['quiet_mode'] = False
        STATE['last_run'] = None
        STATE['last_winner'] = None
        STATE['working_log'] = ''
        STATE['last_metrics_path'] = None
        STATE['last_figures_path'] = None
        
        ACTIVE_MODELS.clear()
        
        summary = f"System reset complete.\n- {deleted_pycache} __pycache__ folders removed.\n- {deleted_files} data files (metrics, graphs, caches) deleted."
        print_panel(summary, title="FACTORY RESET", style_key="success")
    except Exception as e:
        print_error_panel("Failed to reset system.", str(e), "Check file permissions.")

def cmd_about():
    from rich.table import Table
    from rich.padding import Padding
    from rich.text import Text
    from ui.components import console
    from rich.rule import Rule
    c_primary = get_color('primary')
    c_secondary = get_color('secondary')
    c_accent = get_color('accent')
    title = Text('Darshan: QML Benchmarking Suite v2.0', style=c_primary, justify='center')
    intro = Text(justify='center')
    intro.append('This application is engineered to rigorously investigate the ')
    intro.append('"200-Sample Crossover" hypothesis', style='bold white')
    intro.append(' within the field of Quantum Machine Learning (QML).\n\n')
    intro.append('During the current Noisy Intermediate-Scale Quantum (NISQ) era, severe hardware constraints—such as rapid decoherence and limited qubit connectivity—restrict the viable depth and width of executing quantum circuits. To circumvent these physical bottlenecks, our research focuses on discovering practical regimes where quantum advantage might already be accessible.', style='dim')
    grid = Table(box=None, padding=(1, 2), show_header=False)
    grid.add_column('Concept', justify='right', style=c_accent, ratio=1)
    grid.add_column('Description', style='white', ratio=3)
    grid.add_row('Low-Data Regime Advantage', 'We hypothesize that when training data is severely limited (N < 200), Quantum Transfer Learning can capture higher-dimensional correlations and outperform both purely classical SVMs and purely quantum classifiers.')
    grid.add_row('Kernel-VQC Duality', 'The suite evaluates the intrinsic duality between Variational Quantum Classifiers (VQCs) and Quantum Kernel methods, demonstrating that architectural distinctions are secondary to optimization landscapes.')
    grid.add_row('Shadow QML Deployment', 'Darshan explores deployment strategies where quantum circuits learn complex embeddings during training, but actual inference is performed classically to reduce commercial scalability barriers.')
    grid.add_row('Hardware-Compatible Encoding', 'The suite aggressively benchmarks angle-encoding and PCA-reduced feature strategies that fit current physical limits, while systematically validating model robustness against simulated depolarizing noise.')
    console.print()
    console.print(Rule(f'[{c_primary}]ABOUT DARSHAN[/{c_primary}]', style='dim grey'))
    console.print(Padding(title, (1, 0, 0, 0)))
    console.print(Padding(intro, (1, 4)))
    console.print(Rule(f'[{c_secondary}]Core Research Theses[/{c_secondary}]', style='dim grey', characters=' '))
    console.print(Padding(grid, (0, 2)))
    creator = Text('Created by: Aakar Gupta', style='dim italic', justify='right')
    console.print(Padding(creator, (0, 4, 1, 0)))
    console.print(Rule(style='dim grey'))

def boot_sequence():
    os.system('cls' if os.name == 'nt' else 'clear')
    from ui.components import animate_gradient_logo, print_startup_tips
    animate_gradient_logo()
    print_startup_tips()
    console.print()

def repl():
    boot_sequence()
    from prompt_toolkit.completion import NestedCompleter
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.history import InMemoryHistory

    datasets = {k: None for k in sorted(['Iris', 'Wine', 'Moons', 'Breast Cancer', 'Digits', 'Pendigits', 'Complexity Wall'])}
    models = {k: None for k in sorted(['Samyoga Go', 'Samyoga Shadow', 'Samyoga Pro', 'Samyoga Legacy', 'Parampara Legacy', 'Parampara Pro', 'Parampara Pro+', 'Ananta Legacy', 'Ananta Pro'])}
    themes = {k.capitalize(): None for k in sorted(THEMES.keys())}

    command_dict = {
        '/about': None,
        '/checkpoint': {'save': models, 'load': models},
        '/clear': None,
        '/dataset': datasets,
        '/epochs': None,
        '/exit': None,
        '/graphs': {'clear': None, 'show': None},
        '/help': None,
        '/history': None,
        '/menu': None,
        '/model': models,
        '/quiet': {'on': None, 'off': None},
        '/report': None,
        '/reset': None,
        '/restart': None,
        '/results': None,
        '/sweep': models,
        '/test': {'compare': None, 'sweep': None, 'scaling': None, 'noise': None},
        '/theme': themes,
        '/train': {'continue': models},
        '/working': {'on': None, 'off': None, 'show': None, 'clear': None},
    }

    from prompt_toolkit.completion import Completer, Completion

    class DarshanCompleter(Completer):
        def __init__(self, command_dict):
            self.command_dict = command_dict

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor.lstrip()

            if ' ' not in text:
                word = text
                for opt in sorted(self.command_dict.keys()):
                    if opt.lower().startswith(word.lower()):
                        yield self._make_completion(opt, word)
                return

            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            if cmd not in self.command_dict or not isinstance(self.command_dict[cmd], dict):
                return

            node = self.command_dict[cmd]
            args_text = parts[1] if len(parts) > 1 else ""

            matched_subcmd = None
            for k in node.keys():
                if args_text.lower().startswith(k.lower() + ' '):
                    matched_subcmd = k
                    break

            if matched_subcmd:
                node = node[matched_subcmd]
                if not isinstance(node, dict): return
                args_text = args_text[len(matched_subcmd):].lstrip()

            word = args_text
            for opt in sorted(node.keys()):
                if opt.lower().startswith(word.lower()):
                    yield self._make_completion(opt, word)

        def _make_completion(self, option, typed_word):
            c1 = get_color('primary').replace('bold ', '').replace('dim ', '')
            c2 = get_color('secondary').replace('bold ', '').replace('dim ', '')

            def hex_to_rgb(hex_str):
                hex_str = hex_str.lstrip('#')
                if len(hex_str) == 6:
                    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
                return (255, 255, 255)

            r1, g1, b1 = hex_to_rgb(c1)
            r2, g2, b2 = hex_to_rgb(c2)

            length = len(option)
            fragments = []
            for i, char in enumerate(option):
                ratio = i / max(1, (length - 1))
                r = int(r1 + (r2 - r1) * ratio)
                g = int(g1 + (g2 - g1) * ratio)
                b = int(b1 + (b2 - b1) * ratio)
                hex_color = f'#{r:02x}{g:02x}{b:02x}'
                fragments.append((f'fg:{hex_color}', char))

            display = FormattedText(fragments)
            return Completion(option, start_position=-len(typed_word), display=display)

    completer = DarshanCompleter(command_dict)
    session = PromptSession(completer=completer, history=InMemoryHistory(), complete_while_typing=True)
    STATE['theme'] = 'solar'

    def get_dark_blend(hex_color, factor=0.2):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            r, g, b = int(r * factor), int(g * factor), int(b * factor)
            return f'#{r:02x}{g:02x}{b:02x}'
        return '#222222'

    while True:
        try:
            p_color = get_color('primary').replace('bold ', '').replace('dim ', '')
            s_color = get_color('secondary').replace('bold ', '').replace('dim ', '')
            m_color = get_color('muted').replace('bold ', '').replace('dim ', '')

            dark_bg = get_dark_blend(p_color, 0.25)

            status_text = apply_theme_to_prompt()
            console.print(f'[dim {m_color}] ╭{status_text}╮[/dim {m_color}]')

            style = Style.from_dict({
                'completion-menu.completion': f'bg:#141414',
                'completion-menu.completion.current': f'bg:{dark_bg} bold',
                'scrollbar.background': f'bg:#141414',
                'scrollbar.button': f'bg:{p_color}',
                'auto-suggestion': f'{m_color} italic',
                'prompt': p_color
            })

            prompt_fragments = FormattedText([('class:prompt', ' ❯ ')])
            text = session.prompt(prompt_fragments, style=style, auto_suggest=AutoSuggestFromHistory())
            text = text.strip()
            if not text:
                continue

            mapping = {
                'samyoga go': 'samyoga_go',
                'samyoga shadow': 'samyoga_shadow',
                'samyoga pro': 'samyoga_pro',
                'samyoga legacy': 'samyoga_legacy',
                'parampara legacy': 'parampara_legacy',
                'parampara pro+': 'parampara_pro_industry',
                'parampara pro': 'parampara',
                'ananta legacy': 'ananta',
                'ananta pro': 'ananta_pro',
                'breast cancer': 'breast_cancer',
                'complexity wall': 'complexity_wall'
            }
            import re
            for display, internal in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
                text = re.sub(re.escape(display), internal, text, flags=re.IGNORECASE)

            parts = text.split()
            cmd = parts[0].lower()
            args = parts[1:]
            if cmd == '/exit':
                console.print(f"[{get_color('success')}]Closing Darshan... Goodbye!")
                break
            elif cmd == '/restart':
                STATE['dataset'] = None
                boot_sequence()
                continue
            elif cmd == '/reset':
                cmd_reset()
            elif cmd == '/clear':
                console.clear()
            elif cmd == '/help':
                print_help()
            elif cmd == '/about':
                cmd_about()
            elif cmd == '/dataset':
                cmd_dataset(args)
            elif cmd == '/model':
                cmd_model(args)
            elif cmd == '/train':
                cmd_train(args)
            elif cmd == '/checkpoint':
                cmd_checkpoint(args)
            elif cmd == '/test':
                cmd_test(args)
            elif cmd == '/sweep':
                cmd_test(['sweep'] + args)
            elif cmd == '/results':
                cmd_results()
            elif cmd == '/history':
                cmd_history()
            elif cmd == '/report':
                cmd_report()
            elif cmd == '/theme':
                cmd_theme(args)
            elif cmd == '/menu':
                cmd_menu()
            elif cmd == '/working':
                cmd_working(args)
            elif cmd == '/quiet':
                cmd_quiet(args)
            elif cmd == '/graphs':
                cmd_graphs(args)
            elif cmd == '/epochs':
                if args and args[0].isdigit():
                    STATE['epochs'] = int(args[0])
                    print_panel(f"Epochs set to {STATE['epochs']}", style_key='success')
                else:
                    print_error_panel('Invalid epochs', 'Must be integer.', 'Provide a number.', '/epochs 20')
            else:
                print_error_panel('Unknown command', f"'{cmd}' is invalid.", 'Check spelling.', '/help')
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
if __name__ == '__main__':
    repl()