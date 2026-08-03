import json
import os
STATE = {'dataset': None, 'epochs': 10, 'theme': 'solar', 'working_mode': False, 'quiet_mode': False, 'last_run': None, 'last_winner': None, 'working_log': '', 'last_metrics_path': None, 'last_figures_path': None}
THEMES = {
    'neon': {'primary': 'bold #FF107A', 'secondary': 'bold #00E5FF', 'accent': 'bold #7000FF', 'success': 'bold #00FF9D', 'warning': 'bold #FFB300', 'error': 'bold #FF0055', 'muted': 'dim #6A5ACD', 'border': '#FF107A', 'header': 'bold #00E5FF'},
    'cyber': {'primary': 'bold #E6FF00', 'secondary': 'bold #FF00E6', 'accent': 'bold #00FFFF', 'success': 'bold #39FF14', 'warning': 'bold #FF6600', 'error': 'bold #FF003C', 'muted': 'dim #555555', 'border': '#E6FF00', 'header': 'bold #FF00E6'},
    'matrix': {'primary': 'bold #00FF41', 'secondary': 'bold #008F11', 'accent': 'bold #003B00', 'success': 'bold #00FF41', 'warning': 'bold #39FF14', 'error': 'bold #008F11', 'muted': 'dim #003B00', 'border': '#00FF41', 'header': 'bold #008F11'},
    'aurora': {'primary': 'bold #00FA9A', 'secondary': 'bold #9370DB', 'accent': 'bold #87CEFA', 'success': 'bold #00FF7F', 'warning': 'bold #FFD700', 'error': 'bold #FF69B4', 'muted': 'dim #4682B4', 'border': '#00FA9A', 'header': 'bold #9370DB'},
    'royal': {'primary': 'bold #FFD700', 'secondary': 'bold #4B0082', 'accent': 'bold #8A2BE2', 'success': 'bold #32CD32', 'warning': 'bold #FFA500', 'error': 'bold #B22222', 'muted': 'dim #483D8B', 'border': '#FFD700', 'header': 'bold #4B0082'},
    'ocean': {'primary': 'bold #0077BE', 'secondary': 'bold #00A86B', 'accent': 'bold #20B2AA', 'success': 'bold #00FF7F', 'warning': 'bold #F0E68C', 'error': 'bold #FF6347', 'muted': 'dim #5F9EA0', 'border': '#0077BE', 'header': 'bold #00A86B'},
    'forest': {'primary': 'bold #8FBC8F', 'secondary': 'bold #556B2F', 'accent': 'bold #DEB887', 'success': 'bold #228B22', 'warning': 'bold #DAA520', 'error': 'bold #8B0000', 'muted': 'dim #6B8E23', 'border': '#8FBC8F', 'header': 'bold #556B2F'},
    'ember': {'primary': 'bold #FF4500', 'secondary': 'bold #FF8C00', 'accent': 'bold #FFD700', 'success': 'bold #7CFC00', 'warning': 'bold #FFA07A', 'error': 'bold #DC143C', 'muted': 'dim #A0522D', 'border': '#FF4500', 'header': 'bold #FF8C00'},
    'glacier': {'primary': 'bold #F0F8FF', 'secondary': 'bold #ADD8E6', 'accent': 'bold #B0E0E6', 'success': 'bold #00FA9A', 'warning': 'bold #FFFACD', 'error': 'bold #FFB6C1', 'muted': 'dim #708090', 'border': '#F0F8FF', 'header': 'bold #ADD8E6'},
    'monochrome': {'primary': 'bold #FFFFFF', 'secondary': 'bold #A9A9A9', 'accent': 'bold #D3D3D3', 'success': 'bold #E6E6E6', 'warning': 'bold #808080', 'error': 'bold #696969', 'muted': 'dim #777777', 'border': '#FFFFFF', 'header': 'bold #A9A9A9'},
    'amethyst': {'primary': 'bold #DA70D6', 'secondary': 'bold #800080', 'accent': 'bold #DDA0DD', 'success': 'bold #3CB371', 'warning': 'bold #F4A460', 'error': 'bold #C71585', 'muted': 'dim #9370DB', 'border': '#DA70D6', 'header': 'bold #800080'},
    'crimson': {'primary': 'bold #DC143C', 'secondary': 'bold #800000', 'accent': 'bold #B22222', 'success': 'bold #2E8B57', 'warning': 'bold #D2691E', 'error': 'bold #FF0000', 'muted': 'dim #8B0000', 'border': '#DC143C', 'header': 'bold #800000'},
    'quantum': {'primary': 'bold #C0C0C0', 'secondary': 'bold #0055FF', 'accent': 'bold #00FFFF', 'success': 'bold #00FF00', 'warning': 'bold #FFA500', 'error': 'bold #FF0000', 'muted': 'dim #808080', 'border': '#C0C0C0', 'header': 'bold #0055FF'},
    'paper': {'primary': 'bold #F5F5DC', 'secondary': 'bold #8B4513', 'accent': 'bold #CD853F', 'success': 'bold #556B2F', 'warning': 'bold #B8860B', 'error': 'bold #A52A2A', 'muted': 'dim #D2B48C', 'border': '#F5F5DC', 'header': 'bold #8B4513'},
    'solar': {'primary': 'bold #ffdd00', 'secondary': 'bold #ff7700', 'accent': 'bold #ff0066', 'success': 'bold #a2ff00', 'warning': 'bold #ffaa00', 'error': 'bold #cc0000', 'muted': 'dim #aa8855', 'border': '#ffdd00', 'header': 'bold #ff7700'},
    'gemini': {'primary': 'bold #ff33a1', 'secondary': 'bold #c27cff', 'accent': 'bold #33d1ff', 'success': 'bold #33ffcc', 'warning': 'bold #ffcc00', 'error': 'bold #ff0055', 'muted': 'dim #aaaaaa', 'border': '#ff33a1', 'header': 'bold #c27cff'},
    'accessible': {'primary': 'bold #FFFF00', 'secondary': 'bold #00FFFF', 'accent': 'bold #FF00FF', 'success': 'bold #00FF00', 'warning': 'bold #FFA500', 'error': 'bold #FF0000', 'muted': 'dim #AAAAAA', 'border': '#FFFF00', 'header': 'bold #00FFFF'},
    'vibgyor': {'primary': 'bold #FF007F', 'secondary': 'bold #00E5FF', 'accent': 'bold #FFEA00', 'success': 'bold #39FF14', 'warning': 'bold #FF6600', 'error': 'bold #FF0033', 'muted': 'dim #9D00FF', 'border': '#FF007F', 'header': 'bold #00E5FF'}
}

def get_model_color(model_name: str) -> str:
    name = model_name.lower()

    if 'ananta legacy' in name or name == 'ananta':
        return '#8B4513'
    if 'parampara legacy' in name:
        return '#D2B48C'
    if 'samyoga legacy' in name:
        return '#CD7F32'

    if 'ananta' in name:
        base_hex = get_color('accent').replace('bold ', '').replace('dim ', '')
    elif 'parampara' in name:
        base_hex = get_color('secondary').replace('bold ', '').replace('dim ', '')
    elif 'samyoga' in name:
        base_hex = get_color('primary').replace('bold ', '').replace('dim ', '')
    elif 'advantage' in name or 'multiplier' in name or 'quantum' in name:
        return '#A2FF00'
    elif 'complexity' in name or 'sim' in name:
        return '#CC0000'
    elif 'features' in name or 'classical' in name:
        return '#0077FF'
    else:
        palette = [get_color(k).replace('bold ', '').replace('dim ', '') for k in ['primary', 'secondary', 'accent', 'success', 'warning', 'error']]
        base_hex = palette[abs(hash(name)) % len(palette)]

    def adjust_brightness(hex_color, factor):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            r = min(255, max(0, int(r * factor)))
            g = min(255, max(0, int(g * factor)))
            b = min(255, max(0, int(b * factor)))
            return f'#{r:02x}{g:02x}{b:02x}'
        return hex_color

    if 'pro+' in name or 'industry' in name or 'shadow' in name:
        return adjust_brightness(base_hex, 1.4)
    elif 'go' in name or 'fast' in name or 'mock' in name:
        return adjust_brightness(base_hex, 1.8)
    else:
        return base_hex

def get_q_style():
    import prompt_toolkit
    p = get_color('primary').replace('bold ', '').replace('dim ', '').split(' ')[-1]
    s = get_color('secondary').replace('bold ', '').replace('dim ', '').split(' ')[-1]
    a = get_color('accent').replace('bold ', '').replace('dim ', '').split(' ')[-1]
    m = get_color('muted').replace('bold ', '').replace('dim ', '').split(' ')[-1]
    return prompt_toolkit.styles.Style([('qmark', f'fg:{p} bold'), ('question', f'bold fg:{p}'), ('answer', f'fg:{a} bold'), ('pointer', f'fg:{s} bold'), ('highlighted', f'fg:{p} bold'), ('selected', f'fg:{s}'), ('separator', f'fg:{m}'), ('instruction', f'fg:{m} italic'), ('text', '')])

def get_color(key):
    theme_name = STATE.get('theme', 'neon')
    if theme_name not in THEMES:
        theme_name = 'neon'
    return THEMES[theme_name].get(key, 'white')

def model_color(model_name):
    return get_model_color(model_name)

def apply_theme_to_prompt():
    """Generate a compact status bar string showing current Darshan state."""
    ds = STATE['dataset'].upper() if STATE['dataset'] else '—'
    ep = STATE['epochs']
    t = STATE['theme']
    quiet = 'on' if STATE.get('quiet_mode') else 'off'
    return f' {ds} │ {ep}ep │ {t} │ quiet: {quiet} '

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple((int(hex_code[i:i + 2], 16) for i in (0, 2, 4)))

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*[int(c) for c in rgb])

def gradient_text(text, start_hex, end_hex):
    start_rgb = hex_to_rgb(start_hex)
    end_rgb = hex_to_rgb(end_hex)
    lines = text.split('\n')
    max_len = max((len(l) for l in lines), default=1)
    if max_len <= 1:
        max_len = 2
    out_lines = []
    for line in lines:
        out_line = ''
        for i, char in enumerate(line):
            ratio = i / (max_len - 1)
            r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio
            g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio
            b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio
            color = rgb_to_hex((r, g, b))
            out_line += f'[{color}]{char}[/]'
        out_lines.append(out_line)
    return '\n'.join(out_lines)

RAW_LOGOS = {
    'samyoga_legacy': ' ___   _   __  ____   _____   ___   _   \n/ __| /_\\ |  \\/  \\ \\ / / _ \\ / __| /_\\  \n\\__ \\/ _ \\| |\\/| |\\ V / (_) | (_ |/ _ \\  LEGACY\n|___/_/ \\_\\_|  |_| | | \\___/ \\___/_/ \\_\\\n\n        Legacy Hybrid QTL + SVM         ',
    'samyoga_shadow': ' ___   _   __  ____   _____   ___   _   \n/ __| /_\\ |  \\/  \\ \\ / / _ \\ / __| /_\\  \n\\__ \\/ _ \\| |\\/| |\\ V / (_) | (_ |/ _ \\  SHADOW\n|___/_/ \\_\\_|  |_| | | \\___/ \\___/_/ \\_\\\n\n      Trainable Classical Twin Mock      ',
    'samyoga_go': ' ___   _   __  ____   _____   ___   _   \n/ __| /_\\ |  \\/  \\ \\ / / _ \\ / __| /_\\  \n\\__ \\/ _ \\| |\\/| |\\ V / (_) | (_ |/ _ \\  GO\n|___/_/ \\_\\_|  |_| | | \\___/ \\___/_/ \\_\\\n\n      Instant Classical Mock Engine      ',
    'samyoga_pro': ' ___   _   __  ____   _____   ___   _   \n/ __| /_\\ |  \\/  \\ \\ / / _ \\ / __| /_\\  \n\\__ \\/ _ \\| |\\/| |\\ V / (_) | (_ |/ _ \\  PRO\n|___/_/ \\_\\_|  |_| | | \\___/ \\___/_/ \\_\\\n\nEnterprise-Grade Quantum Transfer Learning',
    'parampara_legacy': ' ___  _   ___    _   __  __ ___  _   ___    _   \n| _ \\/_\\ | _ \\  /_\\ |  \\/  | _ \\/_\\ | _ \\  /_\\  \n|  _/ _ \\|   / / _ \\| |\\/| |  _/ _ \\|   / / _ \\  LEGACY\n|_|/_/ \\_\\_|_\\/_/ \\_\\_|  |_|_|/_/ \\_\\_|_\\/_/ \\_\\\n\n         Untuned Classical RBF Strawman         ',
    'parampara_pro_industry': ' ___  _   ___    _   __  __ ___  _   ___    _   \n| _ \\/_\\ | _ \\  /_\\ |  \\/  | _ \\/_\\ | _ \\  /_\\  \n|  _/ _ \\|   / / _ \\| |\\/| |  _/ _ \\|   / / _ \\  PRO+\n|_|/_/ \\_\\_|_\\/_/ \\_\\_|  |_|_|/_/ \\_\\_|_\\/_/ \\_\\\n\n      Rigorous Classical Industry Champion      ',
    'parampara': ' ___  _   ___    _   __  __ ___  _   ___    _   \n| _ \\/_\\ | _ \\  /_\\ |  \\/  | _ \\/_\\ | _ \\  /_\\  \n|  _/ _ \\|   / / _ \\| |\\/| |  _/ _ \\|   / / _ \\  PRO\n|_|/_/ \\_\\_|_\\/_/ \\_\\_|  |_|_|/_/ \\_\\_|_\\/_/ \\_\\\n\n        Fair-Track PCA Bounded Champion         ',
    'ananta': '   _   _  _   _   _  _ _____ _   \n  /_\\ | \\| | /_\\ | \\| |_   _/_\\  \n / _ \\| .` |/ _ \\| .` | | |/ _ \\  LEGACY\n/_/ \\_\\_|\\_/_/ \\_\\_|\\_| |_/_/ \\_\\\n\n    Uncalibrated VQC Strawman    ',
    'ananta_pro': '   _   _  _   _   _  _ _____ _   \n  /_\\ | \\| | /_\\ | \\| |_   _/_\\  \n / _ \\| .` |/ _ \\| .` | | |/ _ \\  PRO\n/_/ \\_\\_|\\_/_/ \\_\\_|\\_| |_/_/ \\_\\\n\n Enterprise-Grade Hybrid Extractor ',
}

def get_model_logo(name):
    from rich.text import Text as RichText
    c1 = get_color('primary').replace('bold ', '').replace('dim ', '')
    c2 = get_color('secondary').replace('bold ', '').replace('dim ', '')
    c3 = get_color('accent').replace('bold ', '').replace('dim ', '')

    if 'legacy' in name or name == 'ananta':
        start_hex, end_hex = ('#CD7F32', '#8C5622')
    elif 'samyoga' in name:
        start_hex, end_hex = (c1, c2)
    elif 'parampara' in name:
        start_hex, end_hex = (c2, c3)
    elif 'ananta' in name:
        start_hex, end_hex = (c3, c1)
    else:
        start_hex, end_hex = (c1, c2)

    raw = RAW_LOGOS.get(name, name)
    lines = raw.split('\n')
    max_len = max((len(l) for l in lines), default=1)
    if max_len <= 1:
        max_len = 2
    start_rgb = hex_to_rgb(start_hex)
    end_rgb = hex_to_rgb(end_hex)
    result = RichText()
    for li, line in enumerate(lines):
        for i, char in enumerate(line):
            ratio = i / (max_len - 1) if max_len > 1 else 0
            r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio
            g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio
            b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio
            color = rgb_to_hex((r, g, b))
            result.append(char, style=f'bold {color}')
        if li < len(lines) - 1:
            result.append('\n')
    return result
import time
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.console import Group
_anim_console = Console()

def animate_panel(text, title, border_style, delay=0.005, header=None, footer=None):
    from rich.align import Align
    from rich.text import Text
    current_text = ''
    initial_items = []
    if header:
        initial_items.append(header)
    initial_items.append(Align.center(Text('')))
    if footer:
        initial_items.append(footer)
    with Live(Panel(Group(*initial_items), title=title, border_style=border_style), refresh_per_second=60, console=_anim_console) as live:
        for char in text:
            current_text += char
            items = []
            if header:
                items.append(header)
            items.append(Align.center(Text(current_text, justify='center')))
            if footer:
                items.append(footer)
            live.update(Panel(Group(*items), title=title, border_style=border_style))
            time.sleep(delay)
LOGO = '\n██       ████████     ███    ████████   ██████  ██     ██    ███    ██    ██\n ██      ██░░░░░██   ██░██   ██░░░░░██ ██░░░░██ ██░    ██░  ██░██   ███   ██░\n  ██     ██░    ██░ ██░░ ██  ██░    ██░██░    ░░██░    ██░ ██░░ ██  ████  ██░\n   ██    ██░    ██░██░░   ██ ████████░░ ██████  █████████░██░░   ██ ██░██ ██░\n  ██░░   ██░    ██░█████████░██░░░██░░   ░░░░██ ██░░░░░██░█████████░██░ ████░\n ██░░    ██░    ██░██░░░░░██░██░   ██  ██    ██░██░    ██░██░░░░░██░██░  ███░\n██░░     ████████░░██░    ██░██░    ██  ██████░░██░    ██░██░    ██░██░   ██░\n ░░       ░░░░░░░░  ░░     ░░ ░░     ░░  ░░░░░░  ░░     ░░ ░░     ░░ ░░    ░░\n'