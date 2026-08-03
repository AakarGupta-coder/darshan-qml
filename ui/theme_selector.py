import math

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl


def interactive_theme_grid(themes, current_theme, themes_dict):
    current_idx = 0
    if current_theme in themes:
        current_idx = themes.index(current_theme)
    rows = 3
    cols = math.ceil(len(themes) / rows)
    kb = KeyBindings()

    @kb.add("c-c")
    def _(event):
        event.app.exit(result=None)

    @kb.add("escape")
    def _(event):
        event.app.exit(result=None)

    @kb.add("enter")
    def _(event):
        event.app.exit(result=themes[current_idx])

    @kb.add("up")
    def _(event):
        nonlocal current_idx
        if current_idx >= cols:
            current_idx -= cols

    @kb.add("down")
    def _(event):
        nonlocal current_idx
        if current_idx + cols < len(themes):
            current_idx += cols

    @kb.add("left")
    def _(event):
        nonlocal current_idx
        if current_idx > 0:
            current_idx -= 1

    @kb.add("right")
    def _(event):
        nonlocal current_idx
        if current_idx < len(themes) - 1:
            current_idx += 1

    def hex_to_rgb(hx):
        hx = hx.lstrip("#")
        if len(hx) == 6:
            return tuple(int(hx[i : i + 2], 16) for i in (0, 2, 4))
        return (255, 255, 255)

    def rgb_to_hex(r, g, b):
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

    def gradient_str(text, c1, c2):
        r1, g1, b1 = hex_to_rgb(c1)
        r2, g2, b2 = hex_to_rgb(c2)
        length = len(text)
        if length <= 1:
            return f'<style fg="{c1}">{text}</style>'
        res = []
        for i, char in enumerate(text):
            ratio = i / (length - 1)
            r = r1 + (r2 - r1) * ratio
            g = g1 + (g2 - g1) * ratio
            b = b1 + (b2 - b1) * ratio
            hx = rgb_to_hex(r, g, b)
            res.append(f'<style fg="{hx}">{char}</style>')
        return "".join(res)

    def hexify(c_str):
        if "#" in c_str:
            return "#" + c_str.split("#")[1][:6]
        return "#ffffff"

    def get_header_text():
        t = themes[current_idx]
        theme_data = themes_dict[t]
        c1 = hexify(theme_data.get("primary", "#ffffff"))
        c2 = hexify(theme_data.get("secondary", "#ffffff"))
        prefix = "Select Theme:"
        suffix = " (Use Arrow Keys, Enter to confirm, Ctrl+C to cancel)"
        grad_prefix = gradient_str(prefix, c1, c2)
        grad_suffix = gradient_str(suffix, c1, c2)
        return HTML(f"<b>{grad_prefix}</b>{grad_suffix}\n")

    def get_grid_text():
        from prompt_toolkit.application.current import get_app

        try:
            terminal_width = get_app().output.get_size().columns
        except Exception:
            terminal_width = 120
        col_width = max(16, terminal_width // cols)
        lines = []
        for r in range(rows):
            line_parts = []
            for c in range(cols):
                idx = r * cols + c
                if idx < len(themes):
                    t = themes[idx]
                    t_display = t.capitalize()
                    if idx == current_idx:
                        theme_data = themes_dict[t]
                        c1 = hexify(theme_data.get("primary", "#ffffff"))
                        c2 = hexify(theme_data.get("secondary", "#ffffff"))
                        grad = gradient_str(t_display, c1, c2)
                        pointer = "❯ "
                        pad = " " * max(0, col_width - len(pointer) - len(t_display))
                        line_parts.append(f"<b>{pointer}{grad}</b>{pad}")
                    elif t == current_theme:
                        pointer = "✓ "
                        pad = " " * max(0, col_width - len(pointer) - len(t_display))
                        line_parts.append(f"<style fg='ansidarkgray'>{pointer}</style><b>{t_display}</b>{pad}")
                    else:
                        pad = " " * max(0, col_width - 2 - len(t_display))
                        line_parts.append(f"  <style fg='ansidarkgray'>{t_display}</style>{pad}")
                else:
                    line_parts.append(" " * col_width)
            lines.append("".join(line_parts))
        return HTML("\n".join(lines))

    body = HSplit(
        [
            Window(FormattedTextControl(get_header_text), height=2),
            Window(FormattedTextControl(get_grid_text), height=rows),
        ]
    )
    from prompt_toolkit.layout.layout import Layout

    app = Application(
        layout=Layout(body),
        key_bindings=kb,
        full_screen=False,
    )
    return app.run()
