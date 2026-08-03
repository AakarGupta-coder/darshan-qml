import os
import time
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

def run_quantum_advantage(dataset_name="Synthetic"):
    """
    Measures the theoretical Quantum Advantage scaling for Samyoga Pro (TNQE) and Ananta (VQC).
    It compares the dimensionality of the quantum Hilbert Space against classical features.
    """
    console.print("\n[bold cyan]==================================================[/bold cyan]")
    console.print("[bold cyan]       QUANTUM ADVANTAGE & SCALING ANALYSIS       [/bold cyan]")
    console.print("[bold cyan]==================================================[/bold cyan]\n")

    qubits = [2, 4, 6, 8, 10, 12, 16]
    results = []

    for q in qubits:
        classical_dim = q

        hilbert_dim = 2 ** q

        classical_sim_complexity = (q ** 2) * np.log2(hilbert_dim + 1)

        advantage_gap = hilbert_dim / classical_dim

        results.append({
            'Qubits': q,
            'Classical Features': classical_dim,
            'Hilbert Space Dim': hilbert_dim,
            'Advantage Multiplier': f"{advantage_gap:,.1f}x",
            'Classical Sim Complexity': f"{classical_sim_complexity:,.1f}"
        })
        time.sleep(0.1)

    df = pd.DataFrame(results)

    table = Table(title="Theoretical Quantum Scaling", show_header=True, header_style="bold magenta")
    for col in df.columns:
        table.add_column(col, justify="center")

    for _, row in df.iterrows():
        table.add_row(*[str(x) for x in row.values])

    console.print(table)

    try:
        from ui.graphs import render_terminal_curve
        curve_dict = {
            'Advantage Multiplier (2^N / N)': [2 ** q / q for q in qubits],
            'Classical Sim Complexity': [(q ** 2) * np.log2(2 ** q + 1) for q in qubits],
            'Classical Features (x100)': [q * 100 for q in qubits]
        }
        render_terminal_curve("Theoretical Quantum Advantage Scaling", qubits, curve_dict, "Qubits (N)", "Complexity / Advantage")
    except Exception as e:
        console.print(f"[dim error]Failed to render graph: {e}[/]")

    console.print("\n[bold yellow]Conclusion:[/bold yellow]")
    console.print("As the number of qubits scales linearly, the representational capacity of the models ")
    console.print("(like Samyoga Pro's TNQE) scales exponentially in the Hilbert space.")
    console.print("This reveals why classical simulation (CPU) slows down dramatically at higher qubits,")
    console.print("and why Quantum Error Correction on real hardware is needed for N > 30.\n")

if __name__ == "__main__":
    run_quantum_advantage()