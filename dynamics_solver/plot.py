from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_results(
    t: np.ndarray,
    outputs: dict[str, np.ndarray],
    signals: Sequence[str | tuple[str, int]] | Sequence[Sequence[str | tuple[str, int]]],
    title: str = "Resultados de la Simulación",
    xlabel: str = "Tiempo (s)",
    grid: bool = True,
    figsize: tuple[float, float] = (10, 6),
    show: bool = True,
) -> tuple[Figure, Any]:
    """Grafica los resultados de una simulación a partir de las salidas y nombres de señales.

    Args:
        t (np.ndarray): Vector de tiempo de la simulación.
        outputs (dict[str, np.ndarray]): Diccionario de salidas devuelto por solver.solve.
        signals: Lista de señales a graficar.
            Puede ser:
            - Una lista plana: ['motor', ('pi', 0)] -> Grafica todo en una sola figura.
            - Una lista de listas: [['motor'], ['pi', 'sensor']] -> Crea subplots alineados.
        title (str): Título principal del gráfico.
        xlabel (str): Etiqueta para el eje X (común a todos los subplots).
        grid (bool): Activar/desactivar la rejilla.
        figsize (tuple): Tamaño de la figura.
        show (bool): Si es True, llama a plt.show() para visualizar inmediatamente.

    Returns:
        tuple[Figure, Any]: La figura y los ejes de matplotlib.
    """
    if len(signals) == 0:
        raise ValueError("La lista de señales a graficar no puede estar vacía.")

    # Determinar si es una lista plana o estructura de subplots (lista de listas)
    is_nested = isinstance(signals[0], (list, tuple, np.ndarray)) and not isinstance(
        signals[0], tuple
    )

    axs: Any = None
    if is_nested:
        # Estructura de subplots
        num_plots = len(signals)
        fig, axs = plt.subplots(num_plots, 1, sharex=True, figsize=figsize)
        if num_plots == 1:
            axs_list = [axs]
        else:
            axs_list = list(axs) if isinstance(axs, np.ndarray) else [axs]
        signals_list: list[Any] = list(signals)
    else:
        # Gráfico único
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axs_list = [ax]
        signals_list = [signals]

    # Iterar sobre cada subplot
    for subplot_idx, ax_current in enumerate(axs_list):
        ax_signals = signals_list[subplot_idx]

        for sig in ax_signals:
            if isinstance(sig, str):
                subsys_name = sig
                out_idx = 0
                label = subsys_name
            elif isinstance(sig, tuple) and len(sig) == 2:
                subsys_name = sig[0]
                out_idx = sig[1]
                label = f"{subsys_name}[{out_idx}]"
            else:
                raise ValueError(
                    f"Especificación de señal inválida '{sig}'. Debe ser un str o tuple (subsys_name, index)."
                )

            if subsys_name not in outputs:
                raise KeyError(
                    f"El subsistema '{subsys_name}' no se encuentra en el diccionario de salidas."
                )

            y_data = outputs[subsys_name]
            if out_idx >= y_data.shape[1]:
                raise IndexError(
                    f"Índice de salida {out_idx} fuera de rango para el subsistema '{subsys_name}' (dimensión de salida={y_data.shape[1]})."
                )

            ax_current.plot(t, y_data[:, out_idx], label=label, linewidth=1.5)

        ax_current.legend(loc="best")
        if grid:
            ax_current.grid(True, linestyle="--", alpha=0.7)

    # Configuración de etiquetas finales
    fig.suptitle(title, fontsize=14, fontweight="bold")
    if is_nested:
        # Colocar xlabel solo en el último subplot inferior
        axs_list[-1].set_xlabel(xlabel)
    else:
        axs_list[0].set_xlabel(xlabel)

    fig.tight_layout()

    if show:
        plt.show()

    if is_nested:
        return fig, axs
    else:
        return fig, axs_list[0]
