import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dynamics_solver import plot_results


def test_plot_results_single():
    """Prueba la generación de un gráfico único con señales planas."""
    t = np.linspace(0.0, 5.0, 50)
    # 2 salidas para sys_a y 1 para sys_b
    outputs = {
        "sys_a": np.column_stack([np.sin(t), np.cos(t)]),
        "sys_b": np.ones((50, 1)) * 3.0,
    }

    # Graficar en un solo plot sin bloquear la interfaz
    fig, ax = plot_results(
        t=t,
        outputs=outputs,
        signals=["sys_b", ("sys_a", 0), ("sys_a", 1)],
        title="Prueba de Plot Único",
        show=False,
    )

    # Validar tipos devueltos
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    # Validar que se dibujaron 3 curvas en el eje
    assert len(ax.get_lines()) == 3
    # Limpiar matplotlib
    plt.close(fig)


def test_plot_results_subplots():
    """Prueba la generación de gráficos múltiples con estructura de subplots."""
    t = np.linspace(0.0, 5.0, 50)
    outputs = {
        "sys_a": np.column_stack([np.sin(t), np.cos(t)]),
        "sys_b": np.ones((50, 1)) * 3.0,
    }

    # 2 subplots: el primero con sys_b, el segundo con las salidas de sys_a
    fig, axs = plot_results(
        t=t,
        outputs=outputs,
        signals=[["sys_b"], [("sys_a", 0), ("sys_a", 1)]],
        title="Prueba de Subplots",
        show=False,
    )

    # Validar tipos
    assert isinstance(fig, Figure)
    assert isinstance(axs, np.ndarray)
    assert len(axs) == 2

    # Validar curvas en cada subplot
    assert len(axs[0].get_lines()) == 1
    assert len(axs[1].get_lines()) == 2
    plt.close(fig)


def test_plot_results_errors():
    """Prueba que el graficador capture y lance errores adecuados ante parámetros inválidos."""
    t = np.linspace(0.0, 5.0, 50)
    outputs = {
        "sys_a": np.ones((50, 1)),
    }

    # Lista vacía
    with pytest.raises(ValueError, match="no puede estar vacía"):
        plot_results(t, outputs, [], show=False)

    # Nombre inexistente
    with pytest.raises(KeyError, match="no se encuentra"):
        plot_results(t, outputs, ["sys_inexistente"], show=False)

    # Índice de salida fuera de rango
    with pytest.raises(IndexError, match="fuera de rango"):
        plot_results(t, outputs, [("sys_a", 5)], show=False)

    # Tipo de señal inválido
    with pytest.raises(ValueError, match="Especificación de señal inválida"):
        plot_results(t, outputs, [123], show=False)  # pyright: ignore[reportArgumentType]
