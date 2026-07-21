import pathlib
import sys

# Añadir el directorio raíz del proyecto al path de Python para importar dynamics_solver localmente
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


import matplotlib.pyplot as plt
import numpy as np

from dynamics_solver import InterconnectedSystem, SubsystemBase, plot_results

# Parámetros del Reactor Nuclear
Tin_val = 270.0  # Temperatura de entrada del refrigerante [°C]
beta = 650e-5  # Fracción de neutrones retardados
Lambda = 1e-3  # Tiempo de generación de neutrones [s]
lamb = 0.077  # Constante de desintegración promedio de precursores [s^-1]
N0 = 1.0  # Potencia neutrónica inicial (normalizada)
C0 = (beta / (Lambda * lamb)) * N0  # Concentración inicial de precursores en estado estacionario

W = 10.0  # Flujo másico del refrigerante [kg/s]
m = 1200.0  # Masa del refrigerante en el canal [kg]
cp = 4.18e3  # Calor específico del refrigerante [J/kg°C]
K = 20.0 * 2.0 * W * cp  # Constante de conversión a potencia [W]
alfa_th = -20e-5  # Coeficiente de reactividad por temperatura [°C^-1]
T0 = K * N0 / (2.0 * W * cp) + Tin_val  # Temperatura promedio de referencia [°C] (290.0 °C)

# 1. Definición de Bloques


class CineticaPuntual(SubsystemBase):
    """Modelo de Cinética Puntual con 1 grupo de neutrones retardados."""

    def __init__(self, name: str = "cinetica_puntual"):
        super().__init__(
            state_dim=2, input_dim=1, output_dim=2, name=name, direct_feedthrough=True, x0=[N0, C0]
        )

    def update_outputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # y[0] = 1/T (Inversa del período del reactor)
        # y[1] = n (Potencia neutrónica)
        # Para evitar divisiones por cero, protegemos x[0]
        n_val = max(x[0], 1e-6)
        inv_T = ((u[0] - beta) / Lambda) + (lamb * x[1] / n_val)
        return np.array([inv_T, x[0]])

    def compute_derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # dx/dt = [dn/dt, dc/dt]
        # dn/dt = (rho - beta)/Lambda * n + lamb * c
        # dc/dt = beta/Lambda * n - lamb * c
        dn = ((u[0] - beta) / Lambda) * x[0] + lamb * x[1]
        dc = (beta / Lambda) * x[0] - lamb * x[1]
        return np.array([dn, dc])


class ModeloTermico(SubsystemBase):
    """Modelo térmico del canal de refrigerante."""

    def __init__(self, name: str = "modelo_termico"):
        super().__init__(
            state_dim=1, input_dim=2, output_dim=1, name=name, direct_feedthrough=False, x0=[T0]
        )

    def update_outputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # y[0] = T (Temperatura promedio del refrigerante)
        return np.array([x[0]])

    def compute_derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # u[0] = n (Potencia neutrónica)
        # u[1] = Tin (Temperatura de entrada)
        # dT/dt = (K*n - 2*W*cp*(T - Tin)) / (m*cp)
        dT = (K * u[0] - 2.0 * W * cp * (x[0] - u[1])) / (m * cp)
        return np.array([dT])


class ReactivityFeedback(SubsystemBase):
    """Bloque algebraico para calcular la reactividad total con realimentación térmica."""

    def __init__(self, name: str = "realimentacion"):
        super().__init__(state_dim=0, input_dim=2, output_dim=3, name=name, direct_feedthrough=True)

    def update_outputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # u[0] = T (Temperatura promedio del refrigerante)
        # u[1] = rho_ext (Reactividad externa / barras de control)
        rho_feedback = alfa_th * (u[0] - T0)
        rho_total = u[1] + rho_feedback
        return np.array([rho_total, rho_feedback, u[1]])

    def compute_derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([])


# 2. Configurar el Sistema Interconectado
# Definimos que el sistema tiene 2 entradas y 3 salidas externas
sys_net = InterconnectedSystem(input_dim=2, output_dim=3)

# Instanciar bloques
cinetica = CineticaPuntual()
termico = ModeloTermico()
feedback = ReactivityFeedback()

# Agregar bloques a la red
sys_net.add_subsystem(cinetica)
sys_net.add_subsystem(termico)
sys_net.add_subsystem(feedback)

# Conexiones internas
sys_net.connect(feedback, 0, cinetica, 0)
sys_net.connect(cinetica, 1, termico, 0)
sys_net.connect(termico, 0, feedback, 0)

# Mapeos de entradas externas
sys_net.map_input(0, feedback, 1)  # Entrada 0: rho_ext (reactividad externa)
sys_net.map_input(1, termico, 1)  # Entrada 1: Tin (temperatura de entrada)

# Mapeos de salidas externas
sys_net.map_output(0, cinetica, 1)  # Salida 0: n (potencia neutrónica)
sys_net.map_output(1, cinetica, 0)  # Salida 1: 1/T (inversa de período)
sys_net.map_output(2, termico, 0)  # Salida 2: T (temperatura promedio)


# 3. Resolver la Simulación
# Definir estímulos externos:
# A los 2.0 segundos, se realiza un escalón de reactividad de 10e-5 deltaK.
# A los 100.0 segundos, se aplica una perturbación de +5.0 °C en la temperatura de entrada Tin.
def input_signals(t: float) -> np.ndarray:
    rho_ext = 10e-5 if t >= 2.0 else 0.0
    Tin = (Tin_val + 5.0) if t >= 100.0 else Tin_val
    return np.array([rho_ext, Tin])


t, x_global, states, outputs = sys_net.solve(
    t_span=(0.0, 200.0),
    u_ext_func=input_signals,
    max_step=0.1,  # Limitamos el paso del integrador para ver la dinámica fina del transitorio
)

# 4. Mostrar resultados e inspeccionar valores clave
print("Simulación de control de reactores finalizada.")
print(f"Pasos de simulación: {len(t)}")
print(f"Potencia inicial: {outputs['interconnected_system'][0, 0]:.4f}")
print(f"Potencia máxima alcanzada (pico): {np.max(outputs['interconnected_system'][:, 0]):.4f}")
print(f"Potencia final (t=200s): {outputs['interconnected_system'][-1, 0]:.4f}")
print(f"Temperatura inicial del refrigerante: {outputs['interconnected_system'][0, 2]:.2f} °C")
print(
    f"Temperatura final del refrigerante (t=200s): {outputs['interconnected_system'][-1, 2]:.2f} °C"
)

# 5. Linealización del Sistema en Estado Estacionario
# El estado estacionario inicial del sistema global se construye a partir de los x0 por defecto de cada bloque
x0_estacionario = sys_net.x0
# Entradas de operación nominal: reactividad externa 0.0 y Tin de 270 °C
u0_operacion = np.array([0.0, Tin_val])

A, B, C, D = sys_net.linearize(x0_estacionario, u0_operacion)

print("\n--- LINEALIZACIÓN DEL SISTEMA ---")
print("Matriz A (Dinámica de Estados):")
print(np.array2string(A, precision=6, suppress_small=True))
print("\nMatriz B (Influencia de Entradas Externas):")
print(np.array2string(B, precision=6, suppress_small=True))
print("\nMatriz C (Relación de Salidas con Estados):")
print(np.array2string(C, precision=6, suppress_small=True))
print("\nMatriz D (Transmisión Directa Entrada-Salida):")
print(np.array2string(D, precision=6, suppress_small=True))

# Calcular los autovalores de A para validar la estabilidad lineal del reactor
autovalores = np.linalg.eigvals(A)
print("\nAutovalores de la matriz A (Estabilidad del reactor):")
for idx, val in enumerate(autovalores):
    real_part = float(np.real(val))
    imag_part = float(np.imag(val))
    print(f"  Autovalor {idx + 1}: {real_part:+.6f} + {imag_part:+.6f}j")

assert np.all(np.real(autovalores) < 0), (
    "El reactor nuclear debería ser linealmente estable con realimentación negativa."
)
print("\n¡Verificación de estabilidad lineal exitosa!")

# 5.b Linealización de Subsistemas Individuales
A_cin, B_cin, C_cin, D_cin = cinetica.linearize(x0=[N0, C0], u0=[0.0])
A_ter, B_ter, C_ter, D_ter = termico.linearize(x0=[T0], u0=[1.0, Tin_val])

print("\n--- LINEALIZACIÓN DE SUBSISTEMAS INDIVIDUALES ---")
print("\n[CINETICA PUNTUAL] (Estados: [n, c], Entrada: [rho_total])")
print("Matriz A_cin:")
print(np.array2string(A_cin, precision=6, suppress_small=True))
print("Matriz B_cin:")
print(np.array2string(B_cin, precision=6, suppress_small=True))
print("Matriz C_cin:")
print(np.array2string(C_cin, precision=6, suppress_small=True))
print("Matriz D_cin:")
print(np.array2string(D_cin, precision=6, suppress_small=True))

print("\n[MODELO TERMICO] (Estado: [T], Entradas: [n, Tin])")
print("Matriz A_ter:")
print(np.array2string(A_ter, precision=6, suppress_small=True))
print("Matriz B_ter:")
print(np.array2string(B_ter, precision=6, suppress_small=True))
print("Matriz C_ter:")
print(np.array2string(C_ter, precision=6, suppress_small=True))
print("Matriz D_ter:")
print(np.array2string(D_ter, precision=6, suppress_small=True))

# 6. Graficar Resultados del Transitorio
# Subplots:
# 1. Temperaturas del refrigerante (Promedio e Entrada)
# 2. Potencia neutrónica n
# 3. Inversa de período 1/T
# 4. Reactividades (Total, Feedback Térmica, Externa)
signals = [
    [("interconnected_system", 2), ("inputs", 1)],
    [("interconnected_system", 0)],
    [("interconnected_system", 1)],
    [("realimentacion", 0), ("realimentacion", 1), ("inputs", 0)],
]

fig, axs = plot_results(
    t=t,
    outputs=outputs,
    signals=signals,
    title="Transitorio de Control de Reactor Nuclear",
    xlabel="Tiempo (s)",
    figsize=(12, 10),
    show=False,
)

# Personalizar etiquetas y leyendas de cada subplot para máxima claridad
axs[0].set_ylabel("Temp. Refrigerante (°C)")
axs[0].legend(["T (Promedio)", "T_in (Entrada / Perturbación)"])

axs[1].set_ylabel("Potencia (norm.)")
axs[1].legend(["n (Potencia)"])

axs[2].set_ylabel("Inversa Período (1/s)")
axs[2].legend(["1/T"])

axs[3].set_ylabel("Reactividad (Δk)")
axs[3].legend(["Total (ρ)", "Térmica (ρ_th)", "Barras (ρ_ext / Entrada)"])

plt.show()
