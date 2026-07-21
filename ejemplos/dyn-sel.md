---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from dynamics_solver import InterconnectedSystem, SubsystemBase, plot_results
%matplotlib widget
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
class ReactivityFeedback(SubsystemBase):
    """Bloque algebraico para calcular la reactividad total con realimentación térmica."""

    def __init__(self, name: str = "realimentacion", alpha_th=alfa_th):
        super().__init__(state_dim=0, input_dim=2, output_dim=3, name=name, direct_feedthrough=True)
        self.alpha_th=alpha_th

    def update_outputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # u[0] = T (Temperatura promedio del refrigerante)
        # u[1] = rho_ext (Reactividad externa / barras de control)
        rho_feedback = self.alpha_th * (u[0] - T0)
        rho_total = u[1] + rho_feedback
        return np.array([rho_total, rho_feedback, u[1]])

    def compute_derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([])

    def set_alpha_th(self, alpha_th):
        self.alpha_th = alpha_th
```

```{code-cell} ipython3
class ExternalInputBlock(SubsystemBase):
    """Bloque puente para extraer señales externas globales del vector de entrada."""

    def __init__(self, external_idx: int, name: str):
        super().__init__(state_dim=0, input_dim=0, output_dim=1, name=name, direct_feedthrough=True)
        self.external_idx = external_idx
        self.val = 0.0

    def get_connected_inputs(self, external_inputs: np.ndarray | None = None) -> np.ndarray:
        if external_inputs is not None and self.external_idx < len(external_inputs):
            self.val = external_inputs[self.external_idx]
        return np.array([])

    def update_outputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([self.val])

    def compute_derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([])
```

```{code-cell} ipython3
# 2. Configurar el Sistema Interconectado
sys_net = InterconnectedSystem()

# Instanciar bloques
cinetica = CineticaPuntual()
termico = ModeloTermico()
feedback = ReactivityFeedback()
rho_ext_src = ExternalInputBlock(external_idx=0, name="rho_ext_src")
Tin_src = ExternalInputBlock(external_idx=1, name="Tin_src")

# Agregar bloques a la red
sys_net.add_subsystem(cinetica)
sys_net.add_subsystem(termico)
sys_net.add_subsystem(feedback)
sys_net.add_subsystem(rho_ext_src)
sys_net.add_subsystem(Tin_src)


# Conexiones
# Cinetica depende de la reactividad total calculada por el bloque de feedback
sys_net.connect(feedback, 0, cinetica, 0)

# Modelo termico depende de la potencia de cinetica y de la temperatura de entrada
sys_net.connect(cinetica, 1, termico, 0)
sys_net.connect(Tin_src, 0, termico, 1)

# El feedback depende de la temperatura del refrigerante y de la reactividad externa
sys_net.connect(termico, 0, feedback, 0)
sys_net.connect(rho_ext_src, 0, feedback, 1)


# 3. Resolver la Simulación
# Definir estímulos externos:
# A los 2.0 segundos, se realiza un escalón de reactividad de 10e-5 deltaK (barras de control).
# La temperatura de entrada Tin se mantiene constante en 270 °C.
def input_signals(t: float) -> np.ndarray:
    rho_ext = 10e-5 if t >= 2.0 else 0.0
    return np.array([rho_ext, Tin_val])
```

```{code-cell} ipython3
t, x_global, states, outputs = sys_net.solve(
    t_span=(0.0, 200.0),
    u_ext_func=input_signals,
    max_step=0.1,  # Limitamos el paso del integrador para ver la dinámica fina del transitorio
)
```

```{code-cell} ipython3
# 4. Mostrar resultados e inspeccionar valores clave
print("Simulación de control de reactores finalizada.")
print(f"Pasos de simulación: {len(t)}")
print(f"Potencia inicial: {outputs['cinetica_puntual'][0, 1]:.4f}")
print(f"Potencia máxima alcanzada (pico): {np.max(outputs['cinetica_puntual'][:, 1]):.4f}")
print(f"Potencia final (t=200s): {outputs['cinetica_puntual'][-1, 1]:.4f}")
print(f"Temperatura inicial del refrigerante: {outputs['modelo_termico'][0, 0]:.2f} °C")
print(f"Temperatura final del refrigerante (t=200s): {outputs['modelo_termico'][-1, 0]:.2f} °C")

# 5. Linealización del Sistema en Estado Estacionario
# El estado estacionario inicial del sistema global se construye a partir de los x0 por defecto de cada bloque
x0_estacionario = np.concatenate([sys.x0 for sys in sys_net.subsystems])
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
```

```{code-cell} ipython3
# 6. Graficar Resultados del Transitorio
# Subplots:
# 1. Temperatura del refrigerante
# 2. Potencia neutrónica n
# 3. Inversa de período 1/T
# 4. Reactividades (Total, Feedback Térmica, Externa)
signals = [
    ["modelo_termico"],
    [("cinetica_puntual", 1)],
    [("cinetica_puntual", 0)],
    [("realimentacion", 0), ("realimentacion", 1), ("realimentacion", 2)],
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
axs[0].legend(["T (Promedio)"])

axs[1].set_ylabel("Potencia (norm.)")
axs[1].legend(["n (Potencia)"])

axs[2].set_ylabel("Inversa Período (1/s)")
axs[2].legend(["1/T"])

axs[3].set_ylabel("Reactividad (Δk)")
axs[3].legend(["Total (ρ)", "Térmica (ρ_th)", "Barras (ρ_ext)"])
fig.tight_layout()
fig
```

## Estudio del efecto de la realimentación de temperatura

Vamos a estudiar como se comporta la dinámica de este sistema pra distintos valores de $\alpha$. A $\alpha$ le haremos tomar los valores de $[0, 5.10^{-5}, 10.10^{-5}, 20e-5, 100.10^{-5}]$

```{code-cell} ipython3
sol=[]
alpha=[0, -5e-5, -10e-5, -20e-5, -100e-5]
for i in alpha:
    alpha_th = i
    sys_net.subsystems[2].set_alpha_th(alpha_th)
    print(f"Simulando con alpha={alpha_th}")
    si = sys_net.solve(
        t_span=(0.0, 200.0),
        u_ext_func=input_signals,
        max_step=0.1,  # Limitamos el paso del integrador para ver la dinámica fina del transitorio
    )
    sol.append(si)
    
```

```{code-cell} ipython3
signals = [
    [("cinetica_puntual", 1)],
]

f=plt.figure()
for i in sol:
    t, x_global, states, outputs = i 
    plt.plot(t, outputs['cinetica_puntual'][:,1])

    
    
```

```{code-cell} ipython3
f
```

```{code-cell} ipython3
sys_net.subsystems[2].set_alpha_th(0e-5)
```

```{code-cell} ipython3

```
