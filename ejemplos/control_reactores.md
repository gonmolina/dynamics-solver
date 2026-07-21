# Control y Dinámica de Reactores Nucleares

Este ejemplo muestra cómo modelar y simular la dinámica de un reactor nuclear con realimentación térmica utilizando el módulo `dynamics_solver`.

El sistema consiste en dos subsistemas principales acoplados en lazo cerrado:
1. **Cinética Puntual**: Representa el comportamiento neutrónico del núcleo del reactor (generación de potencia y concentración de precursores de neutrones retardados).
2. **Modelo Térmico**: Describe el comportamiento térmico del refrigerante que extrae calor del núcleo.

---

## 1. Modelo Físico y Ecuaciones

### A. Cinética Puntual de Neutrones (Con 1 Grupo de Retardados)
Las ecuaciones diferenciales que gobiernan la densidad neutrónica normalizada $n(t)$ (proporcional a la potencia) y la concentración de precursores retardados $c(t)$ son:

$$\frac{dn(t)}{dt} = \frac{\rho(t) - \beta}{\Lambda} n(t) + \lambda c(t)$$

$$\frac{dc(t)}{dt} = \frac{\beta}{\Lambda} n(t) - \lambda c(t)$$

Donde:
* $\rho(t)$ es la reactividad total del reactor.
* $\beta$ es la fracción de neutrones retardados.
* $\Lambda$ es el tiempo medio de generación de neutrones.
* $\lambda$ es la constante de decaimiento de los precursores.

**Salidas de interés**:
* $n(t)$: Densidad neutrónica (potencia del reactor).
* $\frac{1}{T} = \frac{dn/dt}{n}$: Inversa del período del reactor (parámetro clave para control y seguridad).

---

### B. Modelo Térmico del Refrigerante
La temperatura promedio del refrigerante $T(t)$ en el canal se modela como:

$$\frac{dT(t)}{dt} = \frac{1}{m c_p} \left[ K n(t) - 2 W c_p (T(t) - T_{in}) \right]$$

Donde:
* $m$ es la masa del refrigerante en el canal.
* $c_p$ es el calor específico del refrigerante.
* $W$ es el flujo másico del refrigerante.
* $T_{in}$ es la temperatura del refrigerante a la entrada.
* $K$ es la constante de acoplamiento potencia-neutrones.

---

### C. Realimentación de Reactividad por Temperatura
La reactividad total del reactor $\rho(t)$ está dada por la reactividad externa insertada por las barras de control ($\rho_{ext}$) más el efecto térmico (realimentación negativa):

$$\rho(t) = \rho_{ext}(t) + \alpha_{th} (T(t) - T_0)$$

Donde:
* $\alpha_{th}$ es el coeficiente de reactividad por temperatura (negativo para reactores estables).
* $T_0$ es la temperatura promedio de referencia en estado estacionario.

---

## 2. Implementación de los Bloques en Python

Para representar esta topología, definimos tres bloques:
1. `CineticaPuntual` (bloque dinámico con 2 estados).
2. `ModeloTermico` (bloque dinámico con 1 estado).
3. `ReactivityFeedback` (bloque puramente algebraico, sin estados, que calcula la reactividad combinada).

No requerimos bloques puente para las entradas ni salidas, ya que usaremos las capacidades de **mapeo externo** de `InterconnectedSystem`.

```python
import numpy as np
from dynamics_solver import SubsystemBase, InterconnectedSystem

# Parámetros físicos
Tin_val = 270.0      # Temperatura de entrada del refrigerante nominal [°C]
beta = 650e-5        # Fracción de neutrones retardados
Lambda = 1e-3        # Tiempo de generación de neutrones [s]
lamb = 0.077         # Constante de desintegración promedio [s^-1]
N0 = 1.0             # Potencia inicial
C0 = (beta / (Lambda * lamb)) * N0  # Precursores iniciales

W = 10.0             # Flujo másico [kg/s]
m = 1200.0           # Masa de refrigerante [kg]
cp = 4.18e3          # Calor específico [J/kg°C]
K = 20.0 * 2.0 * W * cp  # Constante de escala [W]
alfa_th = -20e-5     # Coeficiente por temperatura [°C^-1]
T0 = K * N0 / (2.0 * W * cp) + Tin_val  # Temperatura promedio inicial (290.0 °C)

class CineticaPuntual(SubsystemBase):
    def __init__(self, name: str = "cinetica_puntual"):
        super().__init__(
            state_dim=2,
            input_dim=1,
            output_dim=2,
            name=name,
            direct_feedthrough=True,
            x0=[N0, C0]
        )

    def update_outputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # y[0] = 1/T (Inversa del período), y[1] = n (Potencia)
        n_val = max(x[0], 1e-6)
        inv_T = ((u[0] - beta) / Lambda) + (lamb * x[1] / n_val)
        return np.array([inv_T, x[0]])

    def compute_derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        dn = ((u[0] - beta) / Lambda) * x[0] + lamb * x[1]
        dc = (beta / Lambda) * x[0] - lamb * x[1]
        return np.array([dn, dc])

class ModeloTermico(SubsystemBase):
    def __init__(self, name: str = "modelo_termico"):
        super().__init__(
            state_dim=1,
            input_dim=2,
            output_dim=1,
            name=name,
            direct_feedthrough=False,
            x0=[T0]
        )

    def update_outputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # y[0] = T (Temperatura promedio del refrigerante)
        return np.array([x[0]])

    def compute_derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # u[0] = n (potencia), u[1] = Tin (temperatura de entrada)
        dT = (K * u[0] - 2.0 * W * cp * (x[0] - u[1])) / (m * cp)
        return np.array([dT])

class ReactivityFeedback(SubsystemBase):
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
```

---

## 3. Construcción de la Red e Interconexiones

Definimos que el sistema `InterconnectedSystem` tiene **2 entradas externas** y **3 salidas externas**:
* **Entradas externas**:
  - Entrada `0`: Reactividad externa $\rho_{\text{ext}}$.
  - Entrada `1`: Temperatura de entrada del refrigerante $T_{\text{in}}$ (perturbación).
* **Salidas externas**:
  - Salida `0`: Potencia neutrónica normalizada $n$.
  - Salida `1`: Inversa del período del reactor $1/T$.
  - Salida `2`: Temperatura promedio del refrigerante $T$ (salida del modelo térmico).

```python
# Instanciamos el resolvedor con 2 entradas y 3 salidas externas
sys_net = InterconnectedSystem(input_dim=2, output_dim=3)

# Bloques
cinetica = CineticaPuntual()
termico = ModeloTermico()
feedback = ReactivityFeedback()

sys_net.add_subsystem(cinetica)
sys_net.add_subsystem(termico)
sys_net.add_subsystem(feedback)

# Conexiones internas
sys_net.connect(feedback, 0, cinetica, 0)
sys_net.connect(cinetica, 1, termico, 0)
sys_net.connect(termico, 0, feedback, 0)

# Mapeos de entradas externas
sys_net.map_input(0, feedback, 1)  # Entrada externa 0 -> Entrada 1 de feedback (rho_ext)
sys_net.map_input(1, termico, 1)   # Entrada externa 1 -> Entrada 1 de termico (Tin)

# Mapeos de salidas externas
sys_net.map_output(0, cinetica, 1)  # Salida externa 0 -> n (Salida 1 de cinética)
sys_net.map_output(1, cinetica, 0)  # Salida externa 1 -> 1/T (Salida 0 de cinética)
sys_net.map_output(2, termico, 0)   # Salida externa 2 -> T (Salida 0 de térmico)
```

---

## 4. Simulación del Transitorio de Reactividad y Perturbación Térmica

Simulamos un transitorio de 200 segundos aplicando dos estímulos secuenciales:
1. Escalón de reactividad de $+10 \times 10^{-5}$ a los $t = 2.0$ segundos (barras de control).
2. Perturbación de $+5.0$ °C en la temperatura de entrada del refrigerante $T_{\text{in}}$ a los $t = 100.0$ segundos.

```python
def input_signals(t: float) -> np.ndarray:
    rho_ext = 10e-5 if t >= 2.0 else 0.0
    Tin = (Tin_val + 5.0) if t >= 100.0 else Tin_val
    return np.array([rho_ext, Tin])

t, x_global, states, outputs = sys_net.solve(
    t_span=(0.0, 400.0),
    u_ext_func=input_signals,
    max_step=0.1
)
```

## 5. Linealización del Sistema en Estado Estacionario

El estado estacionario inicial del sistema global se construye a partir de los $x_0$ por defecto de cada bloque

```python
x0_estacionario = sys_net.x0
```

Entradas de operación nominal: reactividad externa 0.0 y Tin de 270 °C

```python
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
```

```python
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

---

## 5. Linealización de Subsistemas Individuales

Además de linealizar la red de interconexión global, es posible linealizar cada bloque o subsistema de forma independiente. Esto es sumamente útil para analizar por separado la dinámica neutrónica o térmica de forma aislada:

* **Cinética Puntual**: Se linealiza alrededor del punto nominal $n_0 = 1.0$, $c_0 = C_0$ con reactividad nominal $u_0 = [0.0]$:
  ```python
  A_cin, B_cin, C_cin, D_cin = cinetica.linearize(x0=[N0, C0], u0=[0.0])
  ```
* **Modelo Térmico**: Se linealiza alrededor de la temperatura nominal de refrigerante $T_0 = 290.0$ °C y entradas nominales de potencia $n_0 = 1.0$ y temperatura de entrada $T_{\text{in}} = 270.0$ °C:
  ```python
  A_ter, B_ter, C_ter, D_ter = termico.linearize(x0=[T0], u0=[1.0, Tin_val])
  ```

---

```python

```

```python
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
```


## 6. Visualización de Resultados (Entradas y Salidas)

Podemos utilizar la herramienta integrada `plot_results` para graficar la evolución temporal del transitorio. Graficaremos tanto las **3 salidas externas** como las **2 entradas externas**:

* **Subplot 1**: Temperatura promedio del refrigerante $T$ (Salida 2) y Temperatura de entrada $T_{\text{in}}$ (Entrada 1).
* **Subplot 2**: Potencia neutrónica normalizada $n$ (Salida 0).
* **Subplot 3**: Inversa del período del reactor $1/T$ (Salida 1).
* **Subplot 4**: Reactividades del sistema (Reactividad total $\rho$, realimentación térmica $\rho_{\text{th}}$, y reactividad de barras $\rho_{\text{ext}}$ que es la Entrada 0).

```python
from dynamics_solver import plot_results

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
    show=True,
)
```

Al observar las gráficas, se destaca cómo:
1. El escalón de reactividad inicial a $t=2$ s provoca un pico rápido en la potencia $n$ y la inversa de período $1/T$. La realimentación térmica negativa ($\rho_{\text{th}}$) actúa limitando el transitorio.
2. A $t=100$ s, la perturbación térmica externa de $+5$ °C en el refrigerante de entrada ($T_{\text{in}}$) reduce la refrigeración del núcleo, provocando un aumento en la temperatura media $T$ y, consecuentemente, una inserción de reactividad negativa térmica que estabiliza el reactor a un nivel de potencia menor ($n \approx 0.75$).

```python
import control as ct
import matplotlib.pyplot as plt
%matplotlib qt
```

```python
Gcp=ct.ss(A_cin, B_cin, C_cin, D_cin)
Gter= ct.ss(A_ter, B_ter, C_ter, D_ter) 
```

```python
Gcp0=Gcp[0,0]
Gter0=Gter[0,0]
```

```python
ct.rlocus(Gcp0*Gter0)
```

```python

```
