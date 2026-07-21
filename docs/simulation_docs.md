# Documentación de Simulación en `dynamics_solver`

Se han agregado métodos de resolución (`solve`) a las clases `SubsystemBase` e `InterconnectedSystem` (con el alias compatible `InterconnectedSubsystem`) para simular y obtener los estados y salidas de los sistemas dinámicos definidos en forma de diagramas de bloques.

## Métodos Agregados

### 1. `SubsystemBase.solve`
Resuelve la simulación de un subsistema aislado en el tiempo utilizando el integrador numérico `scipy.integrate.solve_ivp`.

```python
def solve(
    self,
    t_span: tuple[float, float],
    x0: np.ndarray | Sequence[float] | None = None,
    u_ext_func: Callable[[float], np.ndarray] | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
```
- **x0 (opcional)**: Vector de estado inicial local. Si se omite o es `None`, se utiliza el estado inicial `self.x0` definido al instanciar el subsistema (por defecto, un vector de ceros de tamaño `state_dim`).
- **Retorna**: Un tuple `(t, x, y)` con:
  - `t` (np.ndarray): Tiempos de los pasos de integración.
  - `x` (np.ndarray): Matriz de estados históricos con forma `(N, state_dim)`.
  - `y` (np.ndarray): Matriz de salidas históricas con forma `(N, output_dim)`.

---

### 2. `InterconnectedSystem.solve` (o alias `InterconnectedSubsystem.solve`)
Resuelve la simulación de la red completa de subsistemas interconectados.

```python
def solve(
    self,
    t_span: tuple[float, float],
    x0_global: np.ndarray | Sequence[float] | None = None,
    u_ext_func: Callable[[float], np.ndarray] | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
```
- **x0_global (opcional)**: Vector de estado inicial global para toda la red. Si se omite o es `None`, el sistema global autoconstruirá el vector concatenando los estados `x0` por defecto de cada subsistema agregado en el orden correspondiente.
- **Retorna**: Un tuple `(t, x_global, states_dict, outputs_dict)` con:
  - `t` (np.ndarray): Tiempos de los pasos de integración.
  - `x_global` (np.ndarray): Matriz consolidada de todos los estados históricos.
  - `states_dict` (dict[str, np.ndarray]): Historial de estados de cada subsistema por nombre.
  - `outputs_dict` (dict[str, np.ndarray]): Historial de salidas de cada subsistema por nombre.

---

## Ejemplo Práctico de Uso

El siguiente ejemplo muestra cómo crear un sistema interconectado que consta de un **Integrador** y un bloque **Ganancia**, conectarlos, simularlos y obtener los resultados:

```python
import numpy as np
from dynamics_solver import SubsystemBase, InterconnectedSystem

# 1. Definir los Bloques del Diagrama
class Integrator(SubsystemBase):
    def __init__(self, name="integrator"):
        super().__init__(state_dim=1, input_dim=1, output_dim=1, name=name, direct_feedthrough=False)

    def update_outputs(self, x, u):
        return x

    def compute_derivatives(self, x, u):
        return u

class Gain(SubsystemBase):
    def __init__(self, k, name="gain"):
        self.k = k
        super().__init__(state_dim=0, input_dim=1, output_dim=1, name=name, direct_feedthrough=True)

    def update_outputs(self, x, u):
        return self.k * u

    def compute_derivatives(self, x, u):
        return np.array([])

# 2. Inicializar y Conectar
sys_net = InterconnectedSystem()
int1 = Integrator("int1")
gain1 = Gain(3.0, "gain1")

sys_net.add_subsystem(int1)
sys_net.add_subsystem(gain1)
sys_net.connect(int1, 0, gain1, 0)

# 3. Resolver / Simular
# Entrada externa: u(t) = 4.0 (Constante)
# Integrador parte de x0 = 2.0
t, x_global, states, outputs = sys_net.solve(
    t_span=(0, 2),
    x0_global=[2.0],
    u_ext_func=lambda t: np.array([4.0])
)

# 4. Obtener Estados y Salidas
print("Estados del Integrador:", states["int1"].flatten())
print("Salidas de la Ganancia:", outputs["gain1"].flatten())
```

> [!NOTE]
> Cualquier argumento keyword adicional (`**kwargs`) provisto a `solve` será reenviado directamente a `scipy.integrate.solve_ivp`, permitiendo configurar el método de integración (e.g., `method='RK45'`), tolerancias (`rtol`, `atol`), pasos de tiempo específicos (`t_eval`), entre otros.

---

## Librería de Bloques Básicos (`dynamics_solver.blocks`)

El módulo expone una librería de bloques y fuentes matemáticas reutilizables, disponibles también con alias en español para mayor comodidad.

### 1. Fuentes de Entrada (Sources)
* **`ConstantSource` (Alias `Constante`)**: Genera una señal constante.
  ```python
  Constante(value=10.0, name="cte")
  ```
* **`StepSource` (Alias `Escalon`)**: Genera un escalón temporal en una transición dada.
  ```python
  Escalon(initial_value=0.0, final_value=5.0, step_time=2.0, name="step")
  ```
* **`TimeFunctionSource` (Alias `FuncionTemporal`)**: Genera una señal definida por una función matemática arbitraria del tiempo.
  ```python
  FuncionTemporal(func=lambda t: 2.0 * np.sin(t), name="seno")
  ```

### 2. Bloques Matemáticos
* **`Adder` (Alias `SumadorRestador`)**: Suma o resta entradas. El número de entradas y sus signos se configuran mediante un string (e.g., `"++-"` para sumar dos entradas y restar la tercera).
  ```python
  SumadorRestador(signs="++-", name="sumador")
  ```
* **`Gain` (Alias `Ganancia`)**: Multiplica una única entrada por una constante.
  ```python
  Ganancia(k=0.5, name="mitad")
  ```
* **`MultiplierDivider` (Alias `MultiplicadorDivisor`)**: Multiplica o divide múltiples entradas consecutivamente. Las operaciones se definen mediante un string (e.g., `"* /"` sin espacios para indicar multiplicación y división consecutiva).
  ```python
  MultiplicadorDivisor(operations="*/", name="mul_div")
  ```

### 3. Bloques Dinámicos
* **`Integrator` (Alias `Integrador`)**: Realiza la integración temporal continua de su señal de entrada ($1/s$).
  ```python
  Integrador(x0=0.0, name="integrador")
  ```
* **`TransferFunction` (Alias `FuncionTransferencia`)**: Representa un filtro dinámico lineal a partir de los coeficientes polinomiales del numerador y denominador (en potencias descendentes de $s$).
  Determina de forma automática si tiene paso directo (`direct_feedthrough=True` si el grado del numerador es igual al del denominador, y `False` si es menor).
  ```python
  # G(s) = 1 / (s^2 + 2s + 1)
  FuncionTransferencia(num=[1], den=[1, 2, 1], name="filtro")
  ```
* **`StateSpace` (Alias `EspacioEstados`)**: Representa un bloque lineal continuo multivariable definido en espacio de estados mediante las matrices constantes $A, B, C, D$.
  Determina automáticamente si tiene paso directo (`direct_feedthrough=True` si al menos un elemento de $D$ no es cero, y `False` si es la matriz nula).
  ```python
  # Sistema de segundo orden
  A = [[-2.0, -1.0], [1.0, 0.0]]
  B = [[1.0], [0.0]]
  C = [[0.0, 1.0]]
  D = [[0.0]]
  EspacioEstados(A, B, C, D, name="filtro_ss")
  ```

---

## Linealización y Estado Estacionario

El solver incluye utilidades numéricas avanzadas para calcular el estado estacionario de una red y aproximar linealmente su comportamiento dinámico en espacio de estados.

### 1. Búsqueda de Estado Estacionario (`find_steady_state`)
Permite resolver numéricamente el vector de estado $x_{\text{ss}}$ que equilibra el sistema (haciendo $\dot{x} = 0$) para una entrada constante y un tiempo de evaluación $t$ dados (necesario si hay componentes de transición temporal como escalones).

```python
# Encuentra el estado de equilibrio x_ss para una entrada r = 10.0 a t = 2.0 s
x_ss = sys_net.find_steady_state(u0_ext=[10.0], t=2.0)
```

### 2. Linealización Numérica (`linearize`)
Calcula mediante diferencias finitas centrales las matrices jacobianas de la red completa $(A, B, C, D)$ alrededor del punto de operación $(x_0, u_0)$:
$$\dot{\delta x} \approx A \delta x + B \delta u_{\text{ext}}$$
$$\delta y \approx C \delta x + D \delta u_{\text{ext}}$$

> [!IMPORTANT]
> Las salidas a linealizar son estrictamente aquellas mapeadas explícitamente mediante `map_output()`. Si no se definen salidas externas, el sistema se considera sin salidas y las matrices $C$ y $D$ tendrán una dimensión de filas igual a $0$.

```python
# Linealiza el sistema sobre el estado estacionario obtenido
A, B, C, D = sys_net.linearize(x_ss, u0_ext=[10.0])
```

---

## Sistemas Jerárquicos y Conexiones Avanzadas

El coordinador de la red, `InterconnectedSystem`, hereda de `SubsystemBase`. Esto habilita el soporte para diagramas de bloques complejos y jerárquicos:

### 1. Jerarquía / Anidamiento (Sistemas Jerárquicos)
Un sistema interconectado completo puede ser instanciado definiendo sus dimensiones de entrada/salida y agregarse como un subsistema dentro de otro sistema contenedor (padre):

```python
# Crear sistema hijo y padre
sys_child = InterconnectedSystem(input_dim=1, output_dim=1, name="child")
sys_parent = InterconnectedSystem(input_dim=1, output_dim=1, name="parent")

# Mapear entradas/salidas externas a bloques internos del hijo
sys_child.map_input(external_input_idx=0, target_subsystem=gain_block, target_input_idx=0)
sys_child.map_output(external_output_idx=0, source_subsystem=integrator_block, source_output_idx=0)

# Agregar el sistema hijo al padre como un bloque normal
sys_parent.add_subsystem(sys_child)
```

### 2. Sumadores Implícitos por Múltiple Conexión
Si una entrada de un subsistema recibe múltiples conexiones (ya sea desde salidas de otros subsistemas o desde entradas externas del contenedor), el solver **las suma automáticamente** sin necesidad de agregar explícitamente un bloque `Adder`:

```python
# La entrada 0 de integrator sumará automáticamente la salida de gain1 y la salida de gain2
sys_net.connect(gain1, 0, integrator, 0)
sys_net.connect(gain2, 0, integrator, 0)
```

### 3. Chequeo de Conectividad (`check_connections`)
Permite inspeccionar la red completa para identificar qué entradas y salidas de los subsistemas están correctamente conectadas y cuáles permanecen libres/desconectadas. Retorna un reporte estructurado e imprime un informe legible en consola:

```python
# Generar e imprimir reporte de conexiones
report = sys_parent.check_connections()
```

---

## Utilidad de Visualización (`dynamics_solver.plot_results`)

El paquete incluye una utilidad para graficar los resultados obtenidos de la simulación mediante `matplotlib`.

* **`plot_results(t, outputs, signals, title, xlabel, grid, figsize, show)`**:
  * **`t`**: Vector de tiempo devuelto por el solver.
  * **`outputs`**: Diccionario de salidas devuelto por el solver.
  * **`signals`**: Lista que define las señales a graficar. Admite dos estructuras:
    * **Lista Plana**: Grafica todas las señales especificadas en la misma figura.
      ```python
      # Grafica la primera salida del motor y la salida del controlador PI juntas
      plot_results(t, outputs, signals=["motor", ("controlador_pi", 0)])
      ```
    * **Lista de Listas (Subplots)**: Genera múltiples subplots alineados verticalmente.
      ```python
      # Subplot superior: motor. Subplot inferior: controlador PI y sensor.
      plot_results(t, outputs, signals=[["motor"], [("controlador_pi", 0), "sensor"]])
      ```
  * **`show`**: Si se establece en `False` (útil en entornos headless o de pruebas), la función no bloquea la interfaz de usuario con `plt.show()` y devuelve los objetos `Figure` y `Axes` para manipulación directa.
