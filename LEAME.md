# Dynamics Solver 🚀

[![Python Version](https://img.shields.io/badge/python->=%203.10-blue.svg)](https://www.python.org/)
[![Development Status](https://img.shields.io/badge/status-work%20in%20progress-orange.svg)]()
[![Built with AI](https://img.shields.io/badge/built%20with-100%25%20AI-purple.svg)]()
[![PyPI](https://img.shields.io/pypi/v/dynamics-solver.svg)](https://pypi.org/project/dynamics-solver/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

🌐 **Leer en otros idiomas:** [English](README.md)

**Dynamics Solver** es una biblioteca en Python diseñada para el modelado, interconexión y simulación de sistemas dinámicos no lineales complejos y diagramas de bloques de control.

> ⚠️ **AVISO IMPORTANTE / DISCLAIMER**  
> Este proyecto ha sido **desarrollado al 100% utilizando herramientas de Inteligencia Artificial** (AI-driven development) y se encuentra en **fase activa de desarrollo (Work in Progress / Alpha)**. Las APIs y funcionalidades pueden evolucionar o modificarse a medida que el proyecto madure.

---

## ✨ Características Principales

* **Arquitectura Desacoplada e Interconectada**: 
  * Los subsistemas (`SubsystemBase`) son modelos matemáticos puros e independientes que no guardan referencias a otros bloques ni a la red contenedora.
  * Toda la topología de red se gestiona de forma centralizada mediante `InterconnectedSystem.connect(...)`.

* **Ordenamiento Topológico Automático**:
  * Resuelve la secuencia de ejecución de los bloques en cada paso de simulación mediante el Algoritmo de Kahn.
  * Detecta automáticamente lazos algebraicos instantáneos e identifica inconsistencias en bloques con paso directo (`direct_feedthrough`).

* **Análisis de Estado Estacionario y Linealización**:
  * `find_steady_state(...)`: Calcula el vector de estado en equilibrio $x_{ss}$ dadas entradas constantes u operacionales.
  * `linearize(...)`: Obtiene numéricamente las matrices jacobianas del sistema linealizado en espacio de estados ($A, B, C, D$) entre las entradas y salidas externas configuradas.

* **Biblioteca de Bloques Básicos y Dinámicos**:
  * **Dinámicos**: Integrador continuo, Funciones de Transferencia $G(s)$, Sistemas en Espacio de Estados ($A, B, C, D$).
  * **Operadores**: Sumador/Restador multientrada (con sumadores implícitos en entradas compuestas), Multiplicador/Divisor.
  * **Fuentes de Entrada**: Fuentes Constantes, Escalón y Funciones Temporales genéricas $u(t)$.

* **Gestión Flexible de Nombres (Estados, Entradas y Salidas)**:
  * Generación automática de etiquetas basadas en la jerarquía: $x_{\text{sistema}\_1}$, $u_{\text{sistema}\_1}$, $y_{\text{sistema}\_1}$.
  * Permite asignar nombres personalizados por índice o en lote (`set_state_name`, `set_input_names`, etc.), los cuales se preservan automáticamente incluso si se renombra el subsistema contenedor.

* **Visualización de Resultados Integrada**:
  * Herramienta `plot_results(...)` basada en `matplotlib` para la generación de figuras compuestas multi-subplot claras para entradas y salidas.

---

## 📦 Instalación

Puedes instalar la biblioteca directamente desde **PyPI**:

```bash
pip install dynamics-solver
```

---

## ⚡ Ejemplo Rápido de Uso

A continuación se muestra cómo conectar un integrador con una ganancia en lazo abierto:

```python
import numpy as np
from dynamics_solver import InterconnectedSystem, SubsystemBase

# 1. Definir bloques personalizados o estándar
class Integrador(SubsystemBase):
    def __init__(self, name="integrador", x0=None):
        super().__init__(state_dim=1, input_dim=1, output_dim=1, name=name, direct_feedthrough=False, x0=x0)

    def update_outputs(self, x, u):
        return x

    def compute_derivatives(self, x, u):
        return u

class Ganancia(SubsystemBase):
    def __init__(self, k=2.0, name="ganancia"):
        self.k = k
        super().__init__(state_dim=0, input_dim=1, output_dim=1, name=name, direct_feedthrough=True)

    def update_outputs(self, x, u):
        return self.k * u

    def compute_derivatives(self, x, u):
        return np.array([])

# 2. Crear el sistema interconectado y agregar bloques
sys_net = InterconnectedSystem()
int1 = Integrador("int1", x0=[1.0])
gain1 = Ganancia(k=3.0, name="gain1")

sys_net.add_subsystem(int1)
sys_net.add_subsystem(gain1)

# 3. Conectar la salida del integrador a la entrada de la ganancia
sys_net.connect(int1, 0, gain1, 0)

# 4. Simular la red
t, x_global, states, outputs = sys_net.solve(
    t_span=(0.0, 5.0),
    u_ext_func=lambda t: np.array([2.0])
)

print("Salida de la ganancia a t=5s:", outputs["gain1"][-1])
```

---

## 📚 Ejemplos Avanzados y Documentación

Para ver casos de aplicación más avanzados, consulta los cuadernos y ejemplos incluidos en el repositorio:

* **Control de Reactor Nuclear**: [`ejemplos/control_reactores.py`](ejemplos/control_reactores.py) y [`ejemplos/control_reactores.md`](ejemplos/control_reactores.md)
  * Modela la cinética puntual de neutrones con precursores, la transferencia térmica en el refrigerante y la realimentación por reactividad térmica.
  * Incluye la simulación de transitorios y la linealización del reactor alrededor del estado estacionario.
* **Documentación Completa de Simulación**: [`docs/simulation_docs.md`](docs/simulation_docs.md)

---

## 🛠️ Estado del Desarrollo (AI-Generated & WIP)

Este módulo ha sido concebido y generado íntegramente mediante modelos de lenguaje e Inteligencia Artificial como una exploración avanzada de ingeniería de software para simulación dinámica. 

Si encuentras algún problema o tienes sugerencias para la arquitectura, ¡las contribuciones e incidencias (issues) son más que bienvenidas!

---

## 📄 Licencia

Este proyecto está distribuido bajo la licencia [MIT](LICENSE).
