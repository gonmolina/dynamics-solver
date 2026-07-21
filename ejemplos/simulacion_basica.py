import pathlib
import sys

# Añadir el directorio raíz del proyecto al path de Python para importar dynamics_solver localmente
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from dynamics_solver import InterconnectedSystem, SubsystemBase


# 1. Definir los Bloques del Diagrama
class Integrator(SubsystemBase):
    def __init__(self, name="integrator", x0=None):
        super().__init__(
            state_dim=1,
            input_dim=1,
            output_dim=1,
            name=name,
            direct_feedthrough=False,
            x0=x0,
        )

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


# 2. Inicializar y Conectar los Bloques
sys_net = InterconnectedSystem()

# Se define el estado inicial x0 = 2.0 por defecto en la inicialización del Integrador
int1 = Integrator("int1", x0=[2.0])
gain1 = Gain(3.0, "gain1")

sys_net.add_subsystem(int1)
sys_net.add_subsystem(gain1)

# Conectar la entrada de la ganancia a la salida del integrador
sys_net.connect(int1, 0, gain1, 0)

# 3. Resolver / Simular
# Entrada externa: u_ext(t) = 4.0 (Constante)
# Al no pasar x0_global, se construirá automáticamente usando los x0 por defecto (en este caso [2.0])
t, x_global, states, outputs = sys_net.solve(t_span=(0, 2), u_ext_func=lambda t: np.array([4.0]))

# 4. Mostrar Resultados de la Simulación
print("Tiempos de simulación:")
print(t)
print("\nHistorial de estados locales ('int1'):")
print(states["int1"].flatten())
print("\nHistorial de salidas locales ('gain1'):")
print(outputs["gain1"].flatten())
