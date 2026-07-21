import pathlib
import sys

# Añadir el directorio raíz del proyecto al path de Python para importar dynamics_solver localmente
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from dynamics_solver import (
    Escalon,
    EspacioEstados,
    InterconnectedSystem,
)

# 1. Crear el sistema de bloques
sys_net = InterconnectedSystem()

# 2. Instanciar bloques y fuentes
# Escalón de entrada: de 0.0 a 1.0 en t=1.0 s
fuente_escalon = Escalon(initial_value=0.0, final_value=1.0, step_time=1.0, name="escalon")

# Representación en Espacio de Estados de G(s) = 1 / (s^2 + 2s + 1)
# matrices:
# dx/dt = A*x + B*u
# y = C*x + D*u
A = [[-2.0, -1.0], [1.0, 0.0]]
B = [[1.0], [0.0]]
C = [[0.0, 1.0]]
D = [[0.0]]

ss_bloque = EspacioEstados(A=A, B=B, C=C, D=D, name="filtro_ss")

# 3. Agregar los bloques a la red
sys_net.add_subsystem(fuente_escalon)
sys_net.add_subsystem(ss_bloque)

# 4. Establecer conexiones
sys_net.connect(fuente_escalon, 0, ss_bloque, 0)

# 5. Resolver la Simulación
t_span = (0.0, 10.0)
t_eval = np.linspace(0.0, 10.0, 11)
t, x_global, states, outputs = sys_net.solve(t_span=t_span, t_eval=t_eval)

# 6. Mostrar y validar resultados
print("Resultados de simulación de Espacio de Estados:")
print(f"Propiedades de ss_bloque (direct_feedthrough): {ss_bloque.direct_feedthrough}")
print(f"Dimensiones de estados globales: {x_global.shape}")

print(f"\n{'t (s)':<8}{'Escalón':<10}{'Salida SS':<15}")
for i, time in enumerate(t):
    val_esc = outputs["escalon"][i, 0]
    val_ss = outputs["filtro_ss"][i, 0]
    print(f"{time:<8.1f}{val_esc:<10.2f}{val_ss:<15.4f}")

# Validaciones de aserción (deben coincidir con la respuesta del filtro de segundo orden)
assert np.allclose(outputs["filtro_ss"][:2, 0], 0.0, atol=1e-5), (
    "La salida debe ser 0 antes del escalón"
)
assert outputs["filtro_ss"][-1, 0] > 0.99, "La salida del filtro debe estar cerca de 1.0 a t=10s"
print("\n¡Las validaciones de la simulación de Espacio de Estados pasaron con éxito!")
