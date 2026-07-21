import pathlib
import sys

# Añadir el directorio raíz del proyecto al path de Python para importar dynamics_solver localmente
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from dynamics_solver import (
    Escalon,
    FuncionTransferencia,
    Integrador,
    InterconnectedSystem,
)

# 1. Crear el sistema de bloques
sys_net = InterconnectedSystem()

# 2. Instanciar bloques dinámicos y fuentes
# Escalón de entrada: de 0.0 a 1.0 en t=1.0 s
fuente_escalon = Escalon(initial_value=0.0, final_value=1.0, step_time=1.0, name="escalon")

# Función de transferencia: G(s) = 1 / (s^2 + 2s + 1)
# Coeficientes: num=[1], den=[1, 2, 1]
# Grado del numerador M=0 < grado del denominador N=2 -> direct_feedthrough debe ser False automáticamente.
tf_bloque = FuncionTransferencia(num=[1], den=[1, 2, 1], name="filtro_2do_orden")

# Integrador continuo (1/s)
integrador = Integrador(x0=0.0, name="integrador")

# 3. Agregar los bloques a la red
sys_net.add_subsystem(fuente_escalon)
sys_net.add_subsystem(tf_bloque)
sys_net.add_subsystem(integrador)

# 4. Establecer conexiones
# Conectar la fuente al bloque de la función de transferencia
sys_net.connect(fuente_escalon, 0, tf_bloque, 0)

# Conectar la salida del filtro a la entrada del integrador
sys_net.connect(tf_bloque, 0, integrador, 0)

# 5. Resolver la Simulación
t_span = (0.0, 10.0)
# Evaluamos en pasos de tiempo fijos para inspeccionar los valores
t_eval = np.linspace(0.0, 10.0, 11)
t, x_global, states, outputs = sys_net.solve(t_span=t_span, t_eval=t_eval)

# 6. Mostrar y validar resultados
print("Resultados de simulación de bloques dinámicos:")
print(f"Propiedades de tf_bloque (direct_feedthrough): {tf_bloque.direct_feedthrough}")
print(f"Dimensiones de estados globales: {x_global.shape}")

# Verificaciones físicas
# - Antes de t=1s, todos los estados y salidas deben ser 0.
# - El filtro es G(s) = 1 / (s+1)^2. Su respuesta al escalón unitario u(t-1) es:
#   y_tf(t) = 1 - e^-(t-1) - (t-1)*e^-(t-1)  para t >= 1.
# - A t=10s (t-1 = 9s), y_tf debe estar muy cerca de 1.0 (estado estacionario).
# - El integrador acumula y_tf.

print(f"\n{'t (s)':<8}{'Escalón':<10}{'Salida TF':<15}{'Salida Integrador':<20}")
for i, time in enumerate(t):
    val_esc = outputs["escalon"][i, 0]
    val_tf = outputs["filtro_2do_orden"][i, 0]
    val_int = outputs["integrador"][i, 0]
    print(f"{time:<8.1f}{val_esc:<10.2f}{val_tf:<15.4f}{val_int:<20.4f}")

# Validaciones de aserción
assert np.allclose(outputs["filtro_2do_orden"][:2, 0], 0.0, atol=1e-5), (
    "La salida debe ser 0 antes del escalón"
)
assert outputs["filtro_2do_orden"][-1, 0] > 0.99, (
    "La salida del filtro debe estar cerca de 1.0 a t=10s"
)
print("\n¡Las validaciones de la simulación de bloques dinámicos pasaron con éxito!")
