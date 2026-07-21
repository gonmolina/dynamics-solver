import pathlib
import sys

# Añadir el directorio raíz del proyecto al path de Python para importar dynamics_solver localmente
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from dynamics_solver import (
    Constante,
    Escalon,
    FuncionTemporal,
    Ganancia,
    InterconnectedSystem,
    MultiplicadorDivisor,
    SumadorRestador,
)

# 1. Crear el sistema de bloques
sys_net = InterconnectedSystem()

# 2. Instanciar Bloques Básicos
# Fuentes de entrada
fuente_constante = Constante(value=10.0, name="constante_10")
fuente_escalon = Escalon(initial_value=0.0, final_value=5.0, step_time=2.0, name="escalon_5")
fuente_seno = FuncionTemporal(func=lambda t: 2.0 * np.sin(t), name="seno_2t")

# Operaciones matemáticas
sumador = SumadorRestador(signs="+++", name="sumador")  # Suma de 3 entradas
ganancia = Ganancia(k=0.5, name="mitad")  # Multiplica por 0.5
multiplicador = MultiplicadorDivisor(operations="*/", name="mul_div")  # Multiplica y luego divide

# 3. Agregar los bloques al sistema
sys_net.add_subsystem(fuente_constante)
sys_net.add_subsystem(fuente_escalon)
sys_net.add_subsystem(fuente_seno)
sys_net.add_subsystem(sumador)
sys_net.add_subsystem(ganancia)
sys_net.add_subsystem(multiplicador)

# 4. Conectar los bloques
# Conectar las 3 fuentes al sumador
sys_net.connect(fuente_constante, 0, sumador, 0)
sys_net.connect(fuente_escalon, 0, sumador, 1)
sys_net.connect(fuente_seno, 0, sumador, 2)

# Conectar la salida del sumador a la ganancia
sys_net.connect(sumador, 0, ganancia, 0)

# Conectar la ganancia al multiplicador (u[0])
sys_net.connect(ganancia, 0, multiplicador, 0)

# Para u[1], multiplicador se divide por la constante_10
sys_net.connect(fuente_constante, 0, multiplicador, 1)

# El cálculo final modelado es:
# Y = ( (10.0 + escalon(t) + 2.0*sin(t)) * 0.5 ) / 10.0

# 5. Resolver la Simulación
t_span = (0.0, 5.0)
t, x_global, states, outputs = sys_net.solve(t_span=t_span, t_eval=np.linspace(0.0, 5.0, 6))

# 6. Mostrar resultados
print("Resultados de simulación de bloques básicos:")
# Imprimimos puntos representativos en t=0.0, t=1.0, t=3.0, t=5.0
for ti in [0.0, 1.0, 3.0, 5.0]:
    idx = np.argmin(np.abs(t - ti))
    t_val = t[idx]
    c_val = outputs["constante_10"][idx, 0]
    e_val = outputs["escalon_5"][idx, 0]
    s_val = outputs["seno_2t"][idx, 0]
    sum_val = outputs["sumador"][idx, 0]
    g_val = outputs["mitad"][idx, 0]
    md_val = outputs["mul_div"][idx, 0]

    print(f"\nt = {t_val:.1f} s:")
    print(f"  Constante: {c_val:.2f}")
    print(f"  Escalón:   {e_val:.2f}")
    print(f"  Seno:      {s_val:.2f}")
    print(f"  Suma:      {sum_val:.2f} (Esperado: {c_val + e_val + s_val:.2f})")
    print(f"  Ganancia:  {g_val:.2f} (Esperado: {sum_val * 0.5:.2f})")
    print(f"  Mul/Div:   {md_val:.2f} (Esperado: {g_val / c_val:.2f})")

# Validaciones básicas
assert np.allclose(
    outputs["sumador"][:, 0],
    outputs["constante_10"][:, 0] + outputs["escalon_5"][:, 0] + outputs["seno_2t"][:, 0],
)
assert np.allclose(outputs["mitad"][:, 0], outputs["sumador"][:, 0] * 0.5)
assert np.allclose(outputs["mul_div"][:, 0], outputs["mitad"][:, 0] / outputs["constante_10"][:, 0])
print("\n¡Todas las validaciones matemáticas del diagrama de bloques pasaron con éxito!")
