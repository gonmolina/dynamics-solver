import numpy as np

from dynamics_solver import (
    Escalon,
    EspacioEstados,
    FuncionTransferencia,
    Gain,
    InterconnectedSystem,
    SumadorRestador,
)


def test_dc_motor_closed_loop():
    """Prueba de integración: Control de velocidad en lazo cerrado de un motor de CC.

    Lazo:
      Ref (Escalón) -> [+] -> Error -> [ Controlador PI (TF) ] -> Va -> [ Motor CC (SS) ] -> w
                        ^                                                                     |
                        |-----------------[ Sensor (TF) ]<-------------------------------------
    """
    sys_net = InterconnectedSystem()

    # 1. Entrada de referencia: Escalón de 0.0 a 10.0 rad/s a t = 1.0 s
    ref = Escalon(initial_value=0.0, final_value=10.0, step_time=1.0, name="referencia")

    # 2. Sumador de error: e(t) = r(t) - y_sensor(t)
    sum_error = SumadorRestador(signs="+-", name="sum_error")

    # 3. Controlador PI: C(s) = Kp + Ki/s = (Kp*s + Ki) / s
    # Elegimos Kp = 0.2, Ki = 0.5
    # num = [0.2, 0.5], den = [1.0, 0.0]
    pi_ctrl = FuncionTransferencia(num=[0.2, 0.5], den=[1.0, 0.0], name="controlador_pi")

    # 4. Motor de CC en Espacio de Estados (Armadura controlada)
    # Estados: [i_a (corriente), w (velocidad angular)]
    # dx/dt = A_m * x + B_m * Va
    # w = C_m * x + D_m * Va
    R_a = 1.0
    L_a = 0.5
    K_t = 0.1
    K_b = 0.1
    J = 0.02
    B = 0.01

    A_m = [[-R_a / L_a, -K_b / L_a], [K_t / J, -B / J]]
    B_m = [[1.0 / L_a], [0.0]]
    C_m = [[0.0, 1.0]]
    D_m = [[0.0]]

    motor = EspacioEstados(A=A_m, B=B_m, C=C_m, D=D_m, name="motor_cc")

    # 5. Sensor de velocidad de primer orden (Filtro pasa-bajos)
    # S(s) = 1 / (tau*s + 1) con tau = 0.1 s
    # num = [1.0], den = [0.1, 1.0]
    sensor = FuncionTransferencia(num=[1.0], den=[0.1, 1.0], name="sensor")

    # Agregar todos los subsistemas a la red
    sys_net.add_subsystem(ref)
    sys_net.add_subsystem(sum_error)
    sys_net.add_subsystem(pi_ctrl)
    sys_net.add_subsystem(motor)
    sys_net.add_subsystem(sensor)

    # Conectar el lazo
    sum_error.connect_input(0, ref, 0)  # Error input 0 = referencia r(t)
    sum_error.connect_input(1, sensor, 0)  # Error input 1 = salida del sensor y_s(t)

    pi_ctrl.connect_input(0, sum_error, 0)  # Entrada del PI = error e(t)

    motor.connect_input(0, pi_ctrl, 0)  # Entrada del motor (Va) = salida del PI

    sensor.connect_input(0, motor, 0)  # Entrada del sensor = velocidad w(t) del motor

    # 6. Resolver la Simulación
    # Simular durante 30.0 segundos para dar tiempo al lazo a estabilizarse
    t_span = (0.0, 30.0)
    t_eval = np.linspace(0.0, 30.0, 301)
    t, x_global, states, outputs = sys_net.solve(t_span=t_span, t_eval=t_eval)

    # 7. Asertar y validar respuestas físicas esperadas
    # w inicial debe ser 0.0 (antes de t=1.0 s)
    idx_before = np.where(t < 1.0)[0]
    assert np.allclose(outputs["motor_cc"][idx_before, 0], 0.0, atol=1e-4)

    # Al final de la simulación (t=30s), la velocidad w debe estar cerca del valor de consigna r = 10.0 rad/s
    # debido a la acción integral del PI que cancela el error permanente.
    w_final = outputs["motor_cc"][-1, 0]
    print(f"\nVelocidad final alcanzada en lazo cerrado: {w_final:.4f} rad/s")
    assert np.isclose(w_final, 10.0, rtol=1e-2)


def test_find_steady_state_dc_motor():
    """Valida la búsqueda numérica del estado estacionario del lazo cerrado del motor de CC."""
    sys_net = InterconnectedSystem()

    # Bloques idénticos al lazo de control
    ref = Escalon(initial_value=0.0, final_value=10.0, step_time=1.0, name="referencia")
    sum_error = SumadorRestador(signs="+-", name="sum_error")
    pi_ctrl = FuncionTransferencia(num=[0.2, 0.5], den=[1.0, 0.0], name="controlador_pi")

    A_m = [[-2.0, -0.2], [5.0, -0.5]]
    B_m = [[2.0], [0.0]]
    C_m = [[0.0, 1.0]]
    D_m = [[0.0]]
    motor = EspacioEstados(A=A_m, B=B_m, C=C_m, D=D_m, name="motor_cc")

    sensor = FuncionTransferencia(num=[1.0], den=[0.1, 1.0], name="sensor")

    sys_net.add_subsystem(ref)
    sys_net.add_subsystem(sum_error)
    sys_net.add_subsystem(pi_ctrl)
    sys_net.add_subsystem(motor)
    sys_net.add_subsystem(sensor)

    sum_error.connect_input(0, ref, 0)
    sum_error.connect_input(1, sensor, 0)
    pi_ctrl.connect_input(0, sum_error, 0)
    motor.connect_input(0, pi_ctrl, 0)
    sensor.connect_input(0, motor, 0)

    # Buscar estado estacionario para referencia constante = 10.0
    # Entrada externa es un vector de dimensión 1 (la referencia que lee el escalón en t > step_time,
    # que en find_steady_state se evalúa con u_ext constante).
    x_ss = sys_net.find_steady_state(u=[10.0], t=2.0)

    # Desglosar los estados obtenidos del vector global
    # Estado global es la concatenación de estados de los bloques dinámicos:
    # - controlador_pi (1 estado)
    # - motor_cc (2 estados)
    # - sensor (1 estado)
    # Total estados = 4
    assert len(x_ss) == 4

    # Extraer valores locales
    x_pi = sys_net._extract_local_state(x_ss, "controlador_pi")
    x_mot = sys_net._extract_local_state(x_ss, "motor_cc")
    x_sens = sys_net._extract_local_state(x_ss, "sensor")

    # Valores teóricos esperados en estado estacionario para w = 10.0 rad/s:
    # - w_ss = 10.0 => x_mot[1] = 10.0
    # - sensor: y_sens = 10 * x_sens = 10.0 => x_sens[0] = 1.0
    # - corriente de armadura i_a: dx_mot/dt = 0 => 5 * i_a - 0.5 * w = 0 => i_a = 0.5 * 10 / 5 = 1.0 A => x_mot[0] = 1.0
    # - Va: dx_mot/dt = 0 => -2 * i_a - 0.2 * w + 2 * Va = 0 => -2 - 2 + 2 * Va = 0 => Va = 2.0 V
    # - controlador PI: Va = 0.5 * x_pi + 0.2 * error. Como error = 0 en estacionario => Va = 0.5 * x_pi => x_pi[0] = Va / 0.5 = 4.0
    assert np.isclose(x_mot[1], 10.0, atol=1e-5)
    assert np.isclose(x_sens[0], 1.0, atol=1e-5)
    assert np.isclose(x_mot[0], 1.0, atol=1e-5)
    assert np.isclose(x_pi[0], 4.0, atol=1e-5)
    print("\n¡Búsqueda numérica de estado estacionario verificada con éxito teórico!")


def test_hierarchical_and_summing():
    """Valida el soporte de sistemas interconectados jerárquicos (anidados) y la suma automática de múltiples conexiones."""
    # 1. Crear sistema padre (input_dim=1, output_dim=1)
    sys_parent = InterconnectedSystem(input_dim=1, output_dim=1, name="parent")

    # 2. Bloques del padre
    from dynamics_solver.blocks import Ganancia

    gain1 = Ganancia(k=2.0, name="gain1")
    gain3 = Ganancia(k=5.0, name="gain3")

    # 3. Crear sistema hijo (input_dim=1, output_dim=1)
    sys_child = InterconnectedSystem(input_dim=1, output_dim=1, name="child")

    # 4. Bloques del hijo
    from dynamics_solver.blocks import Integrador

    gain2 = Ganancia(k=3.0, name="gain2")
    integrator = Integrador(x0=0.0, name="integrador")

    # 5. Agregar bloques al hijo y establecer sus mapeos y conexiones
    sys_child.add_subsystem(gain2)
    sys_child.add_subsystem(integrator)

    # Mapear entrada externa del hijo al gain2
    sys_child.map_input(0, gain2, 0)

    # Conectar integrator a gain2 dos veces (esto debería duplicar el valor por suma automática)
    integrator.connect_input(0, gain2, 0)
    integrator.connect_input(0, gain2, 0)

    # Mapear salida del integrator a la salida externa del hijo
    sys_child.map_output(0, integrator, 0)

    # 6. Agregar bloques al padre y establecer conexiones
    sys_parent.add_subsystem(gain1)
    sys_parent.add_subsystem(sys_child)
    sys_parent.add_subsystem(gain3)

    # Mapear entrada externa del padre a gain1
    sys_parent.map_input(0, gain1, 0)

    # Conectar el sistema hijo a la salida de gain1
    sys_child.connect_input(0, gain1, 0)

    # Conectar gain3 a la salida del sistema hijo
    gain3.connect_input(0, sys_child, 0)

    # Mapear salida de gain3 a la salida externa del padre
    sys_parent.map_output(0, gain3, 0)

    # 7. Ejecutar chequeo de conexiones
    report = sys_parent.check_connections()
    assert len(report["unconnected_inputs"]) == 0
    assert len(report["unconnected_outputs"]) == 0

    # 8. Simular
    # Entrada constante r(t) = 1.0
    def u_ext_func(t: float) -> np.ndarray:
        return np.array([1.0])

    t_span = (0.0, 2.0)
    t, x, states, outputs = sys_parent.solve(t_span=t_span, u_ext_func=u_ext_func)

    # 9. Validar matemáticas:
    # r = 1.0
    # gain1_out = 2.0 * r = 2.0
    # sys_child_in = 2.0
    # gain2_out = 3.0 * sys_child_in = 6.0
    # integrator_in = gain2_out + gain2_out = 12.0 (por doble conexión)
    # integrator_state = 12.0 * t (para t=2.0 => x = 24.0)
    # gain3_out = 5.0 * integrator_state = 60.0 * t (para t=2.0 => y = 120.0)

    # Validar dimensiones de estado global
    # Integrador tiene 1 estado. El hijo reporta 1 estado total.
    # El padre reporta 1 estado total (el del hijo).
    assert sys_parent.state_dim == 1
    assert integrator.state_dim == 1

    # Validar valores finales
    x_final = states["child"][-1]
    assert np.isclose(x_final[0], 24.0, rtol=1e-3)

    # Salida externa del padre en t=2.0
    y_final = outputs["gain3"][-1, 0]
    assert np.isclose(y_final, 120.0, rtol=1e-3)
    print("\n¡Simulación jerárquica y suma de conexiones validadas con éxito!")

    # 10. Validar linealización entrada-salida sobre el punto de operación
    # Si linealizamos sys_parent alrededor de x0=[0.0] y u0=[1.0]:
    # dx/dt = 0*x + 12*u => A = [[0]], B = [[12]]
    # y = 5*x + 0*u => C = [[5]], D = [[0]]
    A, B, C, D = sys_parent.linearize(x0=[0.0], u0=[1.0])

    assert A.shape == (1, 1)
    assert B.shape == (1, 1)
    assert C.shape == (1, 1)
    assert D.shape == (1, 1)

    assert np.isclose(A[0, 0], 0.0, atol=1e-5)
    assert np.isclose(B[0, 0], 12.0, atol=1e-5)
    assert np.isclose(C[0, 0], 5.0, atol=1e-5)
    assert np.isclose(D[0, 0], 0.0, atol=1e-5)
    print("\n¡Linealización entrada-salida jerárquica validada con éxito!")


def test_execution_order_computed_once():
    """Verifica que el orden de ejecución se calcule una sola vez antes de evaluar salidas."""
    sys_net = InterconnectedSystem()
    g1 = Gain(2.0, "g1")
    g2 = Gain(3.0, "g2")
    sys_net.add_subsystem(g1)
    sys_net.add_subsystem(g2)
    sys_net.connect(g1, 0, g2, 0)

    # Antes de evaluar, ordered_subsystems está vacío
    assert len(sys_net.ordered_subsystems) == 0

    # Al evaluar por primera vez, se calcula el orden topológico
    sys_net.update_outputs(np.zeros(0), np.array([1.0]))
    assert len(sys_net.ordered_subsystems) == 2

    first_order = sys_net.ordered_subsystems

    # En evaluaciones subsiguientes, se reutiliza el mismo orden calculado sin recalcular
    sys_net.update_outputs(np.zeros(0), np.array([2.0]))
    assert sys_net.ordered_subsystems is first_order
