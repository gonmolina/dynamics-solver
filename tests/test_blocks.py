import numpy as np
import pytest

from dynamics_solver import (
    Adder,
    ConstantSource,
    Gain,
    Integrator,
    MultiplierDivider,
    StateSpace,
    StepSource,
    SubsystemBase,
    TimeFunctionSource,
    TransferFunction,
)


def test_constant_source():
    """Prueba que el bloque ConstantSource mantenga su valor de salida constante."""
    block = ConstantSource(value=3.14, name="constant_test")
    assert block.state_dim == 0
    assert block.input_dim == 0
    assert block.output_dim == 1
    assert block.direct_feedthrough is True

    # Evaluar salidas
    out = block.update_outputs(np.array([]), np.array([]))
    assert np.allclose(out, [3.14])

    deriv = block.compute_derivatives(np.array([]), np.array([]))
    assert len(deriv) == 0


def test_step_source():
    """Prueba que el bloque StepSource realice la transición correctamente en el tiempo."""
    block = StepSource(initial_value=0.0, final_value=5.0, step_time=2.0, name="step_test")
    assert block.state_dim == 0
    assert block.input_dim == 0
    assert block.output_dim == 1

    # Antes del tiempo del escalón
    block._current_time = 1.0
    out = block.update_outputs(np.array([]), np.array([]))
    assert np.allclose(out, [0.0])

    # En el tiempo del escalón
    block._current_time = 2.0
    out = block.update_outputs(np.array([]), np.array([]))
    assert np.allclose(out, [5.0])

    # Después del tiempo del escalón
    block._current_time = 3.0
    out = block.update_outputs(np.array([]), np.array([]))
    assert np.allclose(out, [5.0])


def test_time_function_source():
    """Prueba que el bloque TimeFunctionSource evalúe la función en el tiempo actual."""
    block = TimeFunctionSource(func=lambda t: t**2 + 1.0, name="func_test")
    assert block.state_dim == 0
    assert block.input_dim == 0

    block._current_time = 3.0
    out = block.update_outputs(np.array([]), np.array([]))
    assert np.allclose(out, [10.0])


def test_adder():
    """Prueba que el bloque Adder sume y reste según el patrón de signos."""
    # Sumador de 3 entradas: + + -
    block = Adder(signs="++-", name="adder_test")
    assert block.state_dim == 0
    assert block.input_dim == 3
    assert block.output_dim == 1

    u = np.array([10.0, 5.0, 3.0])
    out = block.update_outputs(np.array([]), u)
    assert np.allclose(out, [12.0])  # 10 + 5 - 3 = 12

    # Caso de error con signo inválido
    with pytest.raises(ValueError):
        invalid_block = Adder(signs="+*")
        invalid_block.update_outputs(np.array([]), np.array([1.0, 2.0]))


def test_gain():
    """Prueba que el bloque Gain multiplique por la constante de ganancia."""
    block = Gain(k=4.0, name="gain_test")
    assert block.state_dim == 0
    assert block.input_dim == 1
    assert block.output_dim == 1

    out = block.update_outputs(np.array([]), np.array([3.0]))
    assert np.allclose(out, [12.0])


def test_multiplier_divider():
    """Prueba que el bloque MultiplierDivider multiplique y divida correctamente."""
    # Multiplicador/Divisor: * / *
    block = MultiplierDivider(operations="*/*", name="mul_div_test")
    assert block.state_dim == 0
    assert block.input_dim == 3
    assert block.output_dim == 1

    u = np.array([6.0, 2.0, 5.0])
    out = block.update_outputs(np.array([]), u)
    assert np.allclose(out, [15.0])  # (6 / 2) * 5 = 15

    # Prueba de división por cero segura
    block_div = MultiplierDivider(operations="/")
    out_zero = block_div.update_outputs(np.array([]), np.array([0.0]))
    assert np.allclose(out_zero, [1e12])  # 1 / 1e-12 = 1e12

    with pytest.raises(ValueError):
        invalid_block = MultiplierDivider(operations="*+")
        invalid_block.update_outputs(np.array([]), np.array([1.0, 2.0]))


def test_transfer_function():
    """Prueba que TransferFunction configure y calcule correctamente en espacio de estados."""
    # G(s) = 1 / (s^2 + 2s + 1)
    tf = TransferFunction(num=[1], den=[1, 2, 1], name="tf_test")
    assert tf.state_dim == 2
    assert tf.input_dim == 1
    assert tf.output_dim == 1
    assert tf.direct_feedthrough is False  # Grado Num < Grado Den

    # Verificar matrices SS de tf2ss
    assert tf.A.shape == (2, 2)
    assert tf.B.shape == (2, 1)
    assert tf.C.shape == (1, 2)
    assert tf.D.shape == (1, 1)

    # Evaluar salidas para estado x=[1.0, 2.0]
    x = np.array([1.0, 2.0])
    out = tf.update_outputs(x, np.array([0.0]))
    # y = C * x (ya que direct_feedthrough es False)
    expected_y = tf.C @ x
    assert np.allclose(out, expected_y)

    # Evaluar derivadas
    u = np.array([1.5])
    deriv = tf.compute_derivatives(x, u)
    expected_dx = tf.A @ x + tf.B @ u
    assert np.allclose(deriv, expected_dx)

    # Caso estático G(s) = 3
    tf_static = TransferFunction(num=[3], den=[1], name="tf_static")
    assert tf_static.state_dim == 0
    assert tf_static.direct_feedthrough is True
    assert np.allclose(tf_static.update_outputs(np.array([]), np.array([2.0])), [6.0])


def test_integrator():
    """Prueba que el bloque Integrador compute la salida y derivadas básicas."""
    block = Integrator(x0=2.5, name="int_test")
    assert block.state_dim == 1
    assert block.input_dim == 1
    assert block.output_dim == 1
    assert block.direct_feedthrough is False
    assert np.allclose(block.x0, [2.5])

    # Salida es el estado
    out = block.update_outputs(np.array([4.0]), np.array([1.0]))
    assert np.allclose(out, [4.0])

    # Derivada es la entrada
    deriv = block.compute_derivatives(np.array([4.0]), np.array([1.5]))
    assert np.allclose(deriv, [1.5])


def test_state_space():
    """Prueba que StateSpace represente y valide sistemas dinámicos lineales."""
    A = [[-1.0, 0.0], [0.0, -2.0]]
    B = [[2.0], [1.0]]
    C = [[1.0, 1.0]]
    D = [[0.0]]

    ss = StateSpace(A, B, C, D, name="ss_test", x0=[0.5, 0.5])
    assert ss.state_dim == 2
    assert ss.input_dim == 1
    assert ss.output_dim == 1
    assert ss.direct_feedthrough is False  # D es nula

    x = np.array([2.0, 3.0])
    u = np.array([1.0])

    out = ss.update_outputs(x, u)
    assert np.allclose(out, [5.0])  # C*x + D*u = 2 + 3 + 0 = 5

    deriv = ss.compute_derivatives(x, u)
    # A*x + B*u = [-1*2 + 2*1, -2*3 + 1*1] = [0, -5]
    assert np.allclose(deriv, [0.0, -5.0])

    # Validación de dimensiones incorrectas
    with pytest.raises(ValueError):
        # A no cuadrada
        StateSpace(A=[[1.0]], B=B, C=C, D=D)
    with pytest.raises(ValueError):
        # B con filas incorrectas
        StateSpace(A=A, B=[[1.0]], C=C, D=D)
    with pytest.raises(ValueError):
        # D con dimensiones incorrectas
        StateSpace(A=A, B=B, C=C, D=[[1.0, 2.0]])


def test_naming():
    """Prueba la asignación de nombres de estados, entradas y salidas en un subsistema."""

    class DummySystem(SubsystemBase):
        def update_outputs(self, x, u):
            return np.zeros(self.output_dim)

        def compute_derivatives(self, x, u):
            return np.zeros(self.state_dim)

    sys = DummySystem(state_dim=2, input_dim=3, output_dim=1, name="mi_sistema")

    # Valores por defecto basados en el nombre
    assert sys.state_names == ["x_mi_sistema_1", "x_mi_sistema_2"]
    assert sys.input_names == ["u_mi_sistema_1", "u_mi_sistema_2", "u_mi_sistema_3"]
    assert sys.output_names == ["y_mi_sistema_1"]

    # Si se cambia el nombre del sistema, los nombres por defecto cambian
    sys.name = "nuevo_nombre"
    assert sys.state_names == ["x_nuevo_nombre_1", "x_nuevo_nombre_2"]
    assert sys.input_names == ["u_nuevo_nombre_1", "u_nuevo_nombre_2", "u_nuevo_nombre_3"]
    assert sys.output_names == ["y_nuevo_nombre_1"]

    sys.set_name("otro_nombre")
    assert sys.state_names == ["x_otro_nombre_1", "x_otro_nombre_2"]

    # Definir nombres personalizados
    sys.set_state_name(0, "custom_x1")
    sys.set_input_name(1, "custom_u2")
    sys.set_output_names(["custom_y1"])

    # Al cambiar de nombre el sistema, los nombres personalizados se preservan, pero los por defecto cambian
    sys.name = "final_nombre"
    assert sys.state_names == ["custom_x1", "x_final_nombre_2"]
    assert sys.input_names == ["u_final_nombre_1", "custom_u2", "u_final_nombre_3"]
    assert sys.output_names == ["custom_y1"]
