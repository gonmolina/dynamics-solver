"""Librería de bloques y subsistemas básicos para dynamics_solver.

Contiene bloques de entrada (fuentes) y bloques matemáticos comunes.
"""

from typing import Callable, Sequence

import numpy as np
from scipy.signal import tf2ss

from .core import SubsystemBase


class ConstantSource(SubsystemBase):
    """Bloque de entrada constante."""

    def __init__(self, value: float, name: str = "constant"):
        super().__init__(state_dim=0, input_dim=0, output_dim=1, name=name, direct_feedthrough=True)
        self.value: float = value

    def update_outputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([self.value])

    def compute_derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([])


class StepSource(SubsystemBase):
    """Bloque de entrada tipo escalón."""

    def __init__(
        self,
        initial_value: float,
        final_value: float,
        step_time: float,
        name: str = "step",
    ):
        super().__init__(state_dim=0, input_dim=0, output_dim=1, name=name, direct_feedthrough=True)
        self.initial_value: float = initial_value
        self.final_value: float = final_value
        self.step_time: float = step_time

    def update_outputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        val = self.final_value if self._current_time >= self.step_time else self.initial_value
        return np.array([val])

    def compute_derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([])


class TimeFunctionSource(SubsystemBase):
    """Bloque de entrada definido por una función del tiempo."""

    def __init__(self, func: Callable[[float], float], name: str = "time_function"):
        super().__init__(state_dim=0, input_dim=0, output_dim=1, name=name, direct_feedthrough=True)
        self.func: Callable[[float], float] = func

    def update_outputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([self.func(self._current_time)])

    def compute_derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([])


class Adder(SubsystemBase):
    """Bloque sumador / restador de N entradas."""

    def __init__(self, signs: str = "++", name: str = "adder"):
        super().__init__(
            state_dim=0,
            input_dim=len(signs),
            output_dim=1,
            name=name,
            direct_feedthrough=True,
        )
        self.signs: str = signs

    def update_outputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        val = 0.0
        for idx, sign in enumerate(self.signs):
            if sign == "+":
                val += u[idx]
            elif sign == "-":
                val -= u[idx]
            else:
                raise ValueError(f"Signo no válido '{sign}' en el bloque Adder '{self.name}'.")
        return np.array([val])

    def compute_derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([])


class Gain(SubsystemBase):
    """Bloque de ganancia constante."""

    def __init__(self, k: float, name: str = "gain"):
        super().__init__(state_dim=0, input_dim=1, output_dim=1, name=name, direct_feedthrough=True)
        self.k: float = k

    def update_outputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([self.k * u[0]])

    def compute_derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([])


class MultiplierDivider(SubsystemBase):
    """Bloque multiplicador / divisor de N entradas."""

    def __init__(self, operations: str = "*/", name: str = "mul_div"):
        super().__init__(
            state_dim=0,
            input_dim=len(operations),
            output_dim=1,
            name=name,
            direct_feedthrough=True,
        )
        self.operations: str = operations

    def update_outputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        val = 1.0
        for idx, op in enumerate(self.operations):
            if op == "*":
                val *= u[idx]
            elif op == "/":
                divisor = u[idx]
                if abs(divisor) < 1e-12:
                    divisor = 1e-12 if divisor >= 0 else -1e-12
                val /= divisor
            else:
                raise ValueError(
                    f"Operación no válida '{op}' en el bloque MultiplierDivider '{self.name}'."
                )
        return np.array([val])

    def compute_derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([])


# Alias en español para facilidad de uso y compatibilidad directa
Constante = ConstantSource
Escalon = StepSource
FuncionTemporal = TimeFunctionSource
SumadorRestador = Adder
Ganancia = Gain
MultiplicadorDivisor = MultiplierDivider


class TransferFunction(SubsystemBase):
    """Representa una función de transferencia continua G(s) = Num(s) / Den(s).

    Detecta automáticamente si el sistema tiene paso directo (direct feedthrough)
    comparando los grados de los polinomios numerador y denominador.
    """

    def __init__(
        self,
        num: Sequence[float],
        den: Sequence[float],
        name: str = "tf",
        x0: np.ndarray | Sequence[float] | None = None,
    ):
        num_arr = np.trim_zeros(np.asarray(num, dtype=float), "f")
        den_arr = np.trim_zeros(np.asarray(den, dtype=float), "f")

        if len(den_arr) == 0:
            raise ValueError("El polinomio denominador no puede ser nulo.")

        if len(num_arr) > len(den_arr):
            raise ValueError(
                "La función de transferencia no es propia (grado del numerador > grado del denominador)."
            )

        # direct_feedthrough es True si el grado del numerador es igual al grado del denominador
        df = len(num_arr) == len(den_arr)
        state_dim = len(den_arr) - 1

        super().__init__(
            state_dim=state_dim,
            input_dim=1,
            output_dim=1,
            name=name,
            direct_feedthrough=df,
            x0=x0,
        )

        self.num: np.ndarray = num_arr
        self.den: np.ndarray = den_arr

        if state_dim > 0:
            self.A, self.B, self.C, self.D = tf2ss(self.num, self.den)
        else:
            self.A = np.zeros((0, 0))
            self.B = np.zeros((0, 1))
            self.C = np.zeros((1, 0))
            self.D = np.array([[self.num[0] / self.den[0]]])

    def update_outputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        if self.state_dim > 0:
            if self.direct_feedthrough and len(u) > 0:
                return self.C @ x + self.D @ u
            return self.C @ x
        return self.D @ u

    def compute_derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        if self.state_dim > 0:
            return self.A @ x + self.B @ u
        return np.array([])


class Integrator(SubsystemBase):
    """Bloque integrador continuo (1/s)."""

    def __init__(self, x0: float = 0.0, name: str = "integrator"):
        super().__init__(
            state_dim=1,
            input_dim=1,
            output_dim=1,
            name=name,
            direct_feedthrough=False,
            x0=[x0],
        )

    def update_outputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([x[0]])

    def compute_derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([u[0]])


# Alias en español adicionales para los nuevos bloques dinámicos
FuncionTransferencia = TransferFunction
Integrador = Integrator


class StateSpace(SubsystemBase):
    """Representa un subsistema lineal continuo en espacio de estados.

    Ecuaciones:
        dx/dt = A * x + B * u
        y     = C * x + D * u
    """

    def __init__(
        self,
        A: np.ndarray | Sequence[Sequence[float]],
        B: np.ndarray | Sequence[Sequence[float]],
        C: np.ndarray | Sequence[Sequence[float]],
        D: np.ndarray | Sequence[Sequence[float]],
        name: str = "state_space",
        x0: np.ndarray | Sequence[float] | None = None,
    ):
        A_arr = np.asarray(A, dtype=float)
        B_arr = np.asarray(B, dtype=float)
        C_arr = np.asarray(C, dtype=float)
        D_arr = np.asarray(D, dtype=float)

        # Validación de dimensiones
        if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
            raise ValueError("La matriz A debe ser una matriz cuadrada de 2D.")

        state_dim = A_arr.shape[0]

        if B_arr.ndim != 2 or B_arr.shape[0] != state_dim:
            raise ValueError(
                f"La matriz B debe tener dimensiones (state_dim={state_dim}, input_dim)."
            )

        input_dim = B_arr.shape[1]

        if C_arr.ndim != 2 or C_arr.shape[1] != state_dim:
            raise ValueError(
                f"La matriz C debe tener dimensiones (output_dim, state_dim={state_dim})."
            )

        output_dim = C_arr.shape[0]

        if D_arr.ndim != 2 or D_arr.shape[0] != output_dim or D_arr.shape[1] != input_dim:
            raise ValueError(
                f"La matriz D debe tener dimensiones (output_dim={output_dim}, input_dim={input_dim})."
            )

        # Detecta automáticamente si tiene paso directo buscando elementos no nulos en D
        df = not np.allclose(D_arr, 0.0)

        super().__init__(
            state_dim=state_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            name=name,
            direct_feedthrough=df,
            x0=x0,
        )

        self.A: np.ndarray = A_arr
        self.B: np.ndarray = B_arr
        self.C: np.ndarray = C_arr
        self.D: np.ndarray = D_arr

    def update_outputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        if self.state_dim > 0:
            if self.direct_feedthrough and len(u) > 0:
                return self.C @ x + self.D @ u
            return self.C @ x
        return self.D @ u

    def compute_derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        if self.state_dim > 0:
            return self.A @ x + self.B @ u
        return np.array([])


EspacioEstados = StateSpace
