"""Módulo para el modelado y simulación de sistemas dinámicos no lineales interconectados.

Este módulo provee las clases base necesarias para definir subsistemas dinámicos
independientes y un coordinador global capaz de resolver la topología de conexiones
y el orden de ejecución mediante ordenamiento topológico, permitiendo la integración
numérica directa con `scipy.integrate.solve_ivp`.
"""

from typing import Any, Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp


class SubsystemBase:
    """Clase base abstracta para representar un subsistema dinámico no lineal.

    Cada subsistema define sus ecuaciones en variables de estado continuas:
        dx/dt = f(x, u, t)   (Ecuación de estado)
        y     = g(x, u, t)   (Ecuación de salida)

    Atributos:
        state_dim (int): Dimensión del vector de estado x.
        input_dim (int): Dimensión del vector de entrada u.
        output_dim (int): Dimensión del vector de salida y.
        name (str): Nombre único identificador del subsistema.
        direct_feedthrough (bool): True si la entrada u afecta instantáneamente
            a la salida y en el mismo instante de tiempo (lazo algebraico potencial).
            False si la salida y solo depende del estado actual x.
        connections (dict): Mapeo interno de conexiones de entrada.
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        output_dim: int,
        name: str = "subsystem",
        direct_feedthrough: bool = True,
        x0: np.ndarray | Sequence[float] | None = None,
    ):
        """Inicializa las dimensiones y configuraciones básicas del subsistema."""
        self.state_dim: int = state_dim
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self._name: str = name
        self.direct_feedthrough: bool = direct_feedthrough
        self._user_input_names: dict[int, str] = {}
        self._user_output_names: dict[int, str] = {}
        self._user_state_names: dict[int, str] = {}

        self._x0: np.ndarray = np.zeros(self.state_dim)
        if x0 is not None:
            self.x0 = x0

        self._current_output: np.ndarray | None = None
        self._current_time: float = 0.0
        self._temp_connections: list[tuple[int, "SubsystemBase", int]] = []
        self.parent: "InterconnectedSystem | None" = None

    @property
    def name(self) -> str:
        """Nombre del subsistema."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    def set_name(self, value: str) -> None:
        """Configura el nombre del subsistema."""
        self.name = value

    def set_input_name(self, idx: int, name: str) -> None:
        """Establece el nombre de una entrada específica."""
        if idx < 0 or idx >= self.input_dim:
            raise ValueError(
                f"[{self.name}] Índice de entrada {idx} fuera de rango (input_dim={self.input_dim})."
            )
        self._user_input_names[idx] = name

    def get_input_name(self, idx: int) -> str:
        """Obtiene el nombre de una entrada específica."""
        if idx < 0 or idx >= self.input_dim:
            raise ValueError(
                f"[{self.name}] Índice de entrada {idx} fuera de rango (input_dim={self.input_dim})."
            )
        return self._user_input_names.get(idx, f"u_{self.name}_{idx + 1}")

    def set_input_names(self, names: Sequence[str]) -> None:
        """Establece los nombres de todas las entradas."""
        if len(names) != self.input_dim:
            raise ValueError(
                f"[{self.name}] La cantidad de nombres ({len(names)}) debe coincidir con input_dim ({self.input_dim})."
            )
        for idx, name in enumerate(names):
            self.set_input_name(idx, name)

    @property
    def input_names(self) -> list[str]:
        """Lista de nombres de todas las entradas."""
        return [self.get_input_name(i) for i in range(self.input_dim)]

    def set_output_name(self, idx: int, name: str) -> None:
        """Establece el nombre de una salida específica."""
        if idx < 0 or idx >= self.output_dim:
            raise ValueError(
                f"[{self.name}] Índice de salida {idx} fuera de rango (output_dim={self.output_dim})."
            )
        self._user_output_names[idx] = name

    def get_output_name(self, idx: int) -> str:
        """Obtiene el nombre de una salida específica."""
        if idx < 0 or idx >= self.output_dim:
            raise ValueError(
                f"[{self.name}] Índice de salida {idx} fuera de rango (output_dim={self.output_dim})."
            )
        return self._user_output_names.get(idx, f"y_{self.name}_{idx + 1}")

    def set_output_names(self, names: Sequence[str]) -> None:
        """Establece los nombres de todas las salidas."""
        if len(names) != self.output_dim:
            raise ValueError(
                f"[{self.name}] La cantidad de nombres ({len(names)}) debe coincidir con output_dim ({self.output_dim})."
            )
        for idx, name in enumerate(names):
            self.set_output_name(idx, name)

    @property
    def output_names(self) -> list[str]:
        """Lista de nombres de todas las salidas."""
        return [self.get_output_name(i) for i in range(self.output_dim)]

    def set_state_name(self, idx: int, name: str) -> None:
        """Establece el nombre de un estado específico."""
        if idx < 0 or idx >= self.state_dim:
            raise ValueError(
                f"[{self.name}] Índice de estado {idx} fuera de rango (state_dim={self.state_dim})."
            )
        self._user_state_names[idx] = name

    def get_state_name(self, idx: int) -> str:
        """Obtiene el nombre de un estado específico."""
        if idx < 0 or idx >= self.state_dim:
            raise ValueError(
                f"[{self.name}] Índice de estado {idx} fuera de rango (state_dim={self.state_dim})."
            )
        return self._user_state_names.get(idx, f"x_{self.name}_{idx + 1}")

    def set_state_names(self, names: Sequence[str]) -> None:
        """Establece los nombres de todos los estados."""
        if len(names) != self.state_dim:
            raise ValueError(
                f"[{self.name}] La cantidad de nombres ({len(names)}) debe coincidir con state_dim ({self.state_dim})."
            )
        for idx, name in enumerate(names):
            self.set_state_name(idx, name)

    @property
    def state_names(self) -> list[str]:
        """Lista de nombres de todos los estados."""
        return [self.get_state_name(i) for i in range(self.state_dim)]

    @property
    def x0(self) -> np.ndarray:
        """Vector de estado inicial local por defecto del subsistema."""
        return self._x0

    @x0.setter
    def x0(self, value: np.ndarray | Sequence[float]) -> None:
        val_arr = np.asarray(value, dtype=float)
        if len(val_arr) != self.state_dim:
            raise ValueError(
                f"[{self.name}] La longitud de x0 ({len(val_arr)}) no coincide con state_dim ({self.state_dim})."
            )
        self._x0 = val_arr

    def connect_input(
        self, input_idx: int, source_subsystem: "SubsystemBase", source_output_idx: int
    ) -> None:
        """Conecta una entrada específica de este bloque a la salida de otro subsistema.

        Si ya pertenece a un sistema interconectado, delega la conexión a este.
        De lo contrario, almacena la conexión de forma temporal hasta que sea agregado.
        """
        if self.parent is not None:
            self.parent.connect(source_subsystem, source_output_idx, self, input_idx)
        else:
            self._temp_connections.append((input_idx, source_subsystem, source_output_idx))

    def update_outputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Ecuación de salida del sistema: y = g(x, u, t).

        Debe ser sobreescrita por la subclase específica para calcular las salidas actuales.

        Args:
            x (np.ndarray): Vector de estado local actual.
            u (np.ndarray): Vector de entrada local actual.

        Returns:
            np.ndarray: Vector de salida 'y' de tamaño (output_dim,).
        """
        raise NotImplementedError(
            "Cada subsistema debe implementar su propio método update_outputs."
        )

    def compute_derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Ecuación de estado diferencial: dx/dt = f(x, u, t).

        Debe ser sobreescrita por la subclase específica para calcular las derivadas de estado.

        Args:
            x (np.ndarray): Vector de estado local actual.
            u (np.ndarray): Vector de entrada local actual.

        Returns:
            np.ndarray: Vector de derivadas 'dx/dt' de tamaño (state_dim,).
        """
        raise NotImplementedError(
            "Cada subsistema debe implementar su propio método compute_derivatives."
        )

    def system_dynamics(
        self,
        t: float,
        x: np.ndarray,
        u_ext_func: Callable[[float], np.ndarray] | None = None,
    ) -> np.ndarray:
        """Punto de entrada compatible con `scipy.integrate.solve_ivp` para simulación aislada.

        Permite probar, validar y sintonizar este subsistema de forma 100% independiente
        del resto de la red interconectada.

        Args:
            t (float): Tiempo actual de integración.
            x (np.ndarray): Vector de estado local actual.
            u_ext_func (Callable, opcional): Función u(t) que devuelve todas las entradas necesarias
                para el bloque aislado (vector de tamaño input_dim).

        Returns:
            np.ndarray: Vector de derivadas locales.
        """
        self._current_time = t
        u_ext = u_ext_func(t) if u_ext_func else np.zeros(self.input_dim)
        self._current_output = self.update_outputs(x, u_ext)
        return self.compute_derivatives(x, u_ext)

    def solve(
        self,
        t_span: tuple[float, float],
        x0: np.ndarray | Sequence[float] | None = None,
        u_ext_func: Callable[[float], np.ndarray] | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resuelve la simulación de este subsistema de forma aislada.

        Args:
            t_span (tuple[float, float]): Intervalo de tiempo (t_inicio, t_fin).
            x0 (np.ndarray | Sequence[float], opcional): Vector de estado inicial local. Si es None, usa el valor por defecto del subsistema.
            u_ext_func (Callable, opcional): Función u(t) que retorna el vector de entrada en cada instante t.
            **kwargs: Argumentos adicionales que se pasan a scipy.integrate.solve_ivp.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Un tuple conteniendo:
                - t (np.ndarray): Vector de tiempos de tamaño (N,).
                - x (np.ndarray): Matriz de estados de tamaño (N, state_dim).
                - y (np.ndarray): Matriz de salidas de tamaño (N, output_dim).
        """
        if x0 is None:
            x0_arr = self.x0
        else:
            x0_arr = np.asarray(x0, dtype=float)
            if len(x0_arr) != self.state_dim:
                raise ValueError(
                    f"[{self.name}] La longitud de x0 ({len(x0_arr)}) no coincide con state_dim ({self.state_dim})."
                )

        sol = solve_ivp(
            fun=lambda t, x: self.system_dynamics(t, x, u_ext_func),
            t_span=t_span,
            y0=x0_arr,
            **kwargs,
        )
        t_steps = sol.t
        x_steps = sol.y.T

        y_list = []
        for i in range(len(t_steps)):
            t_i = t_steps[i]
            x_i = x_steps[i]
            u_ext = u_ext_func(t_i) if u_ext_func else np.zeros(self.input_dim)
            y_i = self.update_outputs(x_i, u_ext)
            y_list.append(y_i)

        y_steps = np.array(y_list)
        return t_steps, x_steps, y_steps

    def find_steady_state(
        self,
        u: np.ndarray | Sequence[float],
        x_guess: np.ndarray | Sequence[float] | None = None,
        t: float = 0.0,
        method: str = "hybr",
        **kwargs,
    ) -> np.ndarray:
        """Encuentra numéricamente el estado estacionario de este subsistema para una entrada constante dada.

        Busca x_ss tal que compute_derivatives(x_ss, u) = 0.

        Args:
            u (np.ndarray | Sequence[float]): Vector de entrada constante.
            x_guess (np.ndarray | Sequence[float], opcional): Estimación inicial. Si es None, usa x0.
            t (float): Tiempo en el cual evaluar las ecuaciones (por defecto 0.0).
            method (str): Método de resolución de scipy.optimize.root (por defecto 'hybr').
            **kwargs: Parámetros adicionales para scipy.optimize.root.

        Returns:
            np.ndarray: Vector de estado estacionario.
        """
        if self.state_dim == 0:
            return np.zeros(0)

        self._current_time = t
        u_arr = np.asarray(u, dtype=float)
        x_guess_arr = (
            np.asarray(x_guess, dtype=float)
            if x_guess is not None
            else np.asarray(self.x0, dtype=float)
        )

        def residuals(x_val: np.ndarray) -> np.ndarray:
            return self.compute_derivatives(x_val, u_arr)

        from scipy.optimize import root

        res = root(residuals, x_guess_arr, method=method, **kwargs)
        if not res.success:
            raise RuntimeError(
                f"No se pudo encontrar el estado estacionario del subsistema '{self.name}': {res.message}"
            )
        return res.x

    def linearize(
        self,
        x0: np.ndarray | Sequence[float],
        u0: np.ndarray | Sequence[float],
        eps: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Linealiza este subsistema aislado alrededor de un punto de operación (x0, u0).

        Utiliza diferencias finitas centrales para aproximar las matrices jacobianas del subsistema:
            dx/dt ≈ A * delta_x + B * delta_u
            y ≈ C * delta_x + D * delta_u

        Args:
            x0 (np.ndarray | Sequence[float]): Vector de estado de operación.
            u0 (np.ndarray | Sequence[float]): Vector de entrada de operación.
            eps (float): Paso de perturbación para diferencias finitas.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Las matrices (A, B, C, D).
        """
        x0_arr = np.asarray(x0, dtype=float)
        u0_arr = np.asarray(u0, dtype=float)

        n_x = len(x0_arr)
        n_u = len(u0_arr)

        # Evaluar punto nominal para determinar n_y
        y0 = self.update_outputs(x0_arr, u0_arr)
        n_y = len(y0)

        A = np.zeros((n_x, n_x))
        B = np.zeros((n_x, n_u))
        C = np.zeros((n_y, n_x))
        D = np.zeros((n_y, n_u))

        # Calcular A y C (perturbando estados x)
        for j in range(n_x):
            x_plus = x0_arr.copy()
            x_plus[j] += eps
            dx_plus = self.compute_derivatives(x_plus, u0_arr)
            y_plus = self.update_outputs(x_plus, u0_arr)

            x_minus = x0_arr.copy()
            x_minus[j] -= eps
            dx_minus = self.compute_derivatives(x_minus, u0_arr)
            y_minus = self.update_outputs(x_minus, u0_arr)

            A[:, j] = (dx_plus - dx_minus) / (2.0 * eps)
            C[:, j] = (y_plus - y_minus) / (2.0 * eps)

        # Calcular B y D (perturbando entradas u)
        for j in range(n_u):
            u_plus = u0_arr.copy()
            u_plus[j] += eps
            dx_plus = self.compute_derivatives(x0_arr, u_plus)
            y_plus = self.update_outputs(x0_arr, u_plus)

            u_minus = u0_arr.copy()
            u_minus[j] -= eps
            dx_minus = self.compute_derivatives(x0_arr, u_minus)
            y_minus = self.update_outputs(x0_arr, u_minus)

            B[:, j] = (dx_plus - dx_minus) / (2.0 * eps)
            D[:, j] = (y_plus - y_minus) / (2.0 * eps)

        return A, B, C, D


class InterconnectedSystem(SubsystemBase):
    """Clase coordinadora encargada de agrupar, interconectar y resolver la red de subsistemas.

    Puede actuar a su vez como un subsistema dentro de otra red (jerarquía).
    """

    def __init__(
        self,
        input_dim: int = 0,
        output_dim: int = 0,
        name: str = "interconnected_system",
        direct_feedthrough: bool = True,
    ):
        """Inicializa un sistema interconectado vacío."""
        self.subsystems: list[SubsystemBase] = []
        self.state_indices: dict[str, tuple[int, int]] = {}
        self.total_state_dim: int = 0
        self.ordered_subsystems: list[SubsystemBase] = []
        self.output_mappings: dict[int, list[tuple[SubsystemBase, int]]] = {}
        self.connections: dict[tuple[SubsystemBase, int], list[tuple[SubsystemBase, int]]] = {}
        self.external_connections: dict[tuple[SubsystemBase, int], list[int]] = {}

        super().__init__(
            state_dim=0,
            input_dim=input_dim,
            output_dim=output_dim,
            name=name,
            direct_feedthrough=direct_feedthrough,
            x0=np.zeros(0),
        )

    @property
    def x0(self) -> np.ndarray:
        """Retorna el vector de estado inicial consolidado a partir de los subsistemas internos."""
        parts = [np.asarray(sys.x0, dtype=float) for sys in self.subsystems]
        return np.concatenate(parts) if parts else np.zeros(0)

    @x0.setter
    def x0(self, value: np.ndarray | Sequence[float]) -> None:
        """Permite establecer el estado inicial global repartiéndolo entre los subsistemas."""
        val_arr = np.asarray(value, dtype=float)
        expected_dim = sum(sys.state_dim for sys in self.subsystems)
        if len(val_arr) != expected_dim:
            raise ValueError(
                f"[{self.name}] La longitud del vector x0 ({len(val_arr)}) no coincide con la dimensión total de estados ({expected_dim})."
            )

        offset = 0
        for sys in self.subsystems:
            sys.x0 = val_arr[offset : offset + sys.state_dim]
            offset += sys.state_dim

    def add_subsystem(self, sys: SubsystemBase) -> None:
        """Agrega un subsistema a la red global.

        Args:
            sys (SubsystemBase): Instancia del subsistema a añadir.
        """
        if any(existing.name == sys.name for existing in self.subsystems):
            raise ValueError(
                f"Ya existe un subsistema con el nombre '{sys.name}' en '{self.name}'."
            )
        self.subsystems.append(sys)
        sys.parent = self
        self.ordered_subsystems = []

        # Drenar conexiones temporales que el subsistema haya registrado antes de agregarse
        while sys._temp_connections:
            input_idx, source_subsys, source_out_idx = sys._temp_connections.pop(0)
            self.connect(source_subsys, source_out_idx, sys, input_idx)

    def _extract_local_state(self, x_global: np.ndarray, sys_name: str) -> np.ndarray:
        """Extrae el sub-vector de estado correspondiente a un subsistema específico."""
        start, end = self.state_indices[sys_name]
        return x_global[start:end]

    def connect(
        self,
        source_subsystem: SubsystemBase,
        source_output_idx: int,
        target_subsystem: SubsystemBase,
        target_input_idx: int,
    ) -> None:
        """Conecta la salida de un subsistema a la entrada de otro subsistema dentro de esta red.

        Si se conectan múltiples salidas a la misma entrada, sus valores se sumarán automáticamente.
        """
        if source_subsystem not in self.subsystems:
            raise ValueError(
                f"El subsistema origen '{source_subsystem.name}' no pertenece a este sistema interconectado."
            )
        if target_subsystem not in self.subsystems:
            raise ValueError(
                f"El subsistema destino '{target_subsystem.name}' no pertenece a este sistema interconectado."
            )

        if source_output_idx >= source_subsystem.output_dim:
            raise ValueError(
                f"[{source_subsystem.name}] Índice de salida {source_output_idx} fuera de rango (output_dim={source_subsystem.output_dim})."
            )
        if target_input_idx >= target_subsystem.input_dim:
            raise ValueError(
                f"[{target_subsystem.name}] Índice de entrada {target_input_idx} fuera de rango (input_dim={target_subsystem.input_dim})."
            )

        key = (target_subsystem, target_input_idx)
        if key not in self.connections:
            self.connections[key] = []
        self.connections[key].append((source_subsystem, source_output_idx))
        self.ordered_subsystems = []

    def map_input(
        self, external_input_idx: int, target_subsystem: SubsystemBase, target_input_idx: int
    ) -> None:
        """Asocia una entrada externa de este sistema interconectado a una entrada de un subsistema interno."""
        if target_subsystem not in self.subsystems:
            raise ValueError(
                f"El subsistema destino '{target_subsystem.name}' no pertenece a este sistema interconectado."
            )

        if external_input_idx >= self.input_dim:
            self.input_dim = external_input_idx + 1

        if target_input_idx >= target_subsystem.input_dim:
            raise ValueError(
                f"[{target_subsystem.name}] Índice de entrada interna {target_input_idx} fuera de rango."
            )

        key = (target_subsystem, target_input_idx)
        if key not in self.external_connections:
            self.external_connections[key] = []
        self.external_connections[key].append(external_input_idx)
        self.ordered_subsystems = []

    def map_output(
        self, external_output_idx: int, source_subsystem: SubsystemBase, source_output_idx: int
    ) -> None:
        """Asocia una salida externa de este sistema interconectado a una salida de un subsistema interno."""
        if source_subsystem not in self.subsystems:
            raise ValueError(
                f"El subsistema origen '{source_subsystem.name}' no pertenece a este sistema interconectado."
            )

        if external_output_idx >= self.output_dim:
            self.output_dim = external_output_idx + 1

        if source_output_idx >= source_subsystem.output_dim:
            raise ValueError(
                f"[{source_subsystem.name}] Índice de salida interna {source_output_idx} fuera de rango."
            )

        if external_output_idx not in self.output_mappings:
            self.output_mappings[external_output_idx] = []
        self.output_mappings[external_output_idx].append((source_subsystem, source_output_idx))
        self.ordered_subsystems = []

    def compute_execution_order(self) -> None:
        """Resuelve automáticamente el orden de ejecución utilizando ordenamiento topológico (Algoritmo de Kahn)."""
        # 1. Resolver el orden de ejecución de cualquier subsistema interconectado hijo primero
        for sys in self.subsystems:
            if isinstance(sys, InterconnectedSystem):
                sys.compute_execution_order()

        # 2. Re-calcular las dimensiones de estado y los índices de estado
        self.state_indices = {}
        self.total_state_dim = 0
        for sys in self.subsystems:
            if isinstance(sys, InterconnectedSystem):
                sys.state_dim = sys.total_state_dim

            start_idx = self.total_state_dim
            end_idx = start_idx + sys.state_dim
            self.state_indices[sys.name] = (start_idx, end_idx)
            self.total_state_dim += sys.state_dim

        self.state_dim = self.total_state_dim

        # 3. Ordenamiento topológico para dependencias instantáneas
        adj: dict[str, list[str]] = {sys.name: [] for sys in self.subsystems}
        in_degree: dict[str, int] = {sys.name: 0 for sys in self.subsystems}
        name_to_sys = {sys.name: sys for sys in self.subsystems}

        for (target_sys, _input_idx), sources in self.connections.items():
            if target_sys.direct_feedthrough:
                for source_sys, _ in sources:
                    if source_sys.name in adj:
                        adj[source_sys.name].append(target_sys.name)
                        in_degree[target_sys.name] += 1

        # Encolar nodos independientes en el grafo de cálculo inmediato
        queue = [name for name, degree in in_degree.items() if degree == 0]
        order: list[str] = []

        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(order) != len(self.subsystems):
            raise RuntimeError(
                "¡Lazo algebraico circular detectado! Imposible resolver orden dinámico. "
                "Asegúrese de que al menos un subsistema en el lazo cerrado tenga 'direct_feedthrough=False'."
            )

        self.ordered_subsystems = [name_to_sys[name] for name in order]

    def _get_external_outputs(self) -> np.ndarray:
        """Helper para extraer las salidas externas mapeadas a partir del caché actual."""
        y = np.zeros(self.output_dim)
        for i in range(self.output_dim):
            val = 0.0
            if i in self.output_mappings:
                for source_sys, out_idx in self.output_mappings[i]:
                    if source_sys._current_output is not None:
                        val += source_sys._current_output[out_idx]
            y[i] = val
        return y

    def _resolve_subsystem_inputs(
        self, sys: SubsystemBase, external_inputs: np.ndarray | None
    ) -> np.ndarray:
        u_local = np.zeros(sys.input_dim)
        for i in range(sys.input_dim):
            val = 0.0
            has_connection = False

            # 1. Conexiones internas
            key = (sys, i)
            if key in self.connections:
                for src_sys, out_idx in self.connections[key]:
                    if src_sys._current_output is None:
                        raise RuntimeError(
                            f"La salida de '{src_sys.name}' requerida por '{sys.name}' (entrada {i}) no ha sido calculada aún. Verifique el orden de ejecución o lazos algebraicos."
                        )
                    val += src_sys._current_output[out_idx]
                has_connection = True

            # 2. Entradas externas mapeadas
            if key in self.external_connections and external_inputs is not None:
                for ext_idx in self.external_connections[key]:
                    if ext_idx < len(external_inputs):
                        val += external_inputs[ext_idx]
                has_connection = True

            if has_connection:
                u_local[i] = val
            else:
                # Comportamiento legacy: si no hay conexión explícita pero se provee u_ext de la misma dimensión
                if external_inputs is not None and i < len(external_inputs):
                    u_local[i] = external_inputs[i]
        return u_local

    def update_outputs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Ecuación de salida del sistema interconectado (para uso jerárquico)."""
        if not self.ordered_subsystems:
            self.compute_execution_order()

        states = {sys.name: self._extract_local_state(x, sys.name) for sys in self.subsystems}

        # Invalidar caché inter-paso y propagar el tiempo
        for sys in self.subsystems:
            sys._current_output = None
            sys._current_time = self._current_time

        # PASO 1: Calcular bloques sin paso directo
        for sys in self.subsystems:
            if not sys.direct_feedthrough:
                sys._current_output = sys.update_outputs(states[sys.name], np.array([]))

        # PASO 2: Propagación topológica
        for sys in self.ordered_subsystems:
            if sys._current_output is not None:
                continue
            u_local = self._resolve_subsystem_inputs(sys, u)
            sys._current_output = sys.update_outputs(states[sys.name], u_local)

        return self._get_external_outputs()

    def compute_derivatives(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Ecuación de estado del sistema interconectado (para uso jerárquico)."""
        if not self.ordered_subsystems:
            self.compute_execution_order()

        states = {sys.name: self._extract_local_state(x, sys.name) for sys in self.subsystems}

        dx_parts = []
        for sys in self.subsystems:
            u_local = self._resolve_subsystem_inputs(sys, u)
            dx_local = sys.compute_derivatives(states[sys.name], u_local)
            dx_parts.append(dx_local)

        return np.concatenate(dx_parts) if dx_parts else np.zeros(0)

    def system_dynamics(
        self,
        t: float,
        x: np.ndarray,
        u_ext_func: Callable[[float], np.ndarray] | None = None,
    ) -> np.ndarray:
        """Punto de entrada compatible con `scipy.integrate.solve_ivp` para la red completa."""
        if not self.ordered_subsystems:
            self.compute_execution_order()

        states = {sys.name: self._extract_local_state(x, sys.name) for sys in self.subsystems}
        u_ext = u_ext_func(t) if u_ext_func else None

        # Invalidar caché inter-paso y actualizar el tiempo
        for sys in self.subsystems:
            sys._current_output = None
            sys._current_time = t

        # PASO 1: Calcular bloques sin paso directo
        for sys in self.subsystems:
            if not sys.direct_feedthrough:
                sys._current_output = sys.update_outputs(states[sys.name], np.array([]))

        # PASO 2: Propagación topológica
        for sys in self.ordered_subsystems:
            if sys._current_output is not None:
                continue
            u_local = self._resolve_subsystem_inputs(sys, u_ext)
            sys._current_output = sys.update_outputs(states[sys.name], u_local)

        # PASO 3: Construcción indexada del vector de derivadas global
        global_dx = np.zeros(self.total_state_dim)
        for sys in self.subsystems:
            u_local = self._resolve_subsystem_inputs(sys, u_ext)
            dx_local = sys.compute_derivatives(states[sys.name], u_local)
            start, end = self.state_indices[sys.name]
            global_dx[start:end] = dx_local

        return global_dx

    def solve(  # type: ignore
        self,
        t_span: tuple[float, float],
        x0: np.ndarray | Sequence[float] | None = None,
        u_ext_func: Callable[[float], np.ndarray] | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Resuelve la simulación de la red completa de subsistemas interconectados."""
        if not self.ordered_subsystems:
            self.compute_execution_order()

        if x0 is None:
            x0_arr = self.x0
        else:
            x0_arr = np.asarray(x0, dtype=float)
            if len(x0_arr) != self.total_state_dim:
                raise ValueError(
                    f"La longitud de x0 ({len(x0_arr)}) no coincide con la dimensión total de estados ({self.total_state_dim})."
                )

        sol = solve_ivp(
            fun=lambda t, x: self.system_dynamics(t, x, u_ext_func),
            t_span=t_span,
            y0=x0_arr,
            **kwargs,
        )
        t_steps = sol.t
        x_global_steps = sol.y.T

        states_dict: dict[str, list[np.ndarray]] = {sys.name: [] for sys in self.subsystems}
        outputs_dict: dict[str, list[np.ndarray]] = {sys.name: [] for sys in self.subsystems}
        states_dict[self.name] = []
        outputs_dict[self.name] = []

        inputs_list: list[np.ndarray] = []
        for i in range(len(t_steps)):
            t_i = t_steps[i]
            x_g_i = x_global_steps[i]
            u_i = u_ext_func(t_i) if u_ext_func else np.zeros(self.input_dim)
            inputs_list.append(u_i)

            _ = self.system_dynamics(t_i, x_g_i, u_ext_func)
            for sys in self.subsystems:
                x_local = self._extract_local_state(x_g_i, sys.name)
                states_dict[sys.name].append(x_local)
                if sys._current_output is None:
                    raise RuntimeError(
                        f"La salida de '{sys.name}' no pudo ser calculada en t = {t_i}."
                    )
                outputs_dict[sys.name].append(sys._current_output.copy())
            states_dict[self.name].append(x_g_i.copy())
            outputs_dict[self.name].append(self._get_external_outputs().copy())

        states_dict_arr = {name: np.array(lst) for name, lst in states_dict.items()}
        outputs_dict_arr = {name: np.array(lst) for name, lst in outputs_dict.items()}
        outputs_dict_arr["inputs"] = np.array(inputs_list)

        return t_steps, x_global_steps, states_dict_arr, outputs_dict_arr

    def find_steady_state(
        self,
        u: np.ndarray | Sequence[float],
        x_guess: np.ndarray | Sequence[float] | None = None,
        t: float = 0.0,
        method: str = "hybr",
        **kwargs,
    ) -> np.ndarray:
        """Encuentra numéricamente el estado estacionario global de la red para una entrada externa constante dada."""
        if not self.ordered_subsystems:
            self.compute_execution_order()

        if self.total_state_dim == 0:
            return np.zeros(0)

        u0_arr = np.asarray(u, dtype=float)

        if x_guess is not None:
            x_guess_arr = np.asarray(x_guess, dtype=float)
        else:
            x_guess_arr = self.x0

        def u_ext_func(t_val: float) -> np.ndarray:
            return u0_arr

        def residuals(x_val: np.ndarray) -> np.ndarray:
            return self.system_dynamics(t, x_val, u_ext_func)

        from scipy.optimize import root

        res = root(residuals, x_guess_arr, method=method, **kwargs)
        if not res.success:
            raise RuntimeError(
                f"No se pudo encontrar el estado estacionario del sistema interconectado: {res.message}"
            )
        return res.x

    def linearize(
        self,
        x0: np.ndarray | Sequence[float],
        u0: np.ndarray | Sequence[float],
        eps: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Linealiza el sistema interconectado alrededor de un punto de operación (x0, u0)."""
        x0 = np.asarray(x0, dtype=float)
        u0 = np.asarray(u0, dtype=float)

        if not self.ordered_subsystems:
            self.compute_execution_order()

        n_x = len(x0)
        n_u = len(u0)

        # Evaluar el punto nominal para determinar n_y
        dx0, y0 = self._eval_system(x0, u0)
        n_y = len(y0)

        # Inicializar matrices jacobianas
        A = np.zeros((n_x, n_x))
        B = np.zeros((n_x, n_u))
        C = np.zeros((n_y, n_x))
        D = np.zeros((n_y, n_u))

        # Calcular A y C (perturbando los estados x)
        for j in range(n_x):
            x_plus = x0.copy()
            x_plus[j] += eps
            dx_plus, y_plus = self._eval_system(x_plus, u0)

            x_minus = x0.copy()
            x_minus[j] -= eps
            dx_minus, y_minus = self._eval_system(x_minus, u0)

            A[:, j] = (dx_plus - dx_minus) / (2.0 * eps)
            C[:, j] = (y_plus - y_minus) / (2.0 * eps)

        # Calcular B y D (perturbando las entradas u)
        for j in range(n_u):
            u_plus = u0.copy()
            u_plus[j] += eps
            dx_plus, y_plus = self._eval_system(x0, u_plus)

            u_minus = u0.copy()
            u_minus[j] -= eps
            dx_minus, y_minus = self._eval_system(x0, u_minus)

            B[:, j] = (dx_plus - dx_minus) / (2.0 * eps)
            D[:, j] = (y_plus - y_minus) / (2.0 * eps)

        return A, B, C, D

    def _eval_system(self, x: np.ndarray, u_ext: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Helper para evaluar la dinámica global y obtener las salidas combinadas."""
        dx_global = self.system_dynamics(0.0, x, lambda t: u_ext)
        y_global = self._get_external_outputs()
        return dx_global, y_global

    def check_connections(self) -> dict[str, Any]:
        """Realiza un chequeo de conexiones del sistema interconectado.

        Identifica qué entradas y salidas de cada subsistema están conectadas
        y cuáles permanecen desconectadas.
        """
        report = {
            "subsystems": {},
            "unconnected_inputs": [],
            "unconnected_outputs": [],
        }

        # 1. Registrar todos los destinos de cada salida para saber si están conectadas
        output_destinations = {}
        for sys in self.subsystems:
            output_destinations[sys.name] = [[] for _ in range(sys.output_dim)]

        # Escanear conexiones de entrada internas
        for (target_sys, input_idx), sources in self.connections.items():
            for source_sys, out_idx in sources:
                if source_sys.name in output_destinations:
                    output_destinations[source_sys.name][out_idx].append(
                        {"type": "internal", "subsystem": target_sys.name, "input": input_idx}
                    )

        # Escanear mapeos de salida externos del parent
        for ext_out_idx, sources in self.output_mappings.items():
            for source_sys, out_idx in sources:
                if source_sys.name in output_destinations:
                    output_destinations[source_sys.name][out_idx].append(
                        {"type": "external_output", "index": ext_out_idx}
                    )

        # 2. Generar el reporte para cada subsistema
        for sys in self.subsystems:
            sys_report = {
                "inputs": {},
                "outputs": {},
            }

            # Chequear entradas
            for i in range(sys.input_dim):
                sources_info = []

                # Conexiones internas
                key = (sys, i)
                if key in self.connections:
                    for source_sys, out_idx in self.connections[key]:
                        sources_info.append(
                            {"type": "internal", "subsystem": source_sys.name, "output": out_idx}
                        )

                # Conexiones externas (mapeadas desde entradas del parent)
                if key in self.external_connections:
                    for ext_idx in self.external_connections[key]:
                        sources_info.append({"type": "external_input", "index": ext_idx})

                is_connected = len(sources_info) > 0
                sys_report["inputs"][i] = {"connected": is_connected, "sources": sources_info}

                if not is_connected:
                    report["unconnected_inputs"].append((sys.name, i))

            # Chequear salidas
            for j in range(sys.output_dim):
                destinations = output_destinations[sys.name][j]
                is_connected = len(destinations) > 0
                sys_report["outputs"][j] = {"connected": is_connected, "destinations": destinations}
                if not is_connected:
                    report["unconnected_outputs"].append((sys.name, j))

            report["subsystems"][sys.name] = sys_report

        # 3. Imprimir informe formateado
        print("\n==================================================")
        print(f" INFORME DE CONEXIONES: {self.name} ")
        print("==================================================")
        for sys_name, sys_rep in report["subsystems"].items():
            print(f"\nSubsistema: {sys_name}")
            print("  Entradas:")
            for idx, info in sys_rep["inputs"].items():
                status = "CONECTADO" if info["connected"] else "DESCONECTADO"
                print(f"    - Entrada [{idx}]: {status}")
                for src in info["sources"]:
                    if src["type"] == "internal":
                        print(f"        <- Salida [{src['output']}] de '{src['subsystem']}'")
                    elif src["type"] == "external_input":
                        print(f"        <- Entrada Externa [{src['index']}] de '{self.name}'")
            print("  Salidas:")
            for idx, info in sys_rep["outputs"].items():
                status = "CONECTADO" if info["connected"] else "DESCONECTADO"
                print(f"    - Salida  [{idx}]: {status}")
                for dest in info["destinations"]:
                    if dest["type"] == "internal":
                        print(f"        -> Entrada [{dest['input']}] de '{dest['subsystem']}'")
                    elif dest["type"] == "external_output":
                        print(f"        -> Salida Externa [{dest['index']}] de '{self.name}'")

        print("\nResumen de señales sin conectar:")
        if report["unconnected_inputs"]:
            print(f"  Entradas libres: {report['unconnected_inputs']}")
        else:
            print("  Entradas libres: Ninguna")
        if report["unconnected_outputs"]:
            print(f"  Salidas libres:  {report['unconnected_outputs']}")
        else:
            print("  Salidas libres:  Ninguna")
        print("==================================================\n")

        return report


# Alias para mayor flexibilidad y compatibilidad
InterconnectedSubsystem = InterconnectedSystem
