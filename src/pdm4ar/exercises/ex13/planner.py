import ast
from dataclasses import dataclass, field
from math import cos
from re import U
from typing import Union, cast

import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import (
    SatelliteGeometry,
    SatelliteParameters,
)
from matplotlib.pylab import f

# from pdm4ar.exercises.ex13.agent import SatelliteAgent
from pdm4ar.exercises.ex13.discretization import *
from pdm4ar.exercises_def.ex13.utils_params import PlanetParams, AsteroidParams


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "ECOS"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 30  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable weight
    weight_p: NDArray = field(default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time

    tr_radius: float = 5  # initial trust region radius
    min_tr_radius: float = 1e-4  # min trust region radius      #IN THE PAPER IT IS 1E-3
    max_tr_radius: float = 100  # max trust region radius       #IN THE PAPER IT IS 10
    rho_0: float = 0.0  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.0  # div factor trust region update
    beta: float = 3.2  # mult factor trust region update

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    stop_crit: float = 1e-5  # Stopping criteria constant


class SatellitePlanner:
    """
    Feel free to change anything in this class.
    """

    planets: dict[PlayerName, PlanetParams]
    asteroids: dict[PlayerName, AsteroidParams]
    satellite: SatelliteDyn
    sg: SatelliteGeometry
    sp: SatelliteParameters
    params: SolverParameters

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        asteroids: dict[PlayerName, AsteroidParams],
        sg: SatelliteGeometry,
        sp: SatelliteParameters,
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.asteroids = asteroids
        self.sg = sg
        self.sp = sp

        self.error: float = 0.0

        # Solver Parameters
        self.params = SolverParameters()

        # Satellite Dynamics
        self.satellite = SatelliteDyn(self.sg, self.sp)

        # Discretization Method
        # self.integrator = ZeroOrderHold(self.Satellite, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(self.satellite, self.params.K, self.params.N_sub)

        # Check dynamics implementation (pass this test before going further. It is not part of the final evaluation, so you can comment it out later)
        if not self.integrator.check_dynamics():
            raise ValueError("Dynamics check failed.")
        else:
            print("Dynamics check passed.")

        # Variables
        self.variables = self._get_variables()

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()

        # self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        # Constraints
        # constraints = self._get_constraints()

        # Objective
        # objective = self._get_objective()

        # Cvx Optimisation Problem

    # self.problem = cvx.Problem(objective, constraints)

    def compute_trajectory(
        self, init_state: SatelliteState, goal_state: DynObstacleState
    ) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Compute a trajectory from init_state to goal_state using a simple SCvx loop.
        """
        self.init_state = init_state
        self.goal_state = goal_state

        init_vec = np.array([init_state.x, init_state.y, init_state.psi, init_state.vx, init_state.vy, init_state.dpsi])
        goal_vec = np.array([goal_state.x, goal_state.y, goal_state.psi, goal_state.vx, goal_state.vy, goal_state.dpsi])

        self.problem_parameters["init_state"].value = init_vec
        self.problem_parameters["goal_state"].value = goal_vec

        # trust region iniziale (per ora non usiamo rho, ma lasciamo eta)
        self.problem_parameters["eta"].value = self.params.tr_radius

        # initial guess (stati, comandi, tempo finale)
        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        # costruiamo il problema una sola volta
        constraints = self._get_constraints()
        objective = self._get_objective()
        self.problem = cvx.Problem(objective, constraints)

        for iteration in range(self.params.max_iterations):
            # ---- salva la vecchia traiettoria ----

            # ---- convexification: aggiorna A_bar, B_bar, ... e X_bar/U_bar/p_bar come parametri ----
            self._convexification()

            # ---- warm-start delle variabili CVX ----

            # ---- risolvi il problema convesso ----
            try:
                error = self.problem.solve(
                    verbose=self.params.verbose_solver,
                    solver=self.params.solver,
                )
                self.error = float(error)
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")
                break

            if self._check_convergence():
                break
            print(self.variables["nu_ic"].value)

            self._update_trust_region()  # copiando X star in self.X_bar(aggiornaimo X, U, p in self.X_bar ecc, eta)

        # costruisci sequenze di comandi e stati dalla soluzione finale
        mycmds, mystates = self._extract_seq_from_array()
        return mycmds, mystates

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """

        init_state = self.init_state
        goal_state = self.goal_state

        K = self.params.K

        X = np.zeros((self.satellite.n_x, K))
        U = np.zeros((self.satellite.n_u, K))
        p = np.zeros((self.satellite.n_p))

        x0 = np.array(
            [
                init_state.x,
                init_state.y,
                init_state.psi,
                init_state.vx,
                init_state.vy,
                init_state.dpsi,
            ]
        )
        xt = np.array(
            [
                goal_state.x,
                goal_state.y,
                goal_state.psi,
                goal_state.vx,
                goal_state.vy,
                goal_state.dpsi,
            ]
        )

        for i in range(K):
            t = i / (K - 1)
            X[:, i] = (1 - t) * x0 + t * xt
            U[:, i] = np.array([0.0, 0.0])  # constraint: The initial and final inputs needs to be zero

        deltax = xt[0] - x0[0]
        deltay = xt[1] - x0[1]
        p[0] = np.sqrt(deltax**2 + deltay**2) / self.sp.vx_limits[1] * 1.2
        p = np.full(self.satellite.n_p, p)

        return X, U, p

    def _set_goal(self):
        """
        Sets goal for SCvx.
        """
        self.goal = cvx.Parameter((6, 1))
        pass

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        variables = {
            "X": cvx.Variable((self.satellite.n_x, self.params.K)),
            "U": cvx.Variable((self.satellite.n_u, self.params.K)),
            "p": cvx.Variable(self.satellite.n_p),
            "nu": cvx.Variable((self.satellite.n_x, self.params.K)),
            "nu_s": cvx.Variable((self.params.K)),  # NOT SO SURE, IT'S FOR CONTRAINTS
            "nu_ic": cvx.Variable(self.satellite.n_x),
            # "nu_tc": cvx.Variable(self.satellite.n_x - 1),  # JUST POSITION, NOT VELOCITY
        }

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        problem_parameters = {
            "init_state": cvx.Parameter(self.satellite.n_x),
            "eta": cvx.Parameter(),  # trust region radius, does it need to be modified every iteration so every time it should restart from init value or keep updating?
            # when do we set its value the 1 time? for self.problem_parameters["eta"].value
            "goal_state": cvx.Parameter(self.satellite.n_x),
            "A_bar": cvx.Parameter((self.satellite.n_x * self.satellite.n_x, self.params.K - 1)),
            "B_plus_bar": cvx.Parameter((self.satellite.n_x * self.satellite.n_u, self.params.K - 1)),
            "B_minus_bar": cvx.Parameter((self.satellite.n_x * self.satellite.n_u, self.params.K - 1)),
            "F_bar": cvx.Parameter((self.satellite.n_x * self.satellite.n_p, self.params.K - 1)),
            "r_bar": cvx.Parameter((self.satellite.n_x, self.params.K - 1)),
            "X_bar": cvx.Parameter((self.satellite.n_x, self.params.K)),
            "U_bar": cvx.Parameter((self.satellite.n_u, self.params.K)),
            "p_bar": cvx.Parameter(self.satellite.n_p),
            # ...quello su cui non ottimizziamo, namely stato iniziale e finale, la dinamica al k step, la initial guess?
        }

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        Aggiungere box constraint on inputs and states, sui pianeti statici, constraint dinamica, stati iniziali e finali
        """

        constraints = [
            # self.variables["X"][:, 0] == self.problem_parameters["init_state"],  # HARD CONSTRAINT
            # ...
            (  # TRUST REGION CONSTRAINT
                cvx.sum_squares(self.variables["X"] - self.problem_parameters["X_bar"])
                + cvx.sum_squares(self.variables["U"] - self.problem_parameters["U_bar"])
                + cvx.sum_squares(self.variables["p"] - self.problem_parameters["p_bar"])
                <= (self.problem_parameters["eta"]) ** 2  # ETA WITH OR WITHOUT .VALUE?
            ),
        ]

        # TIME COSTRAINTS
        constraints.append(self.variables["p"] <= 80.0)
        constraints.append(self.variables["p"] >= 0.0)

        # PROBLEM COSTRAINS
        constraints.append(self.variables["U"][:, 0] == 0.0)
        constraints.append(self.variables["U"][:, -1] == 0.0)
        constraints.append(self.variables["X"][0:5, -1] == self.problem_parameters["goal_state"][0:5])
        constraints += [
            self.satellite.sp.F_limits[0] <= self.variables["U"][1, :],
            self.variables["U"][1, :] <= self.satellite.sp.F_limits[1],
            self.satellite.sp.F_limits[0] <= self.variables["U"][0, :],
            self.variables["U"][0, :] <= self.satellite.sp.F_limits[1],
        ]
        constraints.append(
            self.variables["X"][:, 0] + self.variables["nu_ic"] == self.problem_parameters["init_state"][:]
        )

        # PLANET AVOIDANCE CONSTRAINTS
        for i in self.planets:
            planet = self.planets[i]
            xp, yp = planet.center
            R = planet.radius + np.sqrt((self.sg.w_panel + self.sg.w_half) ** 2 + self.sg.l_r**2)

            for k in range(self.params.K - 1):
                xk = self.variables["X"][0, k]
                yk = self.variables["X"][1, k]

                xbar = self.problem_parameters["X_bar"][0, k]
                ybar = self.problem_parameters["X_bar"][1, k]

                Cx = -2 * (xbar - xp)
                Cy = -2 * (ybar - yp)

                rprime = -((xbar - xp) ** 2) - (ybar - yp) ** 2 + R**2 + 2 * (xbar - xp) * xbar + 2 * (ybar - yp) * ybar

                constraints.append(Cx * xk + Cy * yk + rprime <= self.variables["nu_s"][k])

        # DYNAMICS CONSTRAINTS

        for k in range(self.params.K - 1):
            A = cvx.reshape(self.problem_parameters["A_bar"][:, k], (self.satellite.n_x, self.satellite.n_x), order="F")

            Bplus = cvx.reshape(
                self.problem_parameters["B_plus_bar"][:, k], (self.satellite.n_x, self.satellite.n_u), order="F"
            )

            Bminus = cvx.reshape(
                self.problem_parameters["B_minus_bar"][:, k], (self.satellite.n_x, self.satellite.n_u), order="F"
            )

            F = cvx.reshape(self.problem_parameters["F_bar"][:, k], (self.satellite.n_x, self.satellite.n_p), order="F")

            constraints.append(
                self.variables["X"][:, k + 1]
                == A @ self.variables["X"][:, k]
                + Bplus @ self.variables["U"][:, k + 1]
                + Bminus @ self.variables["U"][:, k]
                + F @ self.variables["p"]
                + self.problem_parameters["r_bar"][:, k]
                + self.variables["nu"][:, k]
            )

        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # HERE WE CAN PUT ALSO SLACK VAR, FINAL POSITION (AS OBJ, NOT RIGID CONTRAINT), TIME, TRAJECTORY LENGHT, ACTUATION FORCES
        # Example objective
        obj_time = self.params.weight_p @ self.variables["p"]
        obj_terminal_violations = self.params.lambda_nu * cvx.norm(self.variables["nu_ic"][:], p=1)
        phi = obj_time + obj_terminal_violations  # 50, terminal cost

        running_cost = 0.0  # WE COULD ADD ACTUATION FORCES
        Gamma = []  # TO WRITE
        for k in range(self.params.K):
            P = cvx.norm1(self.variables["nu"][:, k]) + cvx.norm1(
                self.variables["nu_s"][k]
            )  # SE RIUSCIAMO AD AVERE TUTIT VINCOLI CONVESSI NU_S INUTILE
            Gamma.append(
                running_cost + self.params.lambda_nu * P
            )  # CAPIAMO COME TRATTARLO NU_S SE ZERO, NONE O TOGLIERLO DEL TUTTO
        objective = phi + self.trapz(Gamma)
        return cvx.Minimize(objective)

    def trapz(self, values: list[float]) -> float:
        """
        Compute trapezoidal integration of a list.
        """
        delta_t = 1.0 / (self.params.K - 1)  # self.p_bar[0] ? forse K ?
        integral = 0
        for i in range(len(values) - 1):  # in paper from 1 to K-1 but here from 0 to K-2
            integral += values[i] + values[i + 1]
        integral = (delta_t / 2) * integral
        return integral

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        Vorremmo spostare l'assegnazione di X_bar, U_bar, p_bar come problem parameters direttamente detro update trust region, senza passare per X_bar
        """
        self.problem_parameters["X_bar"].value = (
            self.X_bar  # copy?
        )  # bisogna fare in modo che quando è chiamta convexification i valori di X_bar ecc descrivono il path precedente
        self.problem_parameters["U_bar"].value = self.U_bar
        self.problem_parameters["p_bar"].value = self.p_bar
        # ...
        # ZOH
        # A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
        # FOH
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(
            self.X_bar, self.U_bar, self.p_bar
        )

        # HINT: be aware that the matrices returned by calculate_discretization are flattened in F order (this way affect your code later when you use them)

        self.problem_parameters["A_bar"].value = A_bar  # aggiornati
        self.problem_parameters["B_plus_bar"].value = B_plus_bar
        self.problem_parameters["B_minus_bar"].value = B_minus_bar
        self.problem_parameters["F_bar"].value = F_bar
        self.problem_parameters["r_bar"].value = r_bar  # aggiornati

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """  # WE COULD INSTEAD CHECK J_lambda - L_lambda <= stop_crit
        delta_x = np.linalg.norm(self.variables["X"].value - self.X_bar, axis=0)
        delta_p = np.linalg.norm(self.variables["p"].value - self.p_bar)

        return bool(delta_p + np.max(delta_x) <= self.params.stop_crit)

    def _update_trust_region(self):
        """
        Update trust region radius.
        """
        rho = self._compute_rho()  # WE MUST PAY ATTENTION TO SELF.ETA IF NEEDS TO BE RESTART FROM INIT VALUE
        # Update trust region considering the computed rho
        print(f"Rho: {rho}")

        # print(rho <= self.params.rho_0)
        if rho <= self.params.rho_0:
            self.problem_parameters["eta"].value = max(
                self.params.min_tr_radius, self.problem_parameters["eta"].value / self.params.alpha
            )

        elif self.params.rho_0 <= rho < self.params.rho_1:
            self.problem_parameters["eta"].value = max(
                self.params.min_tr_radius, self.problem_parameters["eta"].value / self.params.alpha
            )
            self.X_bar = self.variables["X"].value
            self.U_bar = self.variables["U"].value
            self.p_bar = self.variables["p"].value
        elif self.params.rho_1 <= rho < self.params.rho_2:
            self.X_bar = self.variables["X"].value
            self.U_bar = self.variables["U"].value
            self.p_bar = self.variables["p"].value
        else:
            self.problem_parameters["eta"].value = min(
                self.params.max_tr_radius, self.params.beta * self.problem_parameters["eta"].value
            )
            self.X_bar = self.variables["X"].value
            self.U_bar = self.variables["U"].value
            self.p_bar = self.variables["p"].value
        pass

        print(self.problem_parameters["eta"].value)

        # phi should return a scalar

    def phi_lambda(
        self, p: NDArray, X: NDArray
    ) -> float:  # ricordati quando arriverà docking che è non convesso                 X_bar or variables["X"].values
        """
        Compute phi_lambda for trust region update.
        """
        # p = self.params.weight_p @ p
        # ic = self.params.lambda_nu * np.linalg.norm(X[:, 0] - self.problem_parameters["init_state"].value, ord=1)
        tc = self.params.lambda_nu * np.linalg.norm(X[:, -1] - self.problem_parameters["goal_state"].value, ord=1)
        print(tc)
        return float(
            self.params.weight_p @ p
            + self.params.lambda_nu * np.linalg.norm(X[:, 0] - self.problem_parameters["init_state"].value, ord=1)
        )

    # type of arg4 and 5 ??
    def Gamma_lambda(self, running_cost: float, defect: list[NDArray], X: NDArray, U: NDArray, p: NDArray):
        Gamma = []

        for k in range(self.params.K - 1):
            defect_penalty = np.linalg.norm(defect[k], ord=1)

            obstacle_violation_sum = 0.0

            for i in self.planets:
                planet = self.planets[i]
                xp, yp = planet.center

                # Calcolo Raggio aumentato
                R = planet.radius + np.sqrt((self.sg.w_panel + self.sg.w_half) ** 2 + self.sg.l_r**2)

                # Calcolo valore vincolo (scalare)
                # Positivo = Violazione (dentro l'ostacolo)
                val = -((X[0, k] - xp) ** 2) - (X[1, k] - yp) ** 2 + R**2

                if val >= 0:
                    obstacle_violation_sum += val

            # La penalità totale è lambda * (errore dinamica + errore ostacoli)
            total_penalty = self.params.lambda_nu * (defect_penalty + obstacle_violation_sum)

            Gamma.append(running_cost + total_penalty)

        return Gamma

    def defect(self, X: NDArray, U: NDArray, p: NDArray) -> list[NDArray]:
        """
        Compute the discretization defect numerically (NOT as CVX expressions).
        """
        X_nl = self.integrator.integrate_nonlinear_piecewise(X, U, p)  # shape (n_x, K)
        # defect matrix for transitions k=0..K-2 -> columns correspond to k -> k+1
        defects_mat = X[:, 1:] - X_nl[:, 1:]  # shape (n_x, K-1)
        defects_list = [defects_mat[:, k] for k in range(self.params.K - 1)]
        print("Defects norm: ", defects_list[0:5])
        return defects_list

    def _compute_rho(self) -> float:
        """
        Compute rho = actual_improvement / predicted_improvement
        Robust: uses .value safely, uses defects as X[:,1:] - X_nl[:,1:],
        protects against None and near-zero denominators.
        """
        # gamma_lambda non so cosa mettere come argomenti
        # running_cost Lamba potrebbe essere attuatori (U), arg_5 potrebbe essere vincoli non convessi [s(t,X,U,p)]^+
        cost_func_bar = self.phi_lambda(self.p_bar, self.X_bar) + self.trapz(
            self.Gamma_lambda(
                0,
                self.defect(self.X_bar, self.U_bar, self.p_bar),
                self.X_bar,
                self.U_bar,
                self.p_bar,
            )
        )
        cost_func_bar = float(cost_func_bar)
        phi = self.phi_lambda(self.variables["p"].value, self.variables["X"].value)
        gamma = self.trapz(
            self.Gamma_lambda(
                0,
                self.defect(
                    self.variables["X"].value,
                    self.variables["U"].value,
                    self.variables["p"].value,
                ),
                self.variables["X"].value,
                self.variables["U"].value,
                self.variables["p"].value,
            )
        )
        cost_func_opt = phi + gamma
        # print(phi)
        # print(gamma)

        cost_func_opt = float(cost_func_opt)
        rho = (cost_func_bar - cost_func_opt) / (cost_func_bar - float(self.error))
        # print(cost_func_bar, cost_func_opt, float(self.error))
        return rho

    def _extract_seq_from_array(self):

        p_star = float(self.p_bar)
        K = self.params.K

        ts = tuple([i * p_star / (K - 1) for i in range(K)])

        U = self.U_bar
        cmds_list = [SatelliteCommands(F_left=float(U[0, k]), F_right=float(U[1, k])) for k in range(K)]
        cmd_seq = DgSampledSequence[SatelliteCommands](timestamps=ts, values=cmds_list)

        X = self.X_bar
        states_list = [
            SatelliteState(
                x=float(X[0, k]),
                y=float(X[1, k]),
                psi=float(X[2, k]),
                vx=float(X[3, k]),
                vy=float(X[4, k]),
                dpsi=float(X[5, k]),
            )
            for k in range(K)
        ]
        state_seq = DgSampledSequence[SatelliteState](timestamps=ts, values=states_list)

        return cmd_seq, state_seq
