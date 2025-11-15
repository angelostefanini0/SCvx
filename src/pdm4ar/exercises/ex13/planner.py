import ast
from dataclasses import dataclass, field
from re import U
from typing import Union

import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import (
    SatelliteGeometry,
    SatelliteParameters,
)

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
    max_iterations: int = 100  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable weight
    weight_p: NDArray = field(default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time

    tr_radius: float = 5  # initial trust region radius
    min_tr_radius: float = 1e-4  # min trust region radius
    max_tr_radius: float = 100  # max trust region radius
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

        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        # Constraints
        constraints = self._get_constraints()

        # Objective
        objective = self._get_objective()

        # Cvx Optimisation Problem
        self.problem = cvx.Problem(objective, constraints)

    def compute_trajectory(
        self, init_state: SatelliteState, goal_state: DynObstacleState
    ) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Compute a trajectory from init_state to goal_state.
        """
        self.init_state = init_state
        self.goal_state = goal_state

        init_vec = np.array(
            [
                init_state.x,
                init_state.y,
                init_state.psi,
                init_state.vx,
                init_state.vy,
                init_state.dpsi,
            ]
        )

        goal_vec = np.array(
            [
                goal_state.x,
                goal_state.y,
                goal_state.psi,
                goal_state.vx,
                goal_state.vy,
                goal_state.dpsi,
            ]
        )

        self.problem_parameters["init_state"].value = init_vec
        self.problem_parameters["goal_state"].value = goal_vec

        self.problem_parameters["eta"].value = self.params.tr_radius

        #
        # TODO: Implement SCvx algorithm or comparable
        #

        """
        for SCvx it would follow a logic similar to:
        
        initial guess interpolation
        while stopping criterion not satisfied
            convexify
            discretize
            solve convex sub problem
            update trust region
            update stopping criterion
        """
        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        for iteration in range(self.params.max_iterations):
            self._convexification()  # popolando A, B, F, r e riempendo X_bar, U_bar, p_bar
            try:
                error = self.problem.solve(
                    verbose=self.params.verbose_solver, solver=self.params.solver
                )  # linearized cost (denominator of ro)
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")

            if self._check_convergence():
                break

            self._update_trust_region()  # copiando X star in self.X_bar(aggiornaimo X, U, p, radius)

        # Example data: sequence from array
        mycmds, mystates = self._extract_seq_from_array()

        return mycmds, mystates

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """
        K = self.params.K

        X = np.zeros((self.satellite.n_x, K))
        U = np.zeros((self.satellite.n_u, K))
        p = np.zeros((self.satellite.n_p))

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
            "nu": cvx.Variable((self.satellite.n_x, self.params.K - 1)),
            "nu_s": cvx.Variable((self.satellite.n_x, self.params.K - 1)),  # NOT SO SURE, IT'S FOR CONTRAINTS
            "nu_ic": cvx.Variable(self.satellite.n_x),
            "nu_tc": cvx.Variable(self.satellite.n_x - 1),  # JUST POSITION, NOT VELOCITY
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
            self.variables["X"][:, 0] == self.problem_parameters["init_state"],
            # ...
            (  # TRUST REGION CONSTRAINT
                cvx.sum_squares(self.variables["X"] - self.problem_parameters["X_bar"])
                + cvx.sum_squares(self.variables["U"] - self.problem_parameters["U_bar"])
                + cvx.sum_squares(self.variables["p"] - self.problem_parameters["p_bar"])
                <= self.problem_parameters["eta"] ** 2  # ETA WITH OR WITHOUT .VALUE?
            ),
        ]

        constraints.append(self.variables["U"][:, 0] == 0.0)
        constraints.append(self.variables["U"][:, -1] == 0.0)
        constraints.append(
            self.variables["X"][0:5, -1] - self.problem_parameters["goal_state"][0:5] + self.variables["nu_tc"] == 0.0
        )
        constraints.append(self.satellite.sp.F_limits[0] <= self.variables["U"][1, :] <= self.satellite.sp.F_limits[1])
        constraints.append(self.satellite.sp.F_limits[0] <= self.variables["U"][0, :] <= self.satellite.sp.F_limits[1])
        constraints.append(
            self.variables["X"][0:6, 0] - self.problem_parameters["init_state"][0:6] + self.variables["nu_ic"] == 0.0
        )
        for i in self.planets:
            planet = self.planets[i]
            xp, yp = planet.center
            radius = planet.radius
            for k in range(self.params.K):
                xk, yk = self.variables["X"][0, k], self.variables["X"][1, k]
                rprime = (
                    -((xk - xp) ** 2)
                    - (yk - yp) ** 2
                    + (radius + self.sg.w_panel + self.sg.w_half) ** 2
                    + 2 * (xk - xp) * xk
                    + 2 * (yk - yp) * yk
                )
                constraints.append(-2 * (xk - xp) * xk - 2 * (yk - yp) * yk + rprime <= 0)
        for k in range(self.params.K - 1):
            constraints.append(
                self.variables["X"][:, k + 1]
                == (
                    self.problem_parameters["A_bar"][:, k] @ self.variables["X"][:, k]
                    + self.problem_parameters["B_plus_bar"][:, k] @ self.variables["U"][:, k + 1]
                    + self.problem_parameters["B_minus_bar"][:, k] @ self.variables["U"][:, k]
                    + self.problem_parameters["F_bar"][:, k] @ self.variables["p"]
                    + self.problem_parameters["r_bar"][:, k]
                    + self.variables["nu"][:, k]
                )
            )

        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # HERE WE CAN PUT ALSO SLACK VAR, FINAL POSITION (AS OBJ, NOT RIGID CONTRAINT), TIME, TRAJECTORY LENGHT, ACTUATION FORCES
        # Example objective
        obj_time = self.params.weight_p @ self.variables["p"]
        obj_terminal_violations = self.params.lambda_nu * cvx.norm(
            self.variables["nu_ic"][:], p=1
        ) + self.params.lambda_nu * cvx.norm(self.variables["nu_tc"][:], p=1)
        phi = obj_time + obj_terminal_violations  # 50, terminal cost

        running_cost = 0  # WE COULD ADD ACTUATION FORCES
        Gamma = []  # TO WRITE
        for k in range(self.params.K - 1):
            P = cvx.norm1(self.variables["nu"][:, k]) + cvx.norm1(self.variables["nu_s"][:, k])
            Gamma.append(running_cost + self.params.lambda_nu * P)
        delta_t = 1.0 / self.params.K
        trapz = 0
        for i in range(self.params.K - 2):  # in paper from 1 to K-1 but here from 0 to K-2
            trapz += Gamma[i] + Gamma[i + 1]
        trapz = (delta_t / 2) * trapz
        objective = phi + trapz
        return cvx.Minimize(objective)

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

    def _compute_rho(self) -> float:  # NEW
        """
        Compute rho value for trust region update.
        rho = actual improvement / predicted improvement.
        """

        """
        # Define gamma lambda
        gamma_lambda_bar = []
        for k in range(self.params.K - 1):
            gamma_lambda_bar.append(self.params.lambda_nu * np.linalg.norm( bhooo[k], ord=1))

        gamma_lambda_opt = []
        for k in range(self.params.K - 1):
            gamma_lambda_opt.append(self.params.lambda_nu * np.linalg.norm(bhooo[k], ord=1))

        # Compute trapezoidal integration
        delta_t = 1.0 / self.params.K
        gamma_bar = 0
        for k in range(self.params.K - 2):
            gamma_bar += delta_t / 2 * (gamma_lambda_bar[k] + gamma_lambda_bar[k + 1])

        gamma_opt = 0
        for k in range(self.params.K - 2):
            gamma_opt += delta_t / 2 * (gamma_lambda_opt[k] + gamma_lambda_opt[k + 1])
        """
        rho = (cost_func_bar - cost_func_opt) / (cost_func_bar - cost_linear_opt)
        return rho

    @staticmethod
    def _extract_seq_from_array() -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Example of how to create a DgSampledSequence from numpy arrays and timestamps.
        """
        ts = (0, 1, 2, 3, 4)
        # in case my planner returns 3 numpy arrays
        F = np.array([0, 1, 2, 3, 4])
        ddelta = np.array([0, 0, 0, 0, 0])
        cmds_list = [SatelliteCommands(f, dd) for f, dd in zip(F, ddelta)]
        mycmds = DgSampledSequence[SatelliteCommands](timestamps=ts, values=cmds_list)

        # in case my state trajectory is in a 2d array
        npstates = np.random.rand(len(ts), 6)
        states = [SatelliteState(*v) for v in npstates]
        mystates = DgSampledSequence[SatelliteState](timestamps=ts, values=states)
        return mycmds, mystates
