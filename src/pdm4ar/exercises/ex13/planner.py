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
        # X_bar, U_bar, p_bar = self.initial_guess()
        for iteration in range(self.params.max_iterations):
            self._convexification()  # popolando A, B, F, r e riempendo X_bar, U_bar, p_bar con i loro valori correnit
            try:
                error = self.problem.solve(
                    verbose=self.params.verbose_solver, solver=self.params.solver
                )  # linearized cost (denominator of ro)
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")
            # update trust region (compute actual cost, new x star and old x bar)
            # if convergence break

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
            "nu_k": cvx.Variable((self.satellite.n_x, self.params.K)),  # i need a nu for linearized dynamics error, 55b
            "nu_s_k": cvx.Variable((self.satellite.n_x, self.params.K)),  # for obstacles constraints 55d
            "nu_ic": cvx.Variable((self.satellite.n_x, self.params.K)),  # 55e
            "nu_tc": cvx.Variable((self.satellite.n_x, self.params.K)),  # 55f
        }

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        problem_parameters = {
            "init_state": cvx.Parameter(self.satellite.n_x),
            "goal_state": cvx.Parameter(self.satellite.n_x),
            "A_bar": cvx.Parameter((self.satellite.n_x * self.satellite.n_x, self.params.K - 1)),
            "B_plus_bar": cvx.Parameter((self.satellite.n_x * self.satellite.n_u, self.params.K - 1)),
            "B_minus_bar": cvx.Parameter((self.satellite.n_x * self.satellite.n_u, self.params.K - 1)),
            "F_bar": cvx.Parameter((self.satellite.n_x * self.satellite.n_p, self.params.K - 1)),
            "r_bar": cvx.Parameter((self.satellite.n_x, self.params.K - 1)),
            "X_bar": cvx.Parameter((self.satellite.n_x, self.params.K)),
            "U_bar": cvx.Parameter((self.satellite.n_u, self.params.K)),
            "p_bar": cvx.Parameter(self.satellite.n_p),
            "tr_radius": cvx.Parameter(),
            # ...quello su cui non ottimizziamo, namely stato iniziale e finale, la dinamica al k step, la initial guess?
        }

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """
        constraints = [
            self.variables["X"][:, 0] == self.problem_parameters["init_state"],
            # ...
        ]
        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # Example objective
        objective = self.params.weight_p @ self.variables["p"]

        return cvx.Minimize(objective)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        self.problem_parameters["X_bar"].value = (
            self.X_bar
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

        self.problem_parameters["tr_radius"].value = SolverParameters.tr_radius

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """

        pass

    def _update_trust_region(self):
        """
        Update trust region radius.
        """
        pass

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
