# PDM4AR – Exercises

This repository contains our solutions for the **Planning and Decision-Making for Autonomous Robots (PDM4AR)** exercises. The focus of the project is on trajectory optimization, planning under constraints, and simulation-based evaluation.

All official instructions, problem statements, and evaluation details are available on the [course website](https://pdm4ar.github.io/exercises/).

---

## 📂 Project Structure

The implementation follows the structure provided by the course template. The relevant logic is implemented inside the `src/pdm4ar/exercises` folder.

### Key Components
* **Planner:** Computes dynamically feasible trajectories between an initial state and a goal state.
* **Optimization:** A **CVXPY-based** optimization problem incorporating state, control, and dynamics constraints.
* **Trust-Region:** A mechanism to iteratively refine the solution via Sequential Convex Programming (SCP).
* **Integration:** Full compatibility with the `dg_commons` simulator for execution and evaluation.

---

## 🚀 Exercise 13 – Trajectory Optimization

Exercise 13 focuses on computing a collision-free and dynamically feasible trajectory by solving a sequence of convex optimization problems. The planner operates as follows:

1.  **Initialize** a nominal trajectory.
2.  **Linearize** the dynamics around the current solution.
3.  **Solve** a convex optimization problem using CVXPY.
4.  **Evaluate** the solution quality.
5.  **Update** the trust region based on the ratio between predicted and actual cost reduction.
6.  **Repeat** until convergence or termination conditions are met.

The resulting state and control trajectories are then passed to the simulator for execution.

---

## 🛠 Implementation Notes

* **Optimization:** Problems are formulated using **CVXPY**.
* **Solvers:** Solver choice can be configured (e.g., `ECOS` or `Clarabel`).
* **Trust-Region:** Updates are performed after solving the optimization problem and rely only on numerical values (not symbolic CVXPY expressions).
* **DCP Rules:** All constraints and objectives follow CVXPY’s disciplined convex programming rules.
* **Data Extraction:** Care must be taken to extract `.value` from CVXPY variables before using them in logical conditions or numerical comparisons.

---

## 💻 Running the Simulation

The evaluation framework automatically calls the planner through the provided interface. A typical execution flow is:

1.  **Initialization:** The simulator initializes the episode.
2.  **Trigger:** `on_episode_init()` triggers trajectory computation.
3.  **Output:** The planner returns a command sequence and state trajectory.
4.  **Execution:** The simulator executes the trajectory and checks feasibility.
5.  **Diagnostics:** Execution time and solver diagnostics are logged during runtime.
<video src="https://github.com/angelostefanini0/Multi-Agent-Multi-Goal-Pickup-and-Delivery/raw/main/docs/Recording%202026-02-08%20204903.mp4" width="100%" controls autoplay loop muted>
  Il tuo browser non supporta il tag video.
</video>
