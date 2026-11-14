# rddlrepository

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
[![PyPI Version](https://img.shields.io/pypi/v/rddlrepository.svg)](https://pypi.org/project/rddlrepository/)
[![Documentation Status](https://readthedocs.org/projects/pyrddlgym/badge/?version=latest)](https://pyrddlgym.readthedocs.io/en/latest/rddlrepo.html)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

**A library of planning and reinforcement learning problems written in RDDL, ready to run in [pyRDDLGym](https://github.com/pyRDDLGym-project/pyRDDLGym).**

[Installation](#installation) | [Quick start](#quick-start) | [Finding problems](#finding-problems) | [Adding your own](#adding-your-own-problems) | [Domain gallery](#domain-gallery)

## Why rddlrepository?

- **Scalable.** Every domain is written in RDDL, so an instance can grow from a handful of objects to a million with just a few lines of RDDL code.
- **Diverse.** Domains come from many disciplines and showcase the latest official RDDL syntax, which makes them good starting points for learning the language.
- **Visual.** Many domains come with custom visualizers for pyRDDLGym.
- **Plug and play.** Everything works out of the box with the [pyRDDLGym ecosystem](https://github.com/pyRDDLGym-project/pyRDDLGym).

## Installation

Install the latest release with pip:

```shell
pip install rddlrepository
```

Or get the pre-release version from GitHub:

```shell
git clone https://github.com/pyRDDLGym-project/rddlrepository.git
```

The visualizers need [pyRDDLGym](https://github.com/pyRDDLGym-project/pyRDDLGym) (``pip install pyRDDLGym``).

### Rebuilding the manifest

If you add domains or instances by hand, rebuild the manifest (rddlrepository >= 2.2):

```shell
rddlrepo rebuild
```

> [!NOTE]
> If you install a new version of rddlrepository on top of an old one, rebuild the manifest as above.
> In versions older than 2.2, delete the ``manifest.csv`` file in the ``core`` folder of your rddlrepository installation instead; it is rebuilt automatically.

## Quick start

With pyRDDLGym installed, create an environment with ``make()``, as in Gym:

```python
import pyRDDLGym
env = pyRDDLGym.make("EarthObservation_ippc2018", "1")
```

``env`` is a Gym ``Env`` with ``reset()`` and ``step()`` ready to use.

## Finding problems

### Listing domains

Everything starts from the ``RDDLRepoManager``:

```python
from rddlrepository.core.manager import RDDLRepoManager
manager = RDDLRepoManager(rebuild=True)
```

List all domains:

```python
print(manager.list_problems())
```

Domains are grouped into contexts, such as a competition year or ``standalone``:

```python
print(manager.list_contexts())
print(manager.list_problems_by_context("standalone"))   # all standalone problems
print(manager.list_problems_by_context("ippc2018"))     # all problems from IPPC 2018
```

### Working with instances

Each domain is described by a ``ProblemInfo``:

```python
problem_info = manager.get_problem("EarthObservation_ippc2018")
```

List its instances:

```python
print(problem_info.list_instances())
```

Get the paths of the domain and of an instance (here ``1``):

```python
print(problem_info.get_domain())
print(problem_info.get_instance("1"))
```

Get its pyRDDLGym visualizer class:

```python
viz_class = problem_info.get_visualizer()
```

Together, these give you full control over how an environment is built:

```python
import pyRDDLGym
env = pyRDDLGym.make(domain=problem_info.get_domain(), instance=problem_info.get_instance("1"))
env.set_visualizer(problem_info.get_visualizer())
```

## Adding your own problems

Register a new context:

```python
manager.register_context("MyContext")
```

Register a new domain in a context:

```python
domain_content = """
    domain ... {
    ...
    }
"""
manager.register_domain("MyDomain", "MyContext", domain_content,
                        desc="a description of this domain", viz="ModuleName.ClassName") 
```

Here ```"ModuleName.ClassName"``` names the module and class of the domain's visualizer (optional).

Register a new instance of an existing domain:

```python
instance_content = """
    instance ... {
    ...
    }
"""
problem_info.register_instance("MyInstance", instance_content)
```

Once registered, they can be loaded later like any other problem.

## Domain gallery

Every problem in the repository, grouped by context, with the identifier to pass to ``make()`` and its instances. Hover over a picture for a description of the domain.

<!-- DOMAINS:START -->

**127 problems in 10 contexts:** [standalone](#standalone) (14) &middot; [physics](#physics) (14) &middot; [or](#or) (7) &middot; [arcade](#arcade) (13) &middot; [gym](#gym) (6) &middot; [rddlsim](#rddlsim) (6) &middot; [ippc2011](#ippc2011) (16) &middot; [ippc2014](#ippc2014) (16) &middot; [ippc2018](#ippc2018) (27) &middot; [ippc2023](#ippc2023) (8)

### standalone

Domains written for this repository.

<table>
<tr><td align="center" valign="top"><img src="docs/thumbnails/BernoulliBandit.png" width="180" alt="Bayesian MDP formulation of the Bernoulli bandit problem." title="Bayesian MDP formulation of the Bernoulli bandit problem."><br><code>BernoulliBandit</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Elevators.png" width="180" alt="The Elevator domain models evening rush hours when people from different floors in a building want to go down to the bottom floor using elevators." title="The Elevator domain models evening rush hours when people from different floors in a building want to go down to the bottom floor using elevators."><br><code>Elevators</code><br><sub>instances 0, 1</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Epidemic.png" width="180" alt="SIR model with nonlinear infection dynamics, age groups, hospitalization and multiple interventions." title="SIR model with nonlinear infection dynamics, age groups, hospitalization and multiple interventions."><br><code>Epidemic</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Fishery.png" width="180" alt="Harvest of fish stocks that compete with each other and are eaten by predators, with stochastic growth and a penalty for collapse, as in the three-species fishery of Montealegre-Mora et al. (2023)." title="Harvest of fish stocks that compete with each other and are eaten by predators, with stochastic growth and a penalty for collapse, as in the three-species fishery of Montealegre-Mora et al. (2023)."><br><code>Fishery</code><br><sub>instances 0</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/HVAC.png" width="180" alt="Room temperature control simulation." title="Room temperature control simulation."><br><code>HVAC</code><br><sub>instances 0, 1</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Intruders_Continuous.png" width="180" alt="Continuous intruder detection problem on a unit square." title="Continuous intruder detection problem on a unit square."><br><code>Intruders_Continuous</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Intruders_Discrete.png" width="180" alt="Discrete intruder detection problem on a grid." title="Discrete intruder detection problem on a grid."><br><code>Intruders_Discrete</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Navigation_Continuous.png" width="180" alt="Continuous state action navigation problem with regions to be avoided." title="Continuous state action navigation problem with regions to be avoided."><br><code>Navigation_Continuous</code><br><sub>instances 0</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/PowerGen_Continuous.png" width="180" alt="A simple continuous version of the power generation problem, loosely modeled on the problem of unit commitment." title="A simple continuous version of the power generation problem, loosely modeled on the problem of unit commitment."><br><code>PowerGen_Continuous</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/PowerGen_Discrete.png" width="180" alt="A simple power generation problem loosely modeled on the problem of unit commitment." title="A simple power generation problem loosely modeled on the problem of unit commitment."><br><code>PowerGen_Discrete</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Reservoir_Continuous.png" width="180" alt="Continuous action version of management of the water level in interconnected reservoirs." title="Continuous action version of management of the water level in interconnected reservoirs."><br><code>Reservoir_Continuous</code><br><sub>instances 0, 1</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Reservoir_Discrete.png" width="180" alt="Discrete action version of management of the water level in interconnected reservoirs." title="Discrete action version of management of the water level in interconnected reservoirs."><br><code>Reservoir_Discrete</code><br><sub>instances 0, 1</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/TrafficBLX_ComplexPhases.png" width="180" alt="BLX/QTM traffic signal control model with a generic phasing scheme. The goal is to control traffic lights to minimize total travel time." title="BLX/QTM traffic signal control model with a generic phasing scheme. The goal is to control traffic lights to minimize total travel time."><br><code>TrafficBLX_</code><br><code>ComplexPhases</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/TrafficBLX_SimplePhases.png" width="180" alt="BLX/QTM traffic signal control model with a fixed phase progression consisting of 4 phases. The goal is to control traffic lights to minimize total travel time." title="BLX/QTM traffic signal control model with a fixed phase progression consisting of 4 phases. The goal is to control traffic lights to minimize total travel time."><br><code>TrafficBLX_</code><br><code>SimplePhases</code><br><sub>instances 0</sub></td><td><img src="docs/thumbnails/_blank.png" width="180" alt=""></td><td><img src="docs/thumbnails/_blank.png" width="180" alt=""></td></tr>
</table>

<sub>[back to contexts](#domain-gallery)</sub>

### physics

Physical systems and control.

<table>
<tr><td align="center" valign="top"><img src="docs/thumbnails/Bicycle_physics.png" width="180" alt="Control a bicycle physics problem." title="Control a bicycle physics problem."><br><code>Bicycle_physics</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/CartPoleChain_physics.png" width="180" alt="The two_poles and three_poles tasks of the DeepMind Control Suite, in which a cart on a rail must swing a hanging chain of poles up and balance it." title="The two_poles and three_poles tasks of the DeepMind Control Suite, in which a cart on a rail must swing a hanging chain of poles up and balance it."><br><code>CartPoleChain_physics</code><br><sub>instances 0, 1</sub></td><td align="center" valign="top"><img src="docs/thumbnails/CartPoleSwingUp_physics.png" width="180" alt="The cart-pole swing-up task of the DeepMind Control Suite, in which a cart on a rail must swing a hanging pole up and balance it." title="The cart-pole swing-up task of the DeepMind Control Suite, in which a cart on a rail must swing a hanging pole up and balance it."><br><code>CartPoleSwingUp_</code><br><code>physics</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Highway_Continuous_physics.png" width="180" alt="Reference implementation of highway-v0 from highway-env with continuous acceleration and steering of an ego car among IDM/MOBIL traffic on a multi-lane highway." title="Reference implementation of highway-v0 from highway-env with continuous acceleration and steering of an ego car among IDM/MOBIL traffic on a multi-lane highway."><br><code>Highway_Continuous_</code><br><code>physics</code><br><sub>instances 0&ndash;4</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/Highway_Discrete_physics.png" width="180" alt="Reference implementation of highway-v0 from highway-env, in which an ego car driven by lane and speed meta-actions overtakes IDM/MOBIL traffic on a multi-lane highway." title="Reference implementation of highway-v0 from highway-env, in which an ego car driven by lane and speed meta-actions overtakes IDM/MOBIL traffic on a multi-lane highway."><br><code>Highway_Discrete_</code><br><code>physics</code><br><sub>instances 0&ndash;4</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Lander_Continuous_physics.png" width="180" alt="Continuous action version of landing a rigid body with throttles on a pad over any terrain." title="Continuous action version of landing a rigid body with throttles on a pad over any terrain."><br><code>Lander_Continuous_</code><br><code>physics</code><br><sub>instances 0&ndash;2</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Lander_Discrete_physics.png" width="180" alt="Discrete action version of landing a rigid body with on or off engines on a pad over any terrain." title="Discrete action version of landing a rigid body with on or off engines on a pad over any terrain."><br><code>Lander_Discrete_</code><br><code>physics</code><br><sub>instances 0&ndash;2</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Pinball_physics.png" width="180" alt="A ball is pushed around polygonal obstacles to a target, bouncing off their edges." title="A ball is pushed around polygonal obstacles to a target, bouncing off their edges."><br><code>Pinball_physics</code><br><sub>instances 0&ndash;3</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/Quadcopter_physics.png" width="180" alt="Control a swarm of four-propeller drones in 3D space." title="Control a swarm of four-propeller drones in 3D space."><br><code>Quadcopter_physics</code><br><sub>instances 0, 1</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Reacher_physics.png" width="180" alt="A generalized version of the reacher domain." title="A generalized version of the reacher domain."><br><code>Reacher_physics</code><br><sub>instances 0, 1</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Reactor_physics.png" width="180" alt="The CSTR benchmark of pc-gym, in which the coolant temperature of an exothermic stirred-tank reactor holds the concentration of the reactant at a setpoint." title="The CSTR benchmark of pc-gym, in which the coolant temperature of an exothermic stirred-tank reactor holds the concentration of the reactant at a setpoint."><br><code>Reactor_physics</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/UAV_Continuous_physics.png" width="180" alt="Continuous action space version of multi-UAV problem where a group of UAVs have to reach goal positions in the 3d Space." title="Continuous action space version of multi-UAV problem where a group of UAVs have to reach goal positions in the 3d Space."><br><code>UAV_Continuous_physics</code><br><sub>instances 0, 1</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/UAV_Discrete_physics.png" width="180" alt="Discrete action space version of multi-UAV problem where a group of UAVs have to reach goal positions in the 3d Space." title="Discrete action space version of multi-UAV problem where a group of UAVs have to reach goal positions in the 3d Space."><br><code>UAV_Discrete_physics</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/UAV_Mixed_physics.png" width="180" alt="Mixed action space version of multi-UAV problem where a group of UAVs have to reach goal positions in the 3d Space." title="Mixed action space version of multi-UAV problem where a group of UAVs have to reach goal positions in the 3d Space."><br><code>UAV_Mixed_physics</code><br><sub>instances 0</sub></td><td><img src="docs/thumbnails/_blank.png" width="180" alt=""></td><td><img src="docs/thumbnails/_blank.png" width="180" alt=""></td></tr>
</table>

<sub>[back to contexts](#domain-gallery)</sub>

### or

Operations research.

<table>
<tr><td align="center" valign="top"><img src="docs/thumbnails/BinPacking_or.png" width="180" alt="Items of random weight are drawn, the goal is to place them into bins while minimizing the number of bins used and the total weight of each bin is within limits." title="Items of random weight are drawn, the goal is to place them into bins while minimizing the number of bins used and the total weight of each bin is within limits."><br><code>BinPacking_or</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/CarRental_or.png" width="180" alt="Jack's Car Rental of Sutton and Barto for any number of locations: cars are moved overnight between locations to meet Poisson rental requests." title="Jack's Car Rental of Sutton and Barto for any number of locations: cars are moved overnight between locations to meet Poisson rental requests."><br><code>CarRental_or</code><br><sub>instances 0, 1</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Knapsack_or.png" width="180" alt="Items of random weight and value are drawn, the goal is to place them into knapsacks of limited total weight while maximizing total value of all items." title="Items of random weight and value are drawn, the goal is to place them into knapsacks of limited total weight while maximizing total value of all items."><br><code>Knapsack_or</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Option_or.png" width="180" alt="Exercise an American max option on correlated assets." title="Exercise an American max option on correlated assets."><br><code>Option_or</code><br><sub>instances 0, 1</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/StructuralMaintenance_or.png" width="180" alt="Inspection and repair of a k-out-of-n structure whose components crack by fatigue (Paris law), from the IMP-MARL benchmark: cracks are hidden and only found by imperfect inspections." title="Inspection and repair of a k-out-of-n structure whose components crack by fatigue (Paris law), from the IMP-MARL benchmark: cracks are hidden and only found by imperfect inspections."><br><code>StructuralMaintenance_</code><br><code>or</code><br><sub>instances 0&ndash;4</sub></td><td align="center" valign="top"><img src="docs/thumbnails/SupplyChain_or.png" width="180" alt="A supply chain with factory and multiple warehouses." title="A supply chain with factory and multiple warehouses."><br><code>SupplyChain_or</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/TSP_or.png" width="180" alt="The travelling salesman problem." title="The travelling salesman problem."><br><code>TSP_or</code><br><sub>instances 0</sub></td><td><img src="docs/thumbnails/_blank.png" width="180" alt=""></td></tr>
</table>

<sub>[back to contexts](#domain-gallery)</sub>

### arcade

Arcade, puzzle and card games.

<table>
<tr><td align="center" valign="top"><img src="docs/thumbnails/Blackjack_arcade.png" width="180" alt="A card is randomly drawn from a deck and added to the player hand, the goal is to stop with the highest value of the hand without going over 21." title="A card is randomly drawn from a deck and added to the player hand, the goal is to stop with the highest value of the hand without going over 21."><br><code>Blackjack_arcade</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/BrickBreaker_arcade.png" width="180" alt="A paddle bounces a ball into a wall of bricks to break them, with the rules of the Breakout game of MinAtar." title="A paddle bounces a ball into a wall of bricks to break them, with the rules of the Breakout game of MinAtar."><br><code>BrickBreaker_arcade</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Eight_arcade.png" width="180" alt="Squares are moved to adjacent empty cells until a specific arrangement is achieved." title="Squares are moved to adjacent empty cells until a specific arrangement is achieved."><br><code>Eight_arcade</code><br><sub>instances 0, 1</sub></td><td align="center" valign="top"><img src="docs/thumbnails/MazeChase_arcade.png" width="180" alt="A forager gathers berries in a hedge maze while wolves roam it, and a torch frightens them for a while." title="A forager gathers berries in a hedge maze while wolves roam it, and a torch frightens them for a while."><br><code>MazeChase_arcade</code><br><sub>instances 0&ndash;3</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/Minesweeper_arcade.png" width="180" alt="Reveal the safe cells of a grid without hitting one of its hidden mines, with the rules of the Minesweeper game of Jumanji." title="Reveal the safe cells of a grid without hitting one of its hidden mines, with the rules of the Minesweeper game of Jumanji."><br><code>Minesweeper_arcade</code><br><sub>instances 0&ndash;2</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Pong_arcade.png" width="180" alt="Single player pong/tennis problem." title="Single player pong/tennis problem."><br><code>Pong_arcade</code><br><sub>instances 0, 1</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Snake_arcade.png" width="180" alt="A snake crawls around a walled grid eating fruit and grows with each one, with the rules of the Snake game of Jumanji." title="A snake crawls around a walled grid eating fruit and grows with each one, with the rules of the Snake game of Jumanji."><br><code>Snake_arcade</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Sokoban_arcade.png" width="180" alt="A person pushes boxes in a warehouse onto designated storage areas, difficult domain due to dead ends." title="A person pushes boxes in a warehouse onto designated storage areas, difficult domain due to dead ends."><br><code>Sokoban_arcade</code><br><sub>instances 0, 1</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/SpaceShooter_arcade.png" width="180" alt="A cannon shoots down a marching wave of aliens that drop bombs on it, with the rules of the Space Invaders game of MinAtar." title="A cannon shoots down a marching wave of aliens that drop bombs on it, with the rules of the Space Invaders game of MinAtar."><br><code>SpaceShooter_arcade</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Sudoku_arcade.png" width="180" alt="The sudoku number puzzle." title="The sudoku number puzzle."><br><code>Sudoku_arcade</code><br><sub>instances 0, 1</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Tetris_arcade.png" width="180" alt="Tetris is the classic block stacking game." title="Tetris is the classic block stacking game."><br><code>Tetris_arcade</code><br><sub>instances 0, 1</sub></td><td align="center" valign="top"><img src="docs/thumbnails/TowerOfHanoi_arcade.png" width="180" alt="The classic tower of Hanoi puzzle, where disks must be stacked onto a given rod." title="The classic tower of Hanoi puzzle, where disks must be stacked onto a given rod."><br><code>TowerOfHanoi_arcade</code><br><sub>instances 0</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/Zombies_arcade.png" width="180" alt="An epidemic game in which humans avoid becoming infected by zombies." title="An epidemic game in which humans avoid becoming infected by zombies."><br><code>Zombies_arcade</code><br><sub>instances 0&ndash;3</sub></td><td><img src="docs/thumbnails/_blank.png" width="180" alt=""></td><td><img src="docs/thumbnails/_blank.png" width="180" alt=""></td><td><img src="docs/thumbnails/_blank.png" width="180" alt=""></td></tr>
</table>

<sub>[back to contexts](#domain-gallery)</sub>

### gym

RDDL versions of the classic control environments of Gymnasium.

<table>
<tr><td align="center" valign="top"><img src="docs/thumbnails/Acrobot_gym.png" width="180" alt="The classical acrobot control problem." title="The classical acrobot control problem."><br><code>Acrobot_gym</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/CartPole_Continuous_gym.png" width="180" alt="A simple continuous state-action MDP for the classical cart-pole system by Rich Sutton, with actions that describe the continuous force applied to the cart." title="A simple continuous state-action MDP for the classical cart-pole system by Rich Sutton, with actions that describe the continuous force applied to the cart."><br><code>CartPole_Continuous_</code><br><code>gym</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/CartPole_Discrete_gym.png" width="180" alt="A simple continuous state discrete action MDP for the classical cart-pole system by Rich Sutton, with actions that describe the direction of the force applied to the cart." title="A simple continuous state discrete action MDP for the classical cart-pole system by Rich Sutton, with actions that describe the direction of the force applied to the cart."><br><code>CartPole_Discrete_gym</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/MountainCar_Continuous_gym.png" width="180" alt="A simple continuous MDP for the classical mountain car control problem." title="A simple continuous MDP for the classical mountain car control problem."><br><code>MountainCar_</code><br><code>Continuous_gym</code><br><sub>instances 0</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/MountainCar_Discrete_gym.png" width="180" alt="A simple continuous MDP with discrete actions for the classical mountain car control problem." title="A simple continuous MDP with discrete actions for the classical mountain car control problem."><br><code>MountainCar_Discrete_</code><br><code>gym</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Pendulum_gym.png" width="180" alt="The classical pendulum control problem." title="The classical pendulum control problem."><br><code>Pendulum_gym</code><br><sub>instances 0</sub></td><td><img src="docs/thumbnails/_blank.png" width="180" alt=""></td><td><img src="docs/thumbnails/_blank.png" width="180" alt=""></td></tr>
</table>

<sub>[back to contexts](#domain-gallery)</sub>

### rddlsim

Examples from rddlsim, the original Java simulator of RDDL.

<table>
<tr><td align="center" valign="top"><img src="docs/thumbnails/ComplexSysAdmin_rddlsim.png" width="180" alt="The well known sys-admin problem with a number of enhancements." title="The well known sys-admin problem with a number of enhancements."><br><code>ComplexSysAdmin_</code><br><code>rddlsim</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Logistics_rddlsim.png" width="180" alt="A logistics problem extended from the standard Box-Truck World." title="A logistics problem extended from the standard Box-Truck World."><br><code>Logistics_rddlsim</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Pizza_rddlsim.png" width="180" alt="A pizza delivery task." title="A pizza delivery task."><br><code>Pizza_rddlsim</code><br><sub>instances 0</sub></td><td align="center" valign="top"><img src="docs/thumbnails/_blank.png" width="180" alt="" title=""><br><code>PropDBN_rddlsim</code><br><sub>instances 0</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/Sidewalk_rddlsim.png" width="180" alt="One or more people walking down a sidewalk with 2 lanes." title="One or more people walking down a sidewalk with 2 lanes."><br><code>Sidewalk_rddlsim</code><br><sub>instances 0, 1</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Workforce_rddlsim.png" width="180" alt="Running a call center." title="Running a call center."><br><code>Workforce_rddlsim</code><br><sub>instances 0</sub></td><td><img src="docs/thumbnails/_blank.png" width="180" alt=""></td><td><img src="docs/thumbnails/_blank.png" width="180" alt=""></td></tr>
</table>

<sub>[back to contexts](#domain-gallery)</sub>

### ippc2011

The International Probabilistic Planning Competition of 2011 ([site](http://users.cecs.anu.edu.au/~ssanner/IPPC_2011/)).

<table>
<tr><td align="center" valign="top"><img src="docs/thumbnails/CooperativeRecon_MDP_ippc2011.png" width="180" alt="There is a 2d grid with an agent, a base, some hazard squares, and objects in different locations." title="There is a 2d grid with an agent, a base, some hazard squares, and objects in different locations."><br><code>CooperativeRecon_MDP_</code><br><code>ippc2011</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/CooperativeRecon_POMDP_ippc2011.png" width="180" alt="There is a 2d grid with an agent, a base, some hazard squares, and objects in different locations. This is the pomdp version." title="There is a 2d grid with an agent, a base, some hazard squares, and objects in different locations. This is the pomdp version."><br><code>CooperativeRecon_</code><br><code>POMDP_ippc2011</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/CrossingTraffic_MDP_ippc2011.png" width="180" alt="In a grid, a robot must get to a goal and avoid obstacles arriving randomly and moving left." title="In a grid, a robot must get to a goal and avoid obstacles arriving randomly and moving left."><br><code>CrossingTraffic_MDP_</code><br><code>ippc2011</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/CrossingTraffic_POMDP_ippc2011.png" width="180" alt="In a grid, a robot must get to a goal and avoid obstacles arriving randomly and moving left. This is the pomdp version." title="In a grid, a robot must get to a goal and avoid obstacles arriving randomly and moving left. This is the pomdp version."><br><code>CrossingTraffic_POMDP_</code><br><code>ippc2011</code><br><sub>instances 1&ndash;10</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/Elevators_MDP_ippc2011.png" width="180" alt="This domain has a number of elevators delivering passengers to either the top or the bottom floor." title="This domain has a number of elevators delivering passengers to either the top or the bottom floor."><br><code>Elevators_MDP_ippc2011</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Elevators_POMDP_ippc2011.png" width="180" alt="This domain has a number of elevators delivering passengers to either the top or the bottom floor. This is the pomdp version." title="This domain has a number of elevators delivering passengers to either the top or the bottom floor. This is the pomdp version."><br><code>Elevators_POMDP_</code><br><code>ippc2011</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/GameOfLife_MDP_ippc2011.png" width="180" alt="A simple DBN to encode Conway's cellular automata game of life on a grid." title="A simple DBN to encode Conway's cellular automata game of life on a grid."><br><code>GameOfLife_MDP_</code><br><code>ippc2011</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/GameOfLife_POMDP_ippc2011.png" width="180" alt="A simple DBN to encode Conway's cellular automata game of life on a grid. This is the pomdp version." title="A simple DBN to encode Conway's cellular automata game of life on a grid. This is the pomdp version."><br><code>GameOfLife_POMDP_</code><br><code>ippc2011</code><br><sub>instances 1&ndash;10</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/Navigation_MDP_ippc2011.png" width="180" alt="In a grid, a robot must get to a goal G, and every cell offers the robot a (different) chance of disappearing." title="In a grid, a robot must get to a goal G, and every cell offers the robot a (different) chance of disappearing."><br><code>Navigation_MDP_</code><br><code>ippc2011</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Navigation_POMDP_ippc2011.png" width="180" alt="In a grid, a robot must get to a goal G, and every cell offers the robot a (different) chance of disappearing. This is the pomdp version." title="In a grid, a robot must get to a goal G, and every cell offers the robot a (different) chance of disappearing. This is the pomdp version."><br><code>Navigation_POMDP_</code><br><code>ippc2011</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/SkillTeaching_MDP_ippc2011.png" width="180" alt="The agent is trying to teach a series of skills to a student through the use of hints and multiple choice questions." title="The agent is trying to teach a series of skills to a student through the use of hints and multiple choice questions."><br><code>SkillTeaching_MDP_</code><br><code>ippc2011</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/SkillTeaching_POMDP_ippc2011.png" width="180" alt="The agent is trying to teach a series of skills to a student through the use of hints and multiple choice questions. This is the pomdp version." title="The agent is trying to teach a series of skills to a student through the use of hints and multiple choice questions. This is the pomdp version."><br><code>SkillTeaching_POMDP_</code><br><code>ippc2011</code><br><sub>instances 1&ndash;10</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/SysAdmin_MDP_ippc2011.png" width="180" alt="An example RDDL description for the well-known SysAdmin problem." title="An example RDDL description for the well-known SysAdmin problem."><br><code>SysAdmin_MDP_ippc2011</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/SysAdmin_POMDP_ippc2011.png" width="180" alt="An example RDDL description for the well-known SysAdmin problem. This is the pomdp version" title="An example RDDL description for the well-known SysAdmin problem. This is the pomdp version"><br><code>SysAdmin_POMDP_</code><br><code>ippc2011</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Traffic_CTM_MDP_ippc2011.png" width="180" alt="A simple binary version of the cell transition model (CTM) for modeling traffic." title="A simple binary version of the cell transition model (CTM) for modeling traffic."><br><code>Traffic_CTM_MDP_</code><br><code>ippc2011</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Traffic_CTM_POMDP_ippc2011.png" width="180" alt="A simple binary version of the cell transition model (CTM) for modeling traffic. This is the pomdp version" title="A simple binary version of the cell transition model (CTM) for modeling traffic. This is the pomdp version"><br><code>Traffic_CTM_POMDP_</code><br><code>ippc2011</code><br><sub>instances 1&ndash;10</sub></td></tr>
</table>

<sub>[back to contexts](#domain-gallery)</sub>

### ippc2014

The International Probabilistic Planning Competition of 2014 ([site](https://ssanner.github.io/IPPC_2014/)).

<table>
<tr><td align="center" valign="top"><img src="docs/thumbnails/AcademicAdvising_MDP_ippc2014.png" width="180" alt="In this domain, a student may take courses at a given cost and passes the course with a probability determined by how many of the prerequisites they have successfully passed." title="In this domain, a student may take courses at a given cost and passes the course with a probability determined by how many of the prerequisites they have successfully passed."><br><code>AcademicAdvising_MDP_</code><br><code>ippc2014</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/AcademicAdvising_POMDP_ippc2014.png" width="180" alt="In this domain, a student may take courses at a given cost and passes the course with a probability determined by how many of the prerequisites they have successfully passed. This is the pomdp version" title="In this domain, a student may take courses at a given cost and passes the course with a probability determined by how many of the prerequisites they have successfully passed. This is the pomdp version"><br><code>AcademicAdvising_</code><br><code>POMDP_ippc2014</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/CrossingTraffic_MDP_ippc2014.png" width="180" alt="In a grid, a robot must get to a goal and avoid obstacles arriving randomly and moving left." title="In a grid, a robot must get to a goal and avoid obstacles arriving randomly and moving left."><br><code>CrossingTraffic_MDP_</code><br><code>ippc2014</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/CrossingTraffic_POMDP_ippc2014.png" width="180" alt="In a grid, a robot must get to a goal and avoid obstacles arriving randomly and moving left. This is the pomdp version" title="In a grid, a robot must get to a goal and avoid obstacles arriving randomly and moving left. This is the pomdp version"><br><code>CrossingTraffic_POMDP_</code><br><code>ippc2014</code><br><sub>instances 1&ndash;10</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/Elevators_MDP_ippc2014.png" width="180" alt="This domain has a number of elevators delivering passengers to either the top or the bottom floor." title="This domain has a number of elevators delivering passengers to either the top or the bottom floor."><br><code>Elevators_MDP_ippc2014</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Elevators_POMDP_ippc2014.png" width="180" alt="This domain has a number of elevators delivering passengers to either the top or the bottom floor. This is the pomdp" title="This domain has a number of elevators delivering passengers to either the top or the bottom floor. This is the pomdp"><br><code>Elevators_POMDP_</code><br><code>ippc2014</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/SkillTeaching_MDP_ippc2014.png" width="180" alt="In this domain, the agent is trying to teach a series of skills to a student through the use of hints and multiple choice questions." title="In this domain, the agent is trying to teach a series of skills to a student through the use of hints and multiple choice questions."><br><code>SkillTeaching_MDP_</code><br><code>ippc2014</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/SkillTeaching_POMDP_ippc2014.png" width="180" alt="In this domain, the agent is trying to teach a series of skills to a student through the use of hints and multiple choice questions. This is the pomdp version" title="In this domain, the agent is trying to teach a series of skills to a student through the use of hints and multiple choice questions. This is the pomdp version"><br><code>SkillTeaching_POMDP_</code><br><code>ippc2014</code><br><sub>instances 1&ndash;10</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/Tamarisk_MDP_ippc2014.png" width="180" alt="The agent manages the spread of an invasive plant species, by manually intervening to eridaticate them or restore the native species." title="The agent manages the spread of an invasive plant species, by manually intervening to eridaticate them or restore the native species."><br><code>Tamarisk_MDP_ippc2014</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Tamarisk_POMDP_ippc2014.png" width="180" alt="The agent manages the spread of an invasive plant species, by manually intervening to eridaticate them or restore the native species. This is the pomdp version" title="The agent manages the spread of an invasive plant species, by manually intervening to eridaticate them or restore the native species. This is the pomdp version"><br><code>Tamarisk_POMDP_</code><br><code>ippc2014</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Traffic_MDP_ippc2014.png" width="180" alt="A simple binary version of the cell transition model (CTM) for modeling traffic." title="A simple binary version of the cell transition model (CTM) for modeling traffic."><br><code>Traffic_MDP_ippc2014</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Traffic_POMDP_ippc2014.png" width="180" alt="A simple binary version of the cell transition model (CTM) for modeling traffic. This is the pomdp verion" title="A simple binary version of the cell transition model (CTM) for modeling traffic. This is the pomdp verion"><br><code>Traffic_POMDP_ippc2014</code><br><sub>instances 1&ndash;10</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/TriangleTireworld_MDP_ippc2014.png" width="180" alt="In short, this problem was intended to be difficult for determinization/replanning approaches since the highest probability path to the goal is longer than other lower probability (but still possible) paths to the goal." title="In short, this problem was intended to be difficult for determinization/replanning approaches since the highest probability path to the goal is longer than other lower probability (but still possible) paths to the goal."><br><code>TriangleTireworld_MDP_</code><br><code>ippc2014</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/TriangleTireworld_POMDP_ippc2014.png" width="180" alt="In short, this problem was intended to be difficult for determinization/replanning approaches since the highest probability path to the goal is longer than other lower probability (but still possible) paths to the goal. This is the pomdp version" title="In short, this problem was intended to be difficult for determinization/replanning approaches since the highest probability path to the goal is longer than other lower probability (but still possible) paths to the goal. This is the pomdp version"><br><code>TriangleTireworld_</code><br><code>POMDP_ippc2014</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Wildfire_MDP_ippc2014.png" width="180" alt="A boolean version of the wildfire fighting domain." title="A boolean version of the wildfire fighting domain."><br><code>Wildfire_MDP_ippc2014</code><br><sub>instances 1&ndash;10</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Wildfire_POMDP_ippc2014.png" width="180" alt="A boolean version of the wildfire fighting domain. This is the pomdp version" title="A boolean version of the wildfire fighting domain. This is the pomdp version"><br><code>Wildfire_POMDP_</code><br><code>ippc2014</code><br><sub>instances 1&ndash;10</sub></td></tr>
</table>

<sub>[back to contexts](#domain-gallery)</sub>

### ippc2018

The International Probabilistic Planning Competition of 2018 ([site](https://ipc2018-probabilistic.bitbucket.io/)).

<table>
<tr><td align="center" valign="top"><img src="docs/thumbnails/AcademicAdvising_ippc2018.png" width="180" alt="In this domain, a student may take courses at a given cost and passes the course with a probability determined by how many of the prerequisites they have successfully passed." title="In this domain, a student may take courses at a given cost and passes the course with a probability determined by how many of the prerequisites they have successfully passed."><br><code>AcademicAdvising_</code><br><code>ippc2018</code><br><sub>instances 1&ndash;20</sub></td><td align="center" valign="top"><img src="docs/thumbnails/ChromaticDice_ippc2018.png" width="180" alt="Chromatic Dice is a variant of the popular dice game Yahtzee (also known as Kniffel)." title="Chromatic Dice is a variant of the popular dice game Yahtzee (also known as Kniffel)."><br><code>ChromaticDice_ippc2018</code><br><sub>instances 1&ndash;20</sub></td><td align="center" valign="top"><img src="docs/thumbnails/CooperativeRecon_ippc2018.png" width="180" alt="In this domain, the planner controls one or more planetary rovers that examine objects of interest in order to detect life and take a picture of it." title="In this domain, the planner controls one or more planetary rovers that examine objects of interest in order to detect life and take a picture of it."><br><code>CooperativeRecon_</code><br><code>ippc2018</code><br><sub>instances 1&ndash;20</sub></td><td align="center" valign="top"><img src="docs/thumbnails/EarthObservation_ippc2018.png" width="180" alt="The Earth Observation domain models a satellite orbiting Earth that can take pictures of the landscape below with a camera." title="The Earth Observation domain models a satellite orbiting Earth that can take pictures of the landscape below with a camera."><br><code>EarthObservation_</code><br><code>ippc2018</code><br><sub>instances 1&ndash;20</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/Manufacturer_ippc2018.png" width="180" alt="In this domain, the agent manages a manufacturing company that buys goods to use them in the production of other goods." title="In this domain, the agent manages a manufacturing company that buys goods to use them in the production of other goods."><br><code>Manufacturer_ippc2018</code><br><sub>instances 1&ndash;20</sub></td><td align="center" valign="top"><img src="docs/thumbnails/PushYourLuck_ippc2018.png" width="180" alt="As the name suggest, Push Your Luck is an artificial version of a &quot;push your luck&quot; game like, for instance, Can't Stop." title="As the name suggest, Push Your Luck is an artificial version of a &quot;push your luck&quot; game like, for instance, Can't Stop."><br><code>PushYourLuck_ippc2018</code><br><sub>instances 1&ndash;20</sub></td><td align="center" valign="top"><img src="docs/thumbnails/RedFinnedBlueEye_ippc2018.png" width="180" alt="The Red-finned Blue-eye domain tackles the problem of eradicating the invasive Gambusia from the habitat of the red-finned blue-eye." title="The Red-finned Blue-eye domain tackles the problem of eradicating the invasive Gambusia from the habitat of the red-finned blue-eye."><br><code>RedFinnedBlueEye_</code><br><code>ippc2018</code><br><sub>instances 1&ndash;20</sub></td><td align="center" valign="top"><img src="docs/thumbnails/WildlifePreserve_V1_ippc2018.png" width="180" alt="The aim of the Wildlife Preserve domain is to protect a wildlife preserve from poachers by sending available ranger to areas." title="The aim of the Wildlife Preserve domain is to protect a wildlife preserve from poachers by sending available ranger to areas."><br><code>WildlifePreserve_V1_</code><br><code>ippc2018</code><br>to<br><code>WildlifePreserve_V20_</code><br><code>ippc2018</code><br><sub>instances the variant number</sub></td></tr>
</table>

<sub>[back to contexts](#domain-gallery)</sub>

### ippc2023

The International Probabilistic Planning Competition of 2023 ([site](https://ipc2023.github.io)).

<table>
<tr><td align="center" valign="top"><img src="docs/thumbnails/HVAC_ippc2023.png" width="180" alt="Multi-zone and multi-heater HVAC control problem" title="Multi-zone and multi-heater HVAC control problem"><br><code>HVAC_ippc2023</code><br><sub>instances 0&ndash;7</sub></td><td align="center" valign="top"><img src="docs/thumbnails/MarsRover_ippc2023.png" width="180" alt="Multi Rover Navigation, where a group of agent needs to harvest mineral" title="Multi Rover Navigation, where a group of agent needs to harvest mineral"><br><code>MarsRover_ippc2023</code><br><sub>instances 0&ndash;5</sub></td><td align="center" valign="top"><img src="docs/thumbnails/MountainCar_ippc2023.png" width="180" alt="A simple continuous MDP for the classical mountain car control problem" title="A simple continuous MDP for the classical mountain car control problem"><br><code>MountainCar_ippc2023</code><br><sub>instances 1&ndash;5</sub></td><td align="center" valign="top"><img src="docs/thumbnails/PowerGen_ippc2023.png" width="180" alt="A continuous simple power generation problem loosely modeled on the problem of unit commitment" title="A continuous simple power generation problem loosely modeled on the problem of unit commitment"><br><code>PowerGen_ippc2023</code><br><sub>instances 1&ndash;5</sub></td></tr>
<tr><td align="center" valign="top"><img src="docs/thumbnails/RaceCar_ippc2023.png" width="180" alt="A simple continuous MDP for the racecar problem" title="A simple continuous MDP for the racecar problem"><br><code>RaceCar_ippc2023</code><br><sub>instances 0&ndash;6</sub></td><td align="center" valign="top"><img src="docs/thumbnails/RecSim_ippc2023.png" width="180" alt="A problem of recommendation systems, with consumers and providers" title="A problem of recommendation systems, with consumers and providers"><br><code>RecSim_ippc2023</code><br><sub>instances 0&ndash;7</sub></td><td align="center" valign="top"><img src="docs/thumbnails/Reservoir_ippc2023.png" width="180" alt="Continuous action version of management of the water level in interconnected reservoirs" title="Continuous action version of management of the water level in interconnected reservoirs"><br><code>Reservoir_ippc2023</code><br><sub>instances 1&ndash;5</sub></td><td align="center" valign="top"><img src="docs/thumbnails/UAV_ippc2023.png" width="180" alt="Continuous action space version of multi-UAV problem where a group of UAVs have to reach goal positions in the 3d Space" title="Continuous action space version of multi-UAV problem where a group of UAVs have to reach goal positions in the 3d Space"><br><code>UAV_ippc2023</code><br><sub>instances 1&ndash;5</sub></td></tr>
</table>

<sub>[back to contexts](#domain-gallery)</sub>

<!-- DOMAINS:END -->

To refresh the gallery after adding domains, run ``python generate_domain_list.py --thumbnails``.

## License

This software is distributed under the MIT License.
