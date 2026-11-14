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

Every problem in the repository, grouped by context, with the identifier to pass to ``make()``, its instances and what it is about.

<!-- DOMAINS:START -->

**127 problems in 10 contexts:** [standalone](#standalone) (14) &middot; [physics](#physics) (14) &middot; [or](#or) (7) &middot; [arcade](#arcade) (13) &middot; [gym](#gym) (6) &middot; [rddlsim](#rddlsim) (6) &middot; [ippc2011](#ippc2011) (16) &middot; [ippc2014](#ippc2014) (16) &middot; [ippc2018](#ippc2018) (27) &middot; [ippc2023](#ippc2023) (8)

### standalone

Domains written for this repository.

| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Domain | Instances | Description |
|:-:|:--|:--|:--|
| <img src="docs/thumbnails/BernoulliBandit.png" width="64" alt=""> | <code>BernoulliBandit</code> | 0 | Bayesian Bernoulli bandit |
| <img src="docs/thumbnails/Elevators.png" width="64" alt=""> | <code>Elevators</code> | 0, 1 | Elevators carry passengers between floors |
| <img src="docs/thumbnails/Epidemic.png" width="64" alt=""> | <code>Epidemic</code> | 0 | Contain an epidemic with interventions |
| <img src="docs/thumbnails/Fishery.png" width="64" alt=""> | <code>Fishery</code> | 0 | Harvest competing fish stocks |
| <img src="docs/thumbnails/HVAC.png" width="64" alt=""> | <code>HVAC</code> | 0, 1 | Heat the zones of a building |
| <img src="docs/thumbnails/Intruders_Continuous.png" width="64" alt=""> | <code>Intruders_<wbr>Continuous</code> | 0 | Detect intruders in an area |
| <img src="docs/thumbnails/Intruders_Discrete.png" width="64" alt=""> | <code>Intruders_<wbr>Discrete</code> | 0 | Detect intruders in an area |
| <img src="docs/thumbnails/Navigation_Continuous.png" width="64" alt=""> | <code>Navigation_<wbr>Continuous</code> | 0 | Navigate around regions to avoid |
| <img src="docs/thumbnails/PowerGen_Continuous.png" width="64" alt=""> | <code>PowerGen_<wbr>Continuous</code> | 0 | Generate power to meet demand |
| <img src="docs/thumbnails/PowerGen_Discrete.png" width="64" alt=""> | <code>PowerGen_<wbr>Discrete</code> | 0 | Generate power to meet demand |
| <img src="docs/thumbnails/Reservoir_Continuous.png" width="64" alt=""> | <code>Reservoir_<wbr>Continuous</code> | 0, 1 | Manage water in linked reservoirs |
| <img src="docs/thumbnails/Reservoir_Discrete.png" width="64" alt=""> | <code>Reservoir_<wbr>Discrete</code> | 0, 1 | Manage water in linked reservoirs |
| <img src="docs/thumbnails/TrafficBLX_ComplexPhases.png" width="64" alt=""> | <code>TrafficBLX_<wbr>ComplexPhases</code> | 0 | Time traffic signals to cut travel time |
| <img src="docs/thumbnails/TrafficBLX_SimplePhases.png" width="64" alt=""> | <code>TrafficBLX_<wbr>SimplePhases</code> | 0 | Time traffic signals to cut travel time |

<sub>[back to contexts](#domain-gallery)</sub>

### physics

Physical systems and control.

| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Domain | Instances | Description |
|:-:|:--|:--|:--|
| <img src="docs/thumbnails/Bicycle_physics.png" width="64" alt=""> | <code>Bicycle_<wbr>physics</code> | 0 | Balance and ride a bicycle |
| <img src="docs/thumbnails/CartPoleChain_physics.png" width="64" alt=""> | <code>CartPoleChain_<wbr>physics</code> | 0, 1 | Swing up a chain of poles on a cart |
| <img src="docs/thumbnails/CartPoleSwingUp_physics.png" width="64" alt=""> | <code>CartPoleSwingUp_<wbr>physics</code> | 0 | Swing up a pole on a cart |
| <img src="docs/thumbnails/Highway_Continuous_physics.png" width="64" alt=""> | <code>Highway_<wbr>Continuous_<wbr>physics</code> | 0&ndash;4 | Drive through highway traffic |
| <img src="docs/thumbnails/Highway_Discrete_physics.png" width="64" alt=""> | <code>Highway_<wbr>Discrete_<wbr>physics</code> | 0&ndash;4 | Drive through highway traffic |
| <img src="docs/thumbnails/Lander_Continuous_physics.png" width="64" alt=""> | <code>Lander_<wbr>Continuous_<wbr>physics</code> | 0&ndash;2 | Land a craft on a pad |
| <img src="docs/thumbnails/Lander_Discrete_physics.png" width="64" alt=""> | <code>Lander_<wbr>Discrete_<wbr>physics</code> | 0&ndash;2 | Land a craft on a pad |
| <img src="docs/thumbnails/Pinball_physics.png" width="64" alt=""> | <code>Pinball_<wbr>physics</code> | 0&ndash;3 | Steer a ball around obstacles |
| <img src="docs/thumbnails/Quadcopter_physics.png" width="64" alt=""> | <code>Quadcopter_<wbr>physics</code> | 0, 1 | Fly a swarm of quadcopters |
| <img src="docs/thumbnails/Reacher_physics.png" width="64" alt=""> | <code>Reacher_<wbr>physics</code> | 0, 1 | Reach a target with a robot arm |
| <img src="docs/thumbnails/Reactor_physics.png" width="64" alt=""> | <code>Reactor_<wbr>physics</code> | 0 | Control a chemical reactor |
| <img src="docs/thumbnails/UAV_Continuous_physics.png" width="64" alt=""> | <code>UAV_<wbr>Continuous_<wbr>physics</code> | 0, 1 | Fly drones to goal positions |
| <img src="docs/thumbnails/UAV_Discrete_physics.png" width="64" alt=""> | <code>UAV_<wbr>Discrete_<wbr>physics</code> | 0 | Fly drones to goal positions |
| <img src="docs/thumbnails/UAV_Mixed_physics.png" width="64" alt=""> | <code>UAV_<wbr>Mixed_<wbr>physics</code> | 0 | Fly drones to goal positions |

<sub>[back to contexts](#domain-gallery)</sub>

### or

Operations research.

| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Domain | Instances | Description |
|:-:|:--|:--|:--|
| <img src="docs/thumbnails/BinPacking_or.png" width="64" alt=""> | <code>BinPacking_<wbr>or</code> | 0 | Pack random items into few bins |
| <img src="docs/thumbnails/CarRental_or.png" width="64" alt=""> | <code>CarRental_<wbr>or</code> | 0, 1 | Move rental cars between locations |
| <img src="docs/thumbnails/Knapsack_or.png" width="64" alt=""> | <code>Knapsack_<wbr>or</code> | 0 | Fill knapsacks with valuable items |
| <img src="docs/thumbnails/Option_or.png" width="64" alt=""> | <code>Option_<wbr>or</code> | 0, 1 | Exercise an American option |
| <img src="docs/thumbnails/StructuralMaintenance_or.png" width="64" alt=""> | <code>StructuralMaintenance_<wbr>or</code> | 0&ndash;4 | Inspect and repair a cracking structure |
| <img src="docs/thumbnails/SupplyChain_or.png" width="64" alt=""> | <code>SupplyChain_<wbr>or</code> | 0 | Ship stock from a factory to warehouses |
| <img src="docs/thumbnails/TSP_or.png" width="64" alt=""> | <code>TSP_<wbr>or</code> | 0 | Travelling salesman tour |

<sub>[back to contexts](#domain-gallery)</sub>

### arcade

Arcade, puzzle and card games.

| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Domain | Instances | Description |
|:-:|:--|:--|:--|
| <img src="docs/thumbnails/Blackjack_arcade.png" width="64" alt=""> | <code>Blackjack_<wbr>arcade</code> | 0 | Draw cards to 21 without going over |
| <img src="docs/thumbnails/BrickBreaker_arcade.png" width="64" alt=""> | <code>BrickBreaker_<wbr>arcade</code> | 0 | Break bricks with a bouncing ball |
| <img src="docs/thumbnails/Eight_arcade.png" width="64" alt=""> | <code>Eight_<wbr>arcade</code> | 0, 1 | Sliding tile puzzle |
| <img src="docs/thumbnails/MazeChase_arcade.png" width="64" alt=""> | <code>MazeChase_<wbr>arcade</code> | 0&ndash;3 | Gather berries in a maze full of wolves |
| <img src="docs/thumbnails/Minesweeper_arcade.png" width="64" alt=""> | <code>Minesweeper_<wbr>arcade</code> | 0&ndash;2 | Reveal safe cells, avoid hidden mines |
| <img src="docs/thumbnails/Pong_arcade.png" width="64" alt=""> | <code>Pong_<wbr>arcade</code> | 0, 1 | Single player pong |
| <img src="docs/thumbnails/Snake_arcade.png" width="64" alt=""> | <code>Snake_<wbr>arcade</code> | 0 | Grow a snake by eating fruit |
| <img src="docs/thumbnails/Sokoban_arcade.png" width="64" alt=""> | <code>Sokoban_<wbr>arcade</code> | 0, 1 | Push boxes onto storage spots |
| <img src="docs/thumbnails/SpaceShooter_arcade.png" width="64" alt=""> | <code>SpaceShooter_<wbr>arcade</code> | 0 | Shoot down waves of aliens |
| <img src="docs/thumbnails/Sudoku_arcade.png" width="64" alt=""> | <code>Sudoku_<wbr>arcade</code> | 0, 1 | Sudoku number puzzle |
| <img src="docs/thumbnails/Tetris_arcade.png" width="64" alt=""> | <code>Tetris_<wbr>arcade</code> | 0, 1 | Classic block stacking game |
| <img src="docs/thumbnails/TowerOfHanoi_arcade.png" width="64" alt=""> | <code>TowerOfHanoi_<wbr>arcade</code> | 0 | Move a tower of disks between rods |
| <img src="docs/thumbnails/Zombies_arcade.png" width="64" alt=""> | <code>Zombies_<wbr>arcade</code> | 0&ndash;3 | Avoid infection by zombies |

<sub>[back to contexts](#domain-gallery)</sub>

### gym

RDDL versions of the classic control environments of Gymnasium.

| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Domain | Instances | Description |
|:-:|:--|:--|:--|
| <img src="docs/thumbnails/Acrobot_gym.png" width="64" alt=""> | <code>Acrobot_<wbr>gym</code> | 0 | Swing up a two-link arm |
| <img src="docs/thumbnails/CartPole_Continuous_gym.png" width="64" alt=""> | <code>CartPole_<wbr>Continuous_<wbr>gym</code> | 0 | Balance a pole on a cart |
| <img src="docs/thumbnails/CartPole_Discrete_gym.png" width="64" alt=""> | <code>CartPole_<wbr>Discrete_<wbr>gym</code> | 0 | Balance a pole on a cart |
| <img src="docs/thumbnails/MountainCar_Continuous_gym.png" width="64" alt=""> | <code>MountainCar_<wbr>Continuous_<wbr>gym</code> | 0 | Drive a weak car up a hill |
| <img src="docs/thumbnails/MountainCar_Discrete_gym.png" width="64" alt=""> | <code>MountainCar_<wbr>Discrete_<wbr>gym</code> | 0 | Drive a weak car up a hill |
| <img src="docs/thumbnails/Pendulum_gym.png" width="64" alt=""> | <code>Pendulum_<wbr>gym</code> | 0 | Swing up and hold a pendulum |

<sub>[back to contexts](#domain-gallery)</sub>

### rddlsim

Examples from rddlsim, the original Java simulator of RDDL.

| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Domain | Instances | Description |
|:-:|:--|:--|:--|
| <img src="docs/thumbnails/ComplexSysAdmin_rddlsim.png" width="64" alt=""> | <code>ComplexSysAdmin_<wbr>rddlsim</code> | 0 | Keep an enhanced computer network running |
| <img src="docs/thumbnails/Logistics_rddlsim.png" width="64" alt=""> | <code>Logistics_<wbr>rddlsim</code> | 0 | Deliver packages by truck |
| <img src="docs/thumbnails/Pizza_rddlsim.png" width="64" alt=""> | <code>Pizza_<wbr>rddlsim</code> | 0 | Deliver pizzas |
|  | <code>PropDBN_<wbr>rddlsim</code> | 0 | Simple propositional network |
| <img src="docs/thumbnails/Sidewalk_rddlsim.png" width="64" alt=""> | <code>Sidewalk_<wbr>rddlsim</code> | 0, 1 | People walking down a sidewalk |
| <img src="docs/thumbnails/Workforce_rddlsim.png" width="64" alt=""> | <code>Workforce_<wbr>rddlsim</code> | 0 | Run a call center |

<sub>[back to contexts](#domain-gallery)</sub>

### ippc2011

The International Probabilistic Planning Competition of 2011 ([site](http://users.cecs.anu.edu.au/~ssanner/IPPC_2011/)).

| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Domain | Instances | Description |
|:-:|:--|:--|:--|
| <img src="docs/thumbnails/CooperativeRecon_MDP_ippc2011.png" width="64" alt=""> | <code>CooperativeRecon_<wbr>MDP_<wbr>ippc2011</code> | 1&ndash;10 | Rovers survey and photograph objects |
| <img src="docs/thumbnails/CooperativeRecon_POMDP_ippc2011.png" width="64" alt=""> | <code>CooperativeRecon_<wbr>POMDP_<wbr>ippc2011</code> | 1&ndash;10 | Rovers survey and photograph objects |
| <img src="docs/thumbnails/CrossingTraffic_MDP_ippc2011.png" width="64" alt=""> | <code>CrossingTraffic_<wbr>MDP_<wbr>ippc2011</code> | 1&ndash;10 | Cross a grid of moving obstacles |
| <img src="docs/thumbnails/CrossingTraffic_POMDP_ippc2011.png" width="64" alt=""> | <code>CrossingTraffic_<wbr>POMDP_<wbr>ippc2011</code> | 1&ndash;10 | Cross a grid of moving obstacles |
| <img src="docs/thumbnails/Elevators_MDP_ippc2011.png" width="64" alt=""> | <code>Elevators_<wbr>MDP_<wbr>ippc2011</code> | 1&ndash;10 | Elevators carry passengers between floors |
| <img src="docs/thumbnails/Elevators_POMDP_ippc2011.png" width="64" alt=""> | <code>Elevators_<wbr>POMDP_<wbr>ippc2011</code> | 1&ndash;10 | Elevators carry passengers between floors |
| <img src="docs/thumbnails/GameOfLife_MDP_ippc2011.png" width="64" alt=""> | <code>GameOfLife_<wbr>MDP_<wbr>ippc2011</code> | 1&ndash;10 | Control Conway's Game of Life |
| <img src="docs/thumbnails/GameOfLife_POMDP_ippc2011.png" width="64" alt=""> | <code>GameOfLife_<wbr>POMDP_<wbr>ippc2011</code> | 1&ndash;10 | Control Conway's Game of Life |
| <img src="docs/thumbnails/Navigation_MDP_ippc2011.png" width="64" alt=""> | <code>Navigation_<wbr>MDP_<wbr>ippc2011</code> | 1&ndash;10 | Reach a goal across risky cells |
| <img src="docs/thumbnails/Navigation_POMDP_ippc2011.png" width="64" alt=""> | <code>Navigation_<wbr>POMDP_<wbr>ippc2011</code> | 1&ndash;10 | Reach a goal across risky cells |
| <img src="docs/thumbnails/SkillTeaching_MDP_ippc2011.png" width="64" alt=""> | <code>SkillTeaching_<wbr>MDP_<wbr>ippc2011</code> | 1&ndash;10 | Teach skills with hints and quizzes |
| <img src="docs/thumbnails/SkillTeaching_POMDP_ippc2011.png" width="64" alt=""> | <code>SkillTeaching_<wbr>POMDP_<wbr>ippc2011</code> | 1&ndash;10 | Teach skills with hints and quizzes |
| <img src="docs/thumbnails/SysAdmin_MDP_ippc2011.png" width="64" alt=""> | <code>SysAdmin_<wbr>MDP_<wbr>ippc2011</code> | 1&ndash;10 | Keep a computer network running |
| <img src="docs/thumbnails/SysAdmin_POMDP_ippc2011.png" width="64" alt=""> | <code>SysAdmin_<wbr>POMDP_<wbr>ippc2011</code> | 1&ndash;10 | Keep a computer network running |
| <img src="docs/thumbnails/Traffic_CTM_MDP_ippc2011.png" width="64" alt=""> | <code>Traffic_<wbr>CTM_<wbr>MDP_<wbr>ippc2011</code> | 1&ndash;10 | Control traffic lights on a road network |
| <img src="docs/thumbnails/Traffic_CTM_POMDP_ippc2011.png" width="64" alt=""> | <code>Traffic_<wbr>CTM_<wbr>POMDP_<wbr>ippc2011</code> | 1&ndash;10 | Control traffic lights on a road network |

<sub>[back to contexts](#domain-gallery)</sub>

### ippc2014

The International Probabilistic Planning Competition of 2014 ([site](https://ssanner.github.io/IPPC_2014/)).

| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Domain | Instances | Description |
|:-:|:--|:--|:--|
| <img src="docs/thumbnails/AcademicAdvising_MDP_ippc2014.png" width="64" alt=""> | <code>AcademicAdvising_<wbr>MDP_<wbr>ippc2014</code> | 1&ndash;10 | Choose courses for a student to graduate |
| <img src="docs/thumbnails/AcademicAdvising_POMDP_ippc2014.png" width="64" alt=""> | <code>AcademicAdvising_<wbr>POMDP_<wbr>ippc2014</code> | 1&ndash;10 | Choose courses for a student to graduate |
| <img src="docs/thumbnails/CrossingTraffic_MDP_ippc2014.png" width="64" alt=""> | <code>CrossingTraffic_<wbr>MDP_<wbr>ippc2014</code> | 1&ndash;10 | Cross a grid of moving obstacles |
| <img src="docs/thumbnails/CrossingTraffic_POMDP_ippc2014.png" width="64" alt=""> | <code>CrossingTraffic_<wbr>POMDP_<wbr>ippc2014</code> | 1&ndash;10 | Cross a grid of moving obstacles |
| <img src="docs/thumbnails/Elevators_MDP_ippc2014.png" width="64" alt=""> | <code>Elevators_<wbr>MDP_<wbr>ippc2014</code> | 1&ndash;10 | Elevators carry passengers between floors |
| <img src="docs/thumbnails/Elevators_POMDP_ippc2014.png" width="64" alt=""> | <code>Elevators_<wbr>POMDP_<wbr>ippc2014</code> | 1&ndash;10 | Elevators carry passengers between floors |
| <img src="docs/thumbnails/SkillTeaching_MDP_ippc2014.png" width="64" alt=""> | <code>SkillTeaching_<wbr>MDP_<wbr>ippc2014</code> | 1&ndash;10 | Teach skills with hints and quizzes |
| <img src="docs/thumbnails/SkillTeaching_POMDP_ippc2014.png" width="64" alt=""> | <code>SkillTeaching_<wbr>POMDP_<wbr>ippc2014</code> | 1&ndash;10 | Teach skills with hints and quizzes |
| <img src="docs/thumbnails/Tamarisk_MDP_ippc2014.png" width="64" alt=""> | <code>Tamarisk_<wbr>MDP_<wbr>ippc2014</code> | 1&ndash;10 | Contain an invasive plant species |
| <img src="docs/thumbnails/Tamarisk_POMDP_ippc2014.png" width="64" alt=""> | <code>Tamarisk_<wbr>POMDP_<wbr>ippc2014</code> | 1&ndash;10 | Contain an invasive plant species |
| <img src="docs/thumbnails/Traffic_MDP_ippc2014.png" width="64" alt=""> | <code>Traffic_<wbr>MDP_<wbr>ippc2014</code> | 1&ndash;10 | Control traffic lights on a road network |
| <img src="docs/thumbnails/Traffic_POMDP_ippc2014.png" width="64" alt=""> | <code>Traffic_<wbr>POMDP_<wbr>ippc2014</code> | 1&ndash;10 | Control traffic lights on a road network |
| <img src="docs/thumbnails/TriangleTireworld_MDP_ippc2014.png" width="64" alt=""> | <code>TriangleTireworld_<wbr>MDP_<wbr>ippc2014</code> | 1&ndash;10 | Drive to a goal, risking flat tires |
| <img src="docs/thumbnails/TriangleTireworld_POMDP_ippc2014.png" width="64" alt=""> | <code>TriangleTireworld_<wbr>POMDP_<wbr>ippc2014</code> | 1&ndash;10 | Drive to a goal, risking flat tires |
| <img src="docs/thumbnails/Wildfire_MDP_ippc2014.png" width="64" alt=""> | <code>Wildfire_<wbr>MDP_<wbr>ippc2014</code> | 1&ndash;10 | Fight spreading wildfires |
| <img src="docs/thumbnails/Wildfire_POMDP_ippc2014.png" width="64" alt=""> | <code>Wildfire_<wbr>POMDP_<wbr>ippc2014</code> | 1&ndash;10 | Fight spreading wildfires |

<sub>[back to contexts](#domain-gallery)</sub>

### ippc2018

The International Probabilistic Planning Competition of 2018 ([site](https://ipc2018-probabilistic.bitbucket.io/)).

| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Domain | Instances | Description |
|:-:|:--|:--|:--|
| <img src="docs/thumbnails/AcademicAdvising_ippc2018.png" width="64" alt=""> | <code>AcademicAdvising_<wbr>ippc2018</code> | 1&ndash;20 | Choose courses for a student to graduate |
| <img src="docs/thumbnails/ChromaticDice_ippc2018.png" width="64" alt=""> | <code>ChromaticDice_<wbr>ippc2018</code> | 1&ndash;20 | Yahtzee-like dice game |
| <img src="docs/thumbnails/CooperativeRecon_ippc2018.png" width="64" alt=""> | <code>CooperativeRecon_<wbr>ippc2018</code> | 1&ndash;20 | Rovers survey and photograph objects |
| <img src="docs/thumbnails/EarthObservation_ippc2018.png" width="64" alt=""> | <code>EarthObservation_<wbr>ippc2018</code> | 1&ndash;20 | A satellite photographs the Earth |
| <img src="docs/thumbnails/Manufacturer_ippc2018.png" width="64" alt=""> | <code>Manufacturer_<wbr>ippc2018</code> | 1&ndash;20 | Run a manufacturing company |
| <img src="docs/thumbnails/PushYourLuck_ippc2018.png" width="64" alt=""> | <code>PushYourLuck_<wbr>ippc2018</code> | 1&ndash;20 | Push-your-luck dice game |
| <img src="docs/thumbnails/RedFinnedBlueEye_ippc2018.png" width="64" alt=""> | <code>RedFinnedBlueEye_<wbr>ippc2018</code> | 1&ndash;20 | Remove an invasive fish species |
| <img src="docs/thumbnails/WildlifePreserve_V1_ippc2018.png" width="64" alt=""> | <code>WildlifePreserve_<wbr>V1_<wbr>ippc2018</code> to <code>WildlifePreserve_<wbr>V20_<wbr>ippc2018</code> | the variant number | Send rangers to stop poachers |

<sub>[back to contexts](#domain-gallery)</sub>

### ippc2023

The International Probabilistic Planning Competition of 2023 ([site](https://ipc2023.github.io)).

| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Domain | Instances | Description |
|:-:|:--|:--|:--|
| <img src="docs/thumbnails/HVAC_ippc2023.png" width="64" alt=""> | <code>HVAC_<wbr>ippc2023</code> | 0&ndash;7 | Heat the zones of a building |
| <img src="docs/thumbnails/MarsRover_ippc2023.png" width="64" alt=""> | <code>MarsRover_<wbr>ippc2023</code> | 0&ndash;5 | Rovers harvest minerals |
| <img src="docs/thumbnails/MountainCar_ippc2023.png" width="64" alt=""> | <code>MountainCar_<wbr>ippc2023</code> | 1&ndash;5 | Drive a weak car up a hill |
| <img src="docs/thumbnails/PowerGen_ippc2023.png" width="64" alt=""> | <code>PowerGen_<wbr>ippc2023</code> | 1&ndash;5 | Generate power to meet demand |
| <img src="docs/thumbnails/RaceCar_ippc2023.png" width="64" alt=""> | <code>RaceCar_<wbr>ippc2023</code> | 0&ndash;6 | Race a car around a track |
| <img src="docs/thumbnails/RecSim_ippc2023.png" width="64" alt=""> | <code>RecSim_<wbr>ippc2023</code> | 0&ndash;7 | Recommend content to users |
| <img src="docs/thumbnails/Reservoir_ippc2023.png" width="64" alt=""> | <code>Reservoir_<wbr>ippc2023</code> | 1&ndash;5 | Manage water in linked reservoirs |
| <img src="docs/thumbnails/UAV_ippc2023.png" width="64" alt=""> | <code>UAV_<wbr>ippc2023</code> | 1&ndash;5 | Fly drones to goal positions |

<sub>[back to contexts](#domain-gallery)</sub>

<!-- DOMAINS:END -->

To refresh the gallery after adding domains, run ``python generate_domain_list.py --thumbnails``.

## License

This software is distributed under the MIT License.
