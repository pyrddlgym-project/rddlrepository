# rddlrepository

Purpose:
- hosts a diverse collection of domain and instance RDDL files, covering problems from a wide range of disciplines
- contains a growing collection of standalone problems, as well as an archive of problems used in the probabilistic and learning track of the International Planning Competitions in previous years.
- contains custom visualizers for a subset of domains, to be used with the pyRDDLGym package
- out-of-the-box compatibility with pyRDDLGym

## Installation

We require Python 3.8+ and ``numpy``. To install via pip:

```shell
pip install rddlrepository
```

To install via github

```shell
git clone https://github.com/pyRDDLGym-project/rddlrepository.git
```

To make use of the existing visualizers, you also require [pyRDDLGym](https://github.com/pyRDDLGym-project/pyRDDLGym) (``pip install pyRDDLGym```).

## Usage

### Domains

The core object for extracting domains and their instances is the ``RDDLRepoManager`` object:

```python
from rddlrepository.core.manager import RDDLRepoManager
manager = RDDLRepoManager(rebuild=True)
```

> [!NOTE]
> ``rebuild`` instructs the manager to rebuild the manifest, which is an index containing the locations of all domains and instances for fast access.
> While you do not need this option in normal operation, in case you add your own domains or the manifest becomes corrupt, you can force it to be recreated.

To list all domains in rddlrepository:

```python
print(manager.list_problems())
```

Problems are organized by context (e.g. year of the competition, standalone):

```python
print(manager.list_context())
print(manager.list_problems_by_context("standalone"))   # list all standalone problems
print(manager.list_problems_by_context("ippc2018"))     # list all problems from the IPPC 2018 competition
```

### Instances

The information for a specific domain is a ``ProblemInfo`` instance:

```python
problem_info = manager.get_problem("EarthObservation_ippc2018")
```

To list all the instances of a domain:

```python
print(problem_info.list_instances())
```

To return the paths of the domain and instance (1)

```python
print(problem_info.get_domain())
print(problem_info.get_instance("1"))
```

To return the pyRDDLGym visualizer class:

```python
viz_class = problem_info.get_visualizer()
```

### Loading the Environment in pyRDDLGym 

This information can be used in pyRDDLGym to instantiate an OpenAI Gym environment:

```python
import pyRDDLGym
env = pyRDDLGym.make(domain=problem_info.get_domain(), instance=problem_info.get_instance("1"))
env.set_visualizer(problem_info.get_visualizer())
```

However, a shorter way is:

```python
import pyRDDLGym
env = pyRDDLGym.make("EarthObservation_ippc2018", "1")
```

In either case, ``env`` is an instance of OpenAI Gym's ``Env`` class with the ``reset()`` and ``step()`` function already implemented for you!

### Registering your Own Problems

To register a new context in rddlrepository for later access:

```python
manager.register_context("MyContext")
```

To register a new problem in a given context for later access:

```python
domain_content = """
    domain ... {
    ...
    }
"""
manager.register_domain("MyDomain", "MyContext", domain_content, desc="a description of this domain", viz="ModuleName.ClassName") 
```

Here, ```"ModuleName.ClassName"``` refers to the Module name and the Class name of the visualizer (optional).

To register an instance for an existing domain for later access:

```python
instance_content = """
    instance ... {
    ...
    }
"""
problem_info.register_instance("MyInstance", instance_content)
```

## Domains Included

Domains are broken down into contexts, which includes the following past competitions:
* IPC 2011: [http://users.cecs.anu.edu.au/~ssanner/IPPC_2011/](http://users.cecs.anu.edu.au/~ssanner/IPPC_2011/)
* IPC 2014: [https://ssanner.github.io/IPPC_2014/](https://ssanner.github.io/IPPC_2014/)
* IPC 2018: [https://ipc2018-probabilistic.bitbucket.io/](https://ipc2018-probabilistic.bitbucket.io/)
as well as some standalone domains.

The IPC 2011 (context ``ippc2011``) domains are stochastic, discrete-state and discrete-action problems:

|        **Identifier **        	|      **Name**     	| **MDP/POMDP** 	| **Instances** 	|
|:-----------------------------:	|:-----------------:	|:-------------:	|:-------------:	|
| CooperativeRecon_MDP_ippc2011 	| Cooperative Recon 	| Both          	| 1, 2... 10    	|
| CrossingTraffic_MDP_ippc2011  	| Crossing Traffic  	| Both          	| 1, 2... 10    	|
| Elevators_MDP_ippc2011        	| Elevators         	| Both          	| 1, 2... 10    	|
| GameOfLife_MDP_ippc2011       	| Game of Life      	| Both          	| 1, 2... 10    	|
| Navigation_MDP_ippc2011       	| Navigation        	| Both          	| 1, 2... 10    	|
| SkillTeaching_MDP_ippc2011    	| Skill Teaching    	| Both          	| 1, 2... 10    	|
| SysAdmin_MDP_ippc2011         	| Sys Admin         	| Both          	| 1, 2... 10    	|
| Traffic_MDP_ippc2011          	| Traffic           	| Both          	| 1, 2... 10    	|

The IPC 2014 (context ``ippc2014``) domains feature some overlap with IPC 2011, and are also stochastic and discrete:

|        **Identifier **        	|      **Name**     	| **MDP/POMDP** 	| **Instances** 	|
|:-----------------------------:	|:-----------------:	|:-------------:	|:-------------:	|
| AcademicAdvising_MDP_ippc2014 	| Academic Advising 	| Both          	| 1, 2... 10    	|
| CrossingTraffic_MDP_ippc2014  	| Crossing Traffic  	| Both          	| 1, 2... 10    	|
| Elevators_MDP_ippc2014        	| Elevators         	| Both          	| 1, 2... 10    	|
| SkillTeaching_MDP_ippc2014       	| Skill Teaching      	| Both          	| 1, 2... 10    	|
| Tamarisk_MDP_ippc2014       	    | Tamarisk           	| Both          	| 1, 2... 10    	|
| Traffic_MDP_ippc2014    	        | Traffic           	| Both          	| 1, 2... 10    	|
| TriangleTireworld_MDP_ippc2014    | Triangle Tireworld 	| Both          	| 1, 2... 10    	|
| Wildfire_MDP_ippc2014          	| Wildfire           	| Both          	| 1, 2... 10    	|

The IPC 2018 (context ``ippc2018``) domains generally feature new language features of RDDL (e.g. enums), and are generally more complex than previous years':

|        **Identifier **        	|      **Name**     	| **MDP/POMDP** 	| **Instances** 	|
|:-----------------------------:	|:-----------------:	|:-------------:	|:-------------:	|
| AcademicAdvising_MDP_ippc2018 	| Academic Advising 	| MDP            	| 1, 2... 20    	|
| ChromaticDice_MDP_ippc2018 	    | Chromatic Dice    	| MDP            	| 1, 2... 20    	|
| CooperativeRecon_MDP_ippc2018     | Cooperative Recon   	| MDP            	| 1, 2... 20    	|
| EarthObservation_MDP_ippc2018     | Earth Observation   	| MDP            	| 1, 2... 20    	|
| Manufacturer_MDP_ippc2018       	| Manufacturer       	| MDP            	| 1, 2... 20    	|
| PushYourLuck_MDP_ippc2018    	    | Push Your Luck        | MDP            	| 1, 2... 20    	|
| RedFinnedBlueEye_MDP_ippc2018     | Red Finned Blue Eye	| MDP            	| 1, 2... 20    	|
| WildlifePreserve_MDP_ippc2018   	| Wildlife Preserve   	| MDP            	| 1, 2... 20    	|

The IPC 2023 (context ``ippc2023``) domains are a departure from previous years' since they are predominantly continuous-state or action, or hybrid:

|        **Identifier **        	|      **Name**     	| **MDP/POMDP** 	| **Instances** 	|
|:-----------------------------:	|:-----------------:	|:-------------:	|:-------------:	|
| HVAC_ippc2023 	                | HVAC 	                | MDP            	| 1, 2... 5    	    |
| MarsRover_ippc2023 	            | Mars Rover        	| MDP            	| 1, 2... 5      	|
| MountainCar_ippc2023              | Mountain Car   	    | MDP            	| 1, 2... 5    	    |
| PowerGen_ippc2023                 | Power Generation   	| MDP            	| 1, 2... 5    	    |
| RaceCar_ippc2023       	        | Race Car       	    | MDP            	| 1, 2... 5    	    |
| RecSim_ippc2023    	            | Recommendation System | MDP            	| 1, 2... 5    	    |
| Reservoir_ippc2023                | Reservoir          	| MDP            	| 1, 2... 5       	|
| UAV_continuous_ippc2023   	    | UAV   	            | MDP            	| 1, 2... 5    	    |
