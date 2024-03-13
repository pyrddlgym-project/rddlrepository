# rddlrepository

Purpose:
- hosts a diverse collection of domain and instance RDDL files, covering problems from a wide range of disciplines
- contains a growing collection of standalone problems, as well as an archive of problems used in the probabilistic and learning track of the International Planning Competitions in previous years.
- contains custom visualizers for a subset of domains, to be used with the pyRDDLGym package
- out-of-the-box compatibility with [pyRDDLGym](https://github.com/pyRDDLGym-project/pyRDDLGym)

## Contents
- [Installation](#installation)
- [Usage](#usage)
    - [Domains](#domains)
    - [Instances](#instances)
    - [Loading the Environment in pyRDDLGym](#loading-the-environment-in-pyrddlgym)
    - [Registering your Own Problems](#registering-your-own-problems)
- [Domains Included](#domains-included)
  
## Installation

We require Python 3.8+ and ``numpy``. To install via pip:

```shell
pip install rddlrepository
```

To install via github

```shell
git clone https://github.com/pyRDDLGym-project/rddlrepository.git
```

To make use of the existing visualizers, you also require [pyRDDLGym](https://github.com/pyRDDLGym-project/pyRDDLGym) (``pip install pyRDDLGym``).

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
print(manager.list_problems_by_context("ippc2018"))     # list all problems from IPPC 2018
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
manager.register_domain("MyDomain", "MyContext", domain_content,
                        desc="a description of this domain", viz="ModuleName.ClassName") 
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

A listing of all domains currently supported in the main branch of rddlrepository can be found [here](https://github.com/pyrddlgym-project/rddlrepository/blob/main/domains.pdf).

<p align="center">
<img src="domains.png" width="100%" margin=1/>
</p>

Domains are broken down into contexts, which includes some standalone domains, as well as domains used in the past competitions:
* IPC 2011: [http://users.cecs.anu.edu.au/~ssanner/IPPC_2011/](http://users.cecs.anu.edu.au/~ssanner/IPPC_2011/)
* IPC 2014: [https://ssanner.github.io/IPPC_2014/](https://ssanner.github.io/IPPC_2014/)
* IPC 2018: [https://ipc2018-probabilistic.bitbucket.io/](https://ipc2018-probabilistic.bitbucket.io/)
* IPC 2023: [https://ipc2023.github.io](https://ipc2023.github.io).
