#rddlrepository

This repository is a collection of all verified RDDL files.
Specifically it is an archive for the problem used in probabilistic and learning track of the International Planning Competitions.

This repository contains also visualizers for the pyRDDLGym framework only, for appropriate domains.

## Getting started
### Python version
We require Python 3.8+.
### Dependencies
* numpy

If one desires to make use of the included visualizers (note they are pyRDDLGym compatible only), 
[pyRDDLGym](https://github.com/ataitler/pyRDDLGym) must be installed as well.
* pyRDDLGym (`pip install pyRDDLGym`)

### Installation
There are two options:
* Using pip: `pip install rddlrepository`
* Cloning directly: `git clone https://github.com/ataitler/rddlrepository.git`

### Usage example
The following example list all the functions and objects needed to access the problems in the repository:

```python
from rddlrepository.Manager.RDDLRepoManager import RDDLRepoManager as RDDLRepoManager

def main():

    # initializer for the repo manager, repo manifest will be loaded or built automatically if not present.
    manager = RDDLRepoManager()
    
    # rebuild argument force manifest rebuild.
    # manager = RDDLRepoManager(rebuild=True)

    # list all problems (with description) in the repo, unique name is generated based on the name and context of the problem.
    manager.list_problems()
    
    # list all contexts in the database, e.g., independent, ippc2011.
    manager.list_context()

    # list all the problems under a specific context.
    manager.list_problems_by_context('ippc2018')

    # getting a specific problem from the repo.
    envInfo = manager.get_problem('EarthObservation_ippc2018')

    # list available instances available for the problem.
    envInfo.list_instances()

    # return path to the domain file to be used/open by external tools.
    envInfo.get_domain()

    # return path to the desired instance file to be used/open by external tools.
    envInfo.get_instance(1)

    # return the pyRDDLGym visualizer object (pyRDDLGym dependency required) if exists, otherwise return None.
    envInfo.get_visualizer()


if __name__ == '__main__':
    main()
```

## Adding new RDDL problem
In order to add new problem to the repository please follow these instructions:
1. Nest a folder under the Archive folder, in the appropriate place in the hierarchy.
E.g., a competition under the competition folder, standalone problem under Standalone, etc.
2. Make sure you include these three files (at least) in the folder of each problem:
   1. \_\_init\_\_.py
   2. domain.rddl
   3. instance#.rddl (at least one instance)
   4. (optional) pyRDDLGym_visualizer.py
3. In the \_\_init\_\_.py file please include the following dictionary:
```python
info = {
    'name': '<name_of_the_problem>',
    'description': '<description_of_the_problem>',
    'context': '<context_of_the_problem>',
    'tags': '',
    'viz': '<visualizer_class_name>'
}
```
context is the relevant info of where the problem came from or appeared in (it also resolves multiple occurrences).
example of a dictionary for the IPPC2018 AcademicAdvising problem:
```python
info = {
    'name': 'AcademicAdvising',
    'description': 'In this domain, a student may take courses at a given cost and passes the course with a probability determined by how many of the prerequisites they have successfully passed.',
    'context': 'ippc2018',
    'tags': '',
    'viz': 'None'
}
```
5. In case you have nested under the same problem folder multiple folders for different versions of the domain,
please name each of them with a unique identifier. Also, make sure to put a \_\_init\_\_.py file at each level (only the one with the *.rddl should include the info dictionary).
6. Once satisfied run the Repo Manager to rebuild your local manifest file of the repo:
```python
from rddlrepository.Manager.RDDLRepoManager import RDDLRepoManager as RDDLRepoManager

RDDLRepoManager(rebuild=True)
```

## Additional information
More information on the competitions as well as the competition domains in alternative input languages or compilations to (more) restricted RDDL subsets can be found here:

* IPC 2011: [http://users.cecs.anu.edu.au/~ssanner/IPPC_2011/](http://users.cecs.anu.edu.au/~ssanner/IPPC_2011/)
* IPC 2014: [https://ssanner.github.io/IPPC_2014/](https://ssanner.github.io/IPPC_2014/)
* IPC 2018: [https://ipc2018-probabilistic.bitbucket.io/](https://ipc2018-probabilistic.bitbucket.io/)