#rddlrepository

This repository is a collection of all verified RDDL files.
Specifically it is an archive for the problem used in probabilistic and learning track of the International Planning Competitions.

This repository contains also visualizers for the pyRDDLGym framework only, for appropriate domains.

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