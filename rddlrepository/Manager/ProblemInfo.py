import os.path
import sys
import importlib.util
from typing import List

from .ErrorHandling import RDDLRepoInstanceNotExist, RDDLRepoUnresolvedDependency


sys.path.append(os.path.join('..', 'rddlrepository'))
viz_backend_package_name = 'pyRDDLGym'


class ProblemInfo:
    def __init__(self, problem_data: dict) -> None:
        self.name = problem_data['name']
        self.desc = problem_data['description']
        self.loc = problem_data['location']
        self.instances = problem_data['instances']
        self.viz = problem_data['viz']

    def get_domain(self) -> str:
        path = os.path.join(self.loc, 'domain.rddl')
        return path

    def get_instance(self, num: int) -> str:
        if str(num) not in self.instances:
            raise RDDLRepoInstanceNotExist('problem ' + self.name + ' does not have instance ' + str(num))
        instance = 'instance' + str(num) + '.rddl'
        path = os.path.join(self.loc, instance)
        return path

    def list_instances(self, verbose=False) -> List[str]:
        if verbose:
            print(self.instances)
        return self.instances

    def get_visualizer(self):
        if self.viz == 'None':
            return None

        spec = importlib.util.find_spec(viz_backend_package_name)
        if spec is None:
            raise RDDLRepoUnresolvedDependency(viz_backend_package_name + " is not installed")

        path_to_viz = []
        p = os.path.split(self.loc)
        while p[1] != 'rddlrepository':
            path_to_viz.insert(0, p[1])
            p = os.path.split(p[0])
        path_to_viz.insert(0, p[1])
        path_to_viz = '.'.join(path_to_viz)

        viz = None
        viz_info = self.viz
        if viz_info:
            module, viz_class_name = viz_info.strip().split('.')
            viz_package_name = path_to_viz + '.' + module
            viz_package = __import__(viz_package_name, {}, {}, viz_class_name)
            viz = getattr(viz_package, viz_class_name)
        return viz
