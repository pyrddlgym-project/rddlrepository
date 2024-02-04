import os.path
import sys
import importlib.util
from typing import Dict, List

from .error import (
    RDDLRepoInstanceNotExistError,
    RDDLRepoInstanceDuplicationError,
    RDDLRepoUnresolvedDependencyError
)

sys.path.append(os.path.join('..', 'rddlrepository'))

VIZ_BACKEND_PACKAGE_NAME = 'pyRDDLGym'
DOMAIN_NAME = 'domain.rddl'


class ProblemInfo:

    def __init__(self, problem_data: Dict) -> None:
        self.name = problem_data['name']
        self.desc = problem_data['description']
        self.loc = problem_data['location']
        self.instances = problem_data['instances']
        self.viz = problem_data['viz']

    def get_domain(self) -> str:
        path = os.path.join(self.loc, DOMAIN_NAME)
        return path

    def get_instance(self, num: str) -> str:
        if str(num) not in self.instances:
            raise RDDLRepoInstanceNotExistError(
                f'Domain <{self.name}> does not have instance <{num}>.')
        instance = f'instance{num}.rddl'
        path = os.path.join(self.loc, instance)
        return path

    def list_instances(self) -> List[str]:
        return self.instances

    def get_visualizer(self) -> str:
        if self.viz == 'None':
            return None

        # spec = importlib.util.find_spec(VIZ_BACKEND_PACKAGE_NAME)
        # if spec is None:
        #     raise RDDLRepoUnresolvedDependencyError(
        #         f'{VIZ_BACKEND_PACKAGE_NAME} is not installed: '
        #         f'can be installed with \'pip install {VIZ_BACKEND_PACKAGE_NAME}\'.')

        path_to_viz = []
        p = os.path.split(self.loc)
        while p[1] != 'rddlrepository':
            path_to_viz.insert(0, p[1])
            p = os.path.split(p[0])
        path_to_viz.insert(0, p[1])
        path_to_viz = '.'.join(path_to_viz)
        return path_to_viz
    
        # viz = None
        # viz_info = self.viz
        # if viz_info:
        #     module, viz_class_name = viz_info.strip().split('.')
        #     viz_package_name = path_to_viz + '.' + module
        #     viz_package = __import__(viz_package_name, {}, {}, viz_class_name)
        #     viz = getattr(viz_package, viz_class_name)
        # return viz
    
    def __str__(self) -> str:
        attr = self.__dict__
        values = [f'{name}: {value}' for name, value in self.__dict__.items()]
        return '\n'.join(values)
    
    def register_instance(self, num: str, rddl: str) -> None:
        path = os.path.join(self.loc, f'instance{num}.rddl')
        if os.path.exists(path):
            raise RDDLRepoInstanceDuplicationError(
                f'Instance <{num}> already exists in domain <{self.name}>.')
        
        with open(path, 'w') as instance_file:
            instance_file.write(rddl)               
        self.instances.append(num)
        
        print(f'Instance <{num}> was successfully registered in domain <{self.name}>.')
    
