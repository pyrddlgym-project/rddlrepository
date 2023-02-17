import os.path
import sys
import importlib.util

from .ErrorHandling import RDDLRepoInstanceNotExist, RDDLRepoUnresolvedDependency

sys.path.append('../rddl-repository')
viz_backend_package_name = 'pyRDDLGym'


class ProblemInfo:
    def __init__(self, problem_data):
        self.name = problem_data['name']
        self.desc = problem_data['description']
        self.loc = problem_data['location']
        self.instances = problem_data['instances']
        self.viz = problem_data['viz']

    def get_domain(self):
        path = os.path.join(self.loc, 'domain.rddl')
        return path

    def get_instance(self, num):
        if str(num) not in self.instances:
            raise RDDLRepoInstanceNotExist()
        instance = 'instance' + str(num) + '.rddl'
        path = os.path.join(self.loc, instance)
        return path

    def list_instances(self):
        print(self.instances)

    def get_visualizer(self):
        if self.viz == 'None':
            return None

        spec = importlib.util.find_spec(viz_backend_package_name)
        if spec is None:
            raise RDDLRepoUnresolvedDependency(viz_backend_package_name + " is not installed")

        path_to_viz = self.loc.split('/')
        repo_root = path_to_viz.index('Archive')
        path_to_viz = '.'.join(path_to_viz[repo_root-1:])
        viz = None
        viz_info = self.viz
        if viz_info:
            module, viz_class_name = viz_info.strip().split('.')
            viz_package_name = path_to_viz + '.' + module
            viz_package = __import__(viz_package_name, {}, {}, viz_class_name)
            viz = getattr(viz_package, viz_class_name)
        return viz
