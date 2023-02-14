import os.path
import sys

from .ErrorHandling import RDDLRepoInstanceNotExist
sys.path.append('../rddl-repository')

class ProblemInfo:
    def __init__(self, problem_data):
        self.name = problem_data[0]
        self.desc = problem_data[1]
        self.loc = problem_data[2]
        self.instances = problem_data[3]
        self.viz = problem_data[4]

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

        path_to_viz = self.loc.split('/')
        repo_root = path_to_viz.index('Archive')
        path_to_viz = '.'.join(path_to_viz[repo_root:])
        viz = None
        viz_info = self.viz
        if viz_info:
            module, viz_class_name = viz_info.strip().split('.')
            viz_package_name = path_to_viz + '.' + module
            viz_package = __import__(viz_package_name, {}, {}, viz_class_name)
            viz = getattr(viz_package, viz_class_name)
        return viz
