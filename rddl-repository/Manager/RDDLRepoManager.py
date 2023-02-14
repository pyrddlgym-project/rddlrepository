import os
import importlib
import csv

from .ErrorHandling import RDDLRepoDomainNotExist, RDDLRepoProblemDuplication
from .ProblemInfo import ProblemInfo

HEADER = ['name', 'description', 'location', 'instances', 'viz']
manifest = 'manifest.csv'


class RDDLRepoManager:
    def __init__(self, rebuild=False):
        self.Archiver_Dict = {}
        self.manager_path = os.path.dirname(os.path.abspath(__file__))
        if os.path.isfile(self.manager_path+'/manifest.csv') and not rebuild:
            self._LoadRepo()    # load repo to dict
        else:
            self._BuildRepo()   # build repo and load to dict

    def ListProblems(self):
        for key, values in self.Archiver_Dict.items():
            print(key + ": " + values[1])

    def GetProblem(self, name):
        if name in self.Archiver_Dict.keys():
            return ProblemInfo(self.Archiver_Dict[name])

    def _BuildRepo(self):
        root_path = os.path.dirname(os.path.abspath(__file__))
        path_to_manifest = os.path.join(root_path, 'manifest.csv')
        root_path = root_path.split('/')
        root_path = '/'.join(root_path[:-1])
        archive_dir = root_path + '/Archive'
        start_char = len(archive_dir)

        # build the repo dictionary as first step to verify correctness and uniqueness
        for root, dirs, files in os.walk(archive_dir, topdown=False):
            dir = root[start_char:]
            if '__pycache__' in dirs:
                dirs.remove('__pycache__')
            if len(dirs) > 0:
                continue
            d = dir.split('/')
            if d[-1] == '__pycache__':
                continue
            if "__init__.py" in files:
                d = dir.split('/')
                module = 'Archive' + '.'.join(d)
                mymodule = importlib.import_module(module)
                if mymodule.info['name'] in self.Archiver_Dict.keys():
                    raise RDDLRepoProblemDuplication()
                if 'domain.rddl' not in files:
                    raise RDDLRepoDomainNotExist()
                instances = [fname[8:-5] for fname in files
                             if fname.startswith('instance') and fname.endswith('.rddl')]
                context = mymodule.info['context']
                if context:
                    context = '_' + context
                name = mymodule.info['name'] + context
                self.Archiver_Dict[name] = [mymodule.info['name'],
                                                        mymodule.info['description'],
                                                        root,
                                                        instances,
                                                        mymodule.info['viz']]

        # Generate manifest
        with open(path_to_manifest, 'w', newline='') as file:

            # write the csv header of the manifest
            writer = csv.writer(file, delimiter=',')
            writer.writerow(HEADER)

            # iterate through the dictionary
            for keys, values in self.Archiver_Dict.items():
                values[3] = ','.join(values[3])
                writer.writerow(values)

    def _LoadRepo(self):
        root_path = os.path.dirname(os.path.abspath(__file__))
        path_to_manifest = os.path.join(root_path, 'manifest.csv')
        if not os.path.isfile(path_to_manifest):
            return {}

        self.Archiver_Dict = {}
        with open(path_to_manifest) as file:
            reader = csv.reader(file, delimiter=',')
            for i, row in enumerate(reader):
                if i > 0:
                    name, *entries = row
                    self.Archiver_Dict[name] = dict(zip(HEADER[1:], entries))
            return self.Archiver_Dict
