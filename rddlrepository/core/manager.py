import copy
import os
import importlib
import csv
from typing import List

from .error import (
    RDDLRepoDomainNotExistError,
    RDDLRepoProblemDuplicationError,
    RDDLRepoManifestEmptyError,
    RDDLRepoContextNotExistError
)
from .info import ProblemInfo

HEADER = ['name', 'description', 'location', 'instances', 'viz', 'context', 'tags']
manifest = 'manifest.csv'

ARCHIVE_FOLDER = 'rddlrepository.Archive'
DOMAIN_NAME = 'domain.rddl'


class RDDLRepoManager:

    def __init__(self, rebuild=False) -> None:
        self.archiver_dict = {}
        self.archive_by_context = {}
        
        self.manager_path = os.path.dirname(os.path.abspath(__file__))
        manifest_path = os.path.join(self.manager_path, manifest)
        if os.path.isfile(manifest_path) and not rebuild:
            try:
                self._load_repo()
            except:
                raise RDDLRepoManifestEmpty(
                    'An error ocurred while loading current repo manifest, '
                    'please try to re-run with rebuild=True')
        else:
            self._build_repo()

    def list_problems(self) -> List[str]:
        problem_list = []
        if len(self.archiver_dict) == 0:
            raise RDDLRepoManifestEmpty(
                'Repository manifest is empty please re-run with rebuild=True')        
        return list(self.archiver_dict.keys())

    def list_context(self) -> List[str]:
        if not self.archive_by_context:
            raise RDDLRepoManifestEmpty(
                'Repository manifest is empty please re-run with rebuild=True')        
        return list(self.archive_by_context.keys())

    def list_problems_by_context(self, context: str) -> List[str]:
        info = self.archive_by_context.get(context, None)
        if info is None:
            raise RDDLRepoContextNotExist(
                f'Context: {context} does not exist in the RDDL repo')
        return copy.deepcopy(info)
        
    def get_problem(self, name: str) -> ProblemInfo:
        info = self.archiver_dict.get(name, None)
        if info is None:
            raise RDDLRepoDomainNotExist(
                f'Domain: {name} does not exists in the repository')        
        return ProblemInfo(info)            

    def _build_repo(self) -> None:
        root_path = os.path.dirname(os.path.abspath(__file__))
        path_to_manifest = os.path.join(root_path, manifest)
        root_path = os.path.split(root_path)[0]
        archive_dir = os.path.join(root_path, 'Archive')
        start_char = len(archive_dir)

        # build the repo dictionary as first step to verify correctness and uniqueness
        for root, dirs, files in os.walk(archive_dir, topdown=False):
            dir = root[start_char:]
            if '__pycache__' in dirs:
                dirs.remove('__pycache__')
            if len(dirs) > 0:
                continue
            d = os.path.split(root_path)
            if d[1] == '__pycache__':
                continue
            
            if "__init__.py" in files:
                d = self._split_path_to_list(dir)
                module = ARCHIVE_FOLDER + '.' + '.'.join(d)
                mymodule = importlib.import_module(module)
                context = mymodule.info['context']
                if context:
                    context = '_' + context
                name = mymodule.info['name'] + context
                
                if name in self.archiver_dict.keys():
                    raise RDDLRepoProblemDuplication(
                        f'Domain {name} already exists, problem names must be unique.')
                if DOMAIN_NAME not in files:
                    raise RDDLRepoDomainNotExist(
                        f'domain {name} does not have a domain.rddl file')
                    
                instances = [fname[8:-5] for fname in files
                             if fname.startswith('instance') and fname.endswith('.rddl')]
                instances.sort(key=lambda x: int(x))
                self.archiver_dict[name] = {
                    'name': name,
                    'description': mymodule.info['description'],
                    'location': root,
                    'instances': instances,
                    'viz': mymodule.info['viz'],
                    'context': mymodule.info['context'],
                    'tags': mymodule.info['tags']
                }

                context = mymodule.info['context']
                if context == '':
                    context = 'independent'
                self.archive_by_context.setdefault(context, []).append(name)

        # Generate manifest
        with open(path_to_manifest, 'w', newline='') as file:

            # write the csv header of the manifest
            writer = csv.writer(file, delimiter=',')
            writer.writerow(HEADER)

            # iterate through the dictionary
            for keys, values in self.archiver_dict.items():
                values_copy = copy.deepcopy(values)
                values_copy['instances'] = ','.join(values['instances'])
                values_copy['tags'] = ','.join(values['tags'])
                row = [values_copy[key] for key in HEADER]
                writer.writerow(row)

    def _load_repo(self) -> dict:
        root_path = os.path.dirname(os.path.abspath(__file__))
        path_to_manifest = os.path.join(root_path, manifest)
        if not os.path.isfile(path_to_manifest):
            return {}

        self.archiver_dict = {}
        with open(path_to_manifest) as file:
            reader = csv.reader(file, delimiter=',')
            for i, row in enumerate(reader):
                if i > 0:
                    name, *entries = row
                    domain_info = dict(zip(HEADER[1:], entries))
                    domain_info['name'] = name
                    domain_info['instances'] = domain_info['instances'].split(',')                    
                    self.archiver_dict[name] = domain_ifo
                    if entries[4] == '':
                        context = 'independent'
                    else:
                        context = entries[4]
                    self.archive_by_context.setdefault(context, []).append(name)
            return self.archiver_dict

    def _split_path_to_list(self, path):
        l = []
        a = os.path.split(path)
        while a[1] != '':
            l.insert(0, a[1])
            a = os.path.split(a[0])
        return l

