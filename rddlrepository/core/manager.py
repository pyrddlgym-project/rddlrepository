import copy
from datetime import datetime
import os
import importlib
import csv
from typing import Dict, List

from .error import (
    RDDLRepoDomainNotExistError,
    RDDLRepoProblemDuplicationError,
    RDDLRepoManifestEmptyError,
    RDDLRepoContextNotExistError,
    RDDLRepoContextDuplicationError
)
from .info import ProblemInfo

HEADER = ['name', 'description', 'location', 'instances', 'viz', 'context', 'tags']
manifest = 'manifest.csv'

PACKAGE_NAME = 'rddlrepository'
ARCHIVE_NAME = 'archive'
DOMAIN_NAME = 'domain.rddl'
INFO_NAME = '__init__.py'


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
                raise RDDLRepoManifestEmptyError(
                    'An error occurred while loading the repository manifest, '
                    'please re-run with rebuild=True.')
        else:
            self._build_repo()
    
    # ==========================================================================
    # GETTERS
    # ==========================================================================
    
    @staticmethod
    def _print_columns(values, cols=3):
        width = 1 + max(len(v) for v in values)
        result = ''
        for (count, item) in enumerate(values, 1):
            result += item.ljust(width)
            if count % cols == 0:
                result += '\n'
        return result
        
    def list_problems(self) -> List[str]:
        problem_list = []
        if len(self.archiver_dict) == 0:
            raise RDDLRepoManifestEmptyError(
                'Repository manifest is empty: please re-run with rebuild=True.')        
        return list(self.archiver_dict.keys())

    def list_contexts(self) -> List[str]:
        if not self.archive_by_context:
            raise RDDLRepoManifestEmptyError(
                'Repository manifest is empty: please re-run with rebuild=True.')        
        return list(self.archive_by_context.keys())

    def list_problems_by_context(self, context: str) -> List[str]:
        info = self.archive_by_context.get(context, None)
        if info is None:
            valid_keys = list(self.archive_by_context.keys())
            raise RDDLRepoContextNotExistError(
                f'Context <{context}> does not exist in the repository, '
                f'must be one of:\n' + self._print_columns(valid_keys) + '\n')
        return copy.deepcopy(info)
        
    def get_problem(self, name: str) -> ProblemInfo:
        info = self.archiver_dict.get(name, None)
        if info is None:
            standalone, ippc = [], []
            for key in self.archiver_dict.keys():
                if 'ippc' in key or 'IPPC' in key:
                    ippc.append(key)
                else:
                    standalone.append(key)
            valid_keys = standalone + ippc
            raise RDDLRepoDomainNotExistError(
                f'Domain <{name}> does not exist in the repository, '
                f'must be one of:\n' + self._print_columns(valid_keys) + '\n')        
        return ProblemInfo(info)            
    
    # ==========================================================================
    # MANIFEST HANDLING
    # ==========================================================================
    
    def _build_repo(self) -> None:
        root_path = os.path.dirname(os.path.abspath(__file__))
        path_to_manifest = os.path.join(root_path, manifest)
        root_path = os.path.split(root_path)[0]
        archive_dir = os.path.join(root_path, ARCHIVE_NAME)
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
                module = PACKAGE_NAME + '.' + ARCHIVE_NAME + '.' + '.'.join(d)
                mymodule = importlib.import_module(module)
                context = mymodule.info['context']
                if context:
                    context = '_' + context
                name = mymodule.info['name'] + context
                
                if name in self.archiver_dict.keys():
                    raise RDDLRepoProblemDuplicationError(
                        f'Domain <{name}> already exists: problem names must be unique.')
                if DOMAIN_NAME not in files:
                    raise RDDLRepoDomainNotExistError(
                        f'Domain <{name}> does not have a {DOMAIN_NAME} file.')
                    
                instances = [fname[8:-5] for fname in files
                             if fname.startswith('instance') and fname.endswith('.rddl')]
                instances.sort(key=lambda x: int(x))
                context = mymodule.info['context']
                if context == '':
                    context = 'standalone'
                self.archiver_dict[name] = {
                    'name': name,
                    'description': mymodule.info['description'],
                    'location': root,
                    'instances': instances,
                    'viz': mymodule.info['viz'],
                    # 'context': mymodule.info['context'],
                    'context': context,
                    'tags': mymodule.info['tags']
                }

                # context = mymodule.info['context']
                # if context == '':
                #     context = 'standalone'
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

    def _load_repo(self) -> Dict:
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
                    self.archiver_dict[name] = domain_info
                    if entries[4] == '':
                        context = 'standalone'
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
    
    # ==========================================================================
    # REGISTRATION
    # ==========================================================================
    
    def register_context(self, context: str, refresh: bool=True) -> None:
        root_path = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.split(root_path)[0]
        context_dir = os.path.join(root_path, ARCHIVE_NAME, context)
        
        if context in self.archive_by_context or os.path.isdir(context_dir):
            raise RDDLRepoContextDuplicationError(
                f'Context <{context}> already exists.')
        
        os.mkdir(context_dir)
        open(os.path.join(context_dir, '__init__.py'), 'a').close()
        
        if refresh:
            self.archive_by_context[context] = []
        
        print(f'Context <{context}> was successfully registered in rddlrepository.')
    
    def register_domain(self, name: str, context: str, rddl: str,
                        desc: str=None, viz: str='', refresh: bool=True) -> None:
        domains = self.list_problems_by_context(context)
        if name in domains:
            raise RDDLRepoProblemDuplicationError(
                f'Domain <{name}> already exists in context <{context}>.')
        
        root_path = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.split(root_path)[0]
        domain_dir = os.path.join(root_path, ARCHIVE_NAME, context, name)
        if os.path.isdir(domain_dir):
            raise RDDLRepoProblemDuplicationError(
                f'Domain <{name}> already exists in context <{context}>.')
        
        os.mkdir(domain_dir)
        
        if desc is None:
            desc = (f'User-defined domain with name {name} in context {context}, '
                    f'created on {datetime.today()}.')
        info = {'name': name, 'description': desc,
                'context': context, 'tags': '', 'viz': viz}
        
        with open(os.path.join(domain_dir, INFO_NAME), 'a') as info_file:
            info_file.write(f'info = {info}')
        with open(os.path.join(domain_dir, DOMAIN_NAME), 'a') as domain_file:
            domain_file.write(rddl)
        
        if refresh:
            self.archiver_dict = {}
            self.archive_by_context = {}
            self._build_repo()
            self._load_repo()
        
        print(f'Domain <{name}> was successfully registered in rddlrepository '
              f'with context <{context}>.')
        
