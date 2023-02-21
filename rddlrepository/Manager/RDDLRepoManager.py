import copy
import os
import importlib
import csv
from typing import List

from .ErrorHandling import RDDLRepoDomainNotExist, RDDLRepoProblemDuplication, RDDLRepoManifestEmpty,RDDLRepoContextNotExist
from .ProblemInfo import ProblemInfo

HEADER = ['name', 'description', 'location', 'instances', 'viz', 'context', 'tags']
manifest = 'manifest.csv'


class RDDLRepoManager:
    def __init__(self, rebuild=False) -> None:
        self.archiver_dict= {}
        self.archive_by_context = {}
        self.manager_path = os.path.dirname(os.path.abspath(__file__))
        manifest_path = os.path.join(self.manager_path, manifest)
        if os.path.isfile(manifest_path) and not rebuild:
            try:
                self._load_repo()    # load repo to dict
            except:
                raise RDDLRepoManifestEmpty('An error ocurred while loading current repo manifest, please try to re-run with rebuild=True')
        else:
            self._build_repo()   # build repo and load to dict

    def list_problems(self, verbose=False) -> List[str]:
        problem_list = []
        if len(self.archiver_dict) == 0:
            raise RDDLRepoManifestEmpty('Repo manifest is empty please re-run with rebuild=True')
        for key, values in self.archiver_dict.items():
            if verbose:
                print(key + ": " + values['description'])
            problem_list.append(key)
        return problem_list

    def list_context(self, verbose=False) -> List[str]:
        context_list = []
        if len(self.archive_by_context) == 0:
            raise RDDLRepoManifestEmpty('Repo manifest is empty please re-run with rebuild=True')
        for key, _ in self.archive_by_context.items():
            if verbose:
                print(key)
            context_list.append(key)
        return context_list

    def list_problems_by_context(self, context: str, verbose=False) -> List[str]:
        if context not in self.archive_by_context:
            raise RDDLRepoContextNotExist('context: ' + context + ' does not exist in the RDDL repo')
        problems = '\n'.join(self.archive_by_context[context])
        if verbose:
            print(problems)
        problems_list = copy.deepcopy(self.archive_by_context[context])
        return problems_list

    def get_problem(self, name: str) -> ProblemInfo:
        if name in self.archiver_dict.keys():
            return ProblemInfo(self.archiver_dict[name])
        else:
            raise RDDLRepoDomainNotExist('Domain: ' + name + ' does not exists in the repository')

    def _build_repo(self) -> None:
        root_path = os.path.dirname(os.path.abspath(__file__))
        path_to_manifest = os.path.join(root_path, manifest)
        root_path = os.path.split(root_path)[0]
        archive_dir = os.path.join(root_path,'Archive')
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
                d = self.__split_path_to_list(dir)
                module = 'rddlrepository.Archive' + '.' + '.'.join(d)
                mymodule = importlib.import_module(module)
                context = mymodule.info['context']
                if context:
                    context = '_' + context
                name = mymodule.info['name'] + context
                if name in self.archiver_dict.keys():
                    raise RDDLRepoProblemDuplication('domain: ' + name + ' already exists, problem names must be unique')
                if 'domain.rddl' not in files:
                    raise RDDLRepoDomainNotExist('domain: ' + name + ' does not have a domain.rddl file')
                instances = [fname[8:-5] for fname in files
                             if fname.startswith('instance') and fname.endswith('.rddl')]
                instances.sort(key=lambda x: int(x))
                self.archiver_dict[name] = {'name': name,
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
                if context in self.archive_by_context:
                    self.archive_by_context[context].append(name)
                else:
                    self.archive_by_context[context] = [name]


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

        self.archiver_dict= {}
        with open(path_to_manifest) as file:
            reader = csv.reader(file, delimiter=',')
            for i, row in enumerate(reader):
                if i > 0:
                    name, *entries = row
                    self.archiver_dict[name] = dict(zip(HEADER[1:], entries))
                    self.archiver_dict[name]['name'] = name
                    self.archiver_dict[name]['instances'] = (self.archiver_dict[name]['instances']).split(',')
                    if entries[4] == '':
                        context = 'independent'
                    else:
                        context = entries[4]
                    if context in self.archive_by_context:
                        self.archive_by_context[context].append(name)
                    else:
                        self.archive_by_context[context] = [name]
            return self.archiver_dict

    def __split_path_to_list(self, path):
        l = []
        a = os.path.split(path)
        while a[1] != '':
            l.insert(0, a[1])
            a = os.path.split(a[0])
        return l

