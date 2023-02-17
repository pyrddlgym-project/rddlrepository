import os
import importlib
import csv

from .ErrorHandling import RDDLRepoDomainNotExist, RDDLRepoProblemDuplication, RDDLRepoManifestEmpty,RDDLRepoContextNotExist
from .ProblemInfo import ProblemInfo

HEADER = ['name', 'description', 'location', 'instances', 'viz', 'context', 'tags']
manifest = 'manifest.csv'


class RDDLRepoManager:
    def __init__(self, rebuild=False):
        self.archiver_dict= {}
        self.archive_by_context = {}
        self.manager_path = os.path.dirname(os.path.abspath(__file__))
        if os.path.isfile(self.manager_path+'/manifest.csv') and not rebuild:
            try:
                self._load_repo()    # load repo to dict
            except:
                raise RDDLRepoManifestEmpty('An error ocurred while loading current repo manifest, please try to re-run with rebuild=True')
        else:
            self._build_repo()   # build repo and load to dict

    def list_problems(self):
        if len(self.archiver_dict) == 0:
            raise RDDLRepoManifestEmpty('Repo manifest is empty please re-run with rebuild=True')
        for key, values in self.archiver_dict.items():
            print(key + ": " + values['description'])

    def list_context(self):
        if len(self.archive_by_context) == 0:
            raise RDDLRepoManifestEmpty('Repo manifest is empty please re-run with rebuild=True')
        for key, _ in self.archive_by_context.items():
            print(key)

    def list_problems_by_context(self, context):
        if context not in self.archive_by_context:
            raise RDDLRepoContextNotExist('context: ' + context + ' does not exist in the RDDL repo')
        problems = '\n'.join(self.archive_by_context[context])
        # problems = '\n'.join(problems)
        print(problems)

    def get_problem(self, name):
        if name in self.archiver_dict.keys():
            return ProblemInfo(self.archiver_dict[name])
        else:
            raise RDDLRepoDomainNotExist('Domain: ' + name + ' does not exists in the repository')

    def _build_repo(self):
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
                module = 'rddlrepository.Archive' + '.'.join(d)
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
                self.archiver_dict[name] = {'name': mymodule.info['name'],
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
                values['instances'] = ','.join(values['instances'])
                values['tags'] = ','.join(values['tags'])
                row = [values[key] for key in HEADER]
                writer.writerow(row)

    def _load_repo(self):
        root_path = os.path.dirname(os.path.abspath(__file__))
        path_to_manifest = os.path.join(root_path, 'manifest.csv')
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
                    if entries[4] == '':
                        context = 'independent'
                    else:
                        context = entries[4]
                    if context in self.archive_by_context:
                        self.archive_by_context[context].append(name)
                    else:
                        self.archive_by_context[context] = [name]
            return self.archiver_dict

