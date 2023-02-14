import os
import importlib

class RDDLEnvironmentNotExist(ValueError):
    pass

class RDDLInstanceNotExist(ValueError):
    pass

class RDDLDomainNotExist(ValueError):
    pass


HEADER = ['name', 'description', 'location', 'instances', 'viz']


class RDDLManager:
    def __init__(self):
        self.Archiver_Dict = {}
        pass

    def get_domain(self):
        pass

    def get_instance(self):
        pass

    def list_instances(self):
        pass

    @staticmethod
    def BuildRepo():
        Archiver_Dict = {}
        archive_dir = os.path.dirname(os.path.abspath(__file__)) + '/Archive'
        start_char = len(archive_dir)
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
                if 'domain.rddl' not in files:
                    raise RDDLDomainNotExist()
                instances = [fname[8:-5] for fname in files
                             if fname.startswith('instance') and fname.endswith('.rddl')]
                Archiver_Dict[mymodule.info['name']] = [mymodule.info['name'],
                                                        mymodule.info['description'],
                                                        root,
                                                        instances,
                                                        mymodule.info['viz']]
        print(Archiver_Dict)
        #TODO generate manifest
