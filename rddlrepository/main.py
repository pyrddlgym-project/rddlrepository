from rddlrepository.core.manager import RDDLRepoManager


def main():

    # initializer repo manager, repo manifest will be loaded or built automatically if not present.
    # rebuild argument force manifest rebuild.
    manager = RDDLRepoManager(rebuild=True)
    
    # list all problems in the repo, unique name is generated based on the name and context of the problem
    print(manager.list_problems())

    for context in manager.list_contexts():
        print(f'context {context}: {manager.list_problems_by_context(context)}')

    # getting a specific problem from the repo.
    envInfo = manager.get_problem('EarthObservation_ippc2018')

    # list available instances in for the problem.
    print(envInfo.list_instances())

    # return path to the domain file to be used/open by external tools.
    print(envInfo.get_domain())

    # return path to the desired instance file to be used/open by external tools.
    print(envInfo.get_instance(1))

    # return the pyRDDLGym visualizer object (pyRDDLGym dependency required) if exists, otherwise return None.
    print(envInfo.get_visualizer())


if __name__ == '__main__':
    main()
