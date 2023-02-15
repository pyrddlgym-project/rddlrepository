from Manager.RDDLRepoManager import RDDLRepoManager as RDDLRepoManager


HEADER = ['name', 'description', 'location', 'instances', 'viz']
Archiver_Dict = {}

def main():
    manager = RDDLRepoManager(rebuild=True)
    manager.list_problems()
    # envInfo = manager.get_problem('CooperativeRecon_ippc2018')
    envInfo = manager.get_problem('RaceCar')
    envInfo.list_instances()
    envInfo.get_domain()
    envInfo.get_instance(0)
    envInfo.get_visualizer()





if __name__ == '__main__':
    main()
