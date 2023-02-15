from Manager.RDDLRepoManager import RDDLRepoManager as RDDLRepoManager


HEADER = ['name', 'description', 'location', 'instances', 'viz']
Archiver_Dict = {}

def main():
    manager = RDDLRepoManager(rebuild=True)
    # manager.ListProblems()
    # envInfo = manager.GetProblem('CooperativeRecon_ippc2018')
    envInfo = manager.GetProblem('RaceCar')
    # envInfo.list_instances()
    envInfo.get_domain()
    envInfo.get_instance(0)
    envInfo.get_visualizer()





if __name__ == '__main__':
    main()
