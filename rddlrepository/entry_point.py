import argparse


EPILOG = 'For complete documentation, see https://pyrddlgym.readthedocs.io/en/latest/rddlrepo.html.'

def main():
    parser = argparse.ArgumentParser(prog='rddlrepo',
                                     description="command line parser for the rddlrepository",
                                     epilog=EPILOG)
    subparsers = parser.add_subparsers(dest="rddlrepo", required=True)

    # rebuilding
    parser_build = subparsers.add_parser("build", 
                                         help="(re)build the manifest file that archives all problem paths",
                                         epilog=EPILOG)
    
    # listing
    parser_list = subparsers.add_parser("list",
                                        help="list all problems by context",
                                        epilog=EPILOG)

    # dispatch
    args = parser.parse_args()
    if args.rddlrepo == "build":
        from rddlrepository.core.manager import RDDLRepoManager
        manager = RDDLRepoManager(rebuild=True)
        num_context = len(manager.list_contexts())
        num_problems = len(manager.list_problems())
        print(f'Successfully built rddlrepository manifest: '
              f'found {num_problems} problems across {num_context} contexts.')
        
    elif args.rddlrepo == "list":
        from rddlrepository.core.manager import RDDLRepoManager
        manager = RDDLRepoManager()
        print(manager.get_problems_as_string())
        
    else:
        parser.print_help()

if __name__ == "__main__": 
    main()
