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
                                        help="list all problems by context, or those with given tags",
                                        epilog=EPILOG)
    parser_list.add_argument("--tags", default=None,
                             help="comma-separated tags the problems must all have (see 'rddlrepo tags')")
    parser_list.add_argument("--context", default=None, help="only the problems of this context")

    # the vocabulary of tags
    parser_tags = subparsers.add_parser("tags",
                                        help="list the tags that describe problems",
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
        if args.tags is None and args.context is None:
            print(manager.get_problems_as_string())
        else:
            for name in manager.list_problems(context=args.context, tags=args.tags):
                print(name)

    elif args.rddlrepo == "tags":
        from rddlrepository.core.tags import FACETS
        for facet, (desc, tags) in FACETS.items():
            print(f'{facet} ({desc}):')
            for tag, tag_desc in tags.items():
                print(f'    {tag:<22} {tag_desc}')
        
    else:
        parser.print_help()

if __name__ == "__main__": 
    main()
