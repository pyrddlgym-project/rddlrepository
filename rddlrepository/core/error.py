class RDDLRepoEnvironmentNotExistError(ValueError):
    pass


class RDDLRepoInstanceNotExistError(ValueError):
    pass


class RDDLRepoDomainNotExistError(ValueError):
    pass


class RDDLRepoProblemDuplicationError(ValueError):
    pass


class RDDLRepoInstanceDuplicationError(ValueError):
    pass


class RDDLRepoUnresolvedDependencyError(ValueError):
    pass


class RDDLRepoManifestEmptyError(ValueError):
    pass


class RDDLRepoContextNotExistError(ValueError):
    pass
