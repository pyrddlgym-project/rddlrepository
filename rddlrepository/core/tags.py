import re
from typing import Dict, Iterable, List, Tuple

from .error import RDDLRepoTagNotExistError

# facet: (description, {tag: description})
FACETS: Dict[str, Tuple[str, Dict[str, str]]] = {
    'observability': ('what the agent sees', {
        'fully-observable': 'the agent sees the state (no observation fluents)',
        'partially-observable': 'the agent sees only observation fluents (a POMDP)',
    }),
    'dynamics': ('whether anything is random', {
        'deterministic': 'nothing draws a random value',
        'stochastic': 'some CPF or the reward draws a random value',
    }),
    'state': ('the types of the state fluents', {
        'discrete-state': 'every state fluent is bool, int or an object (enum) type',
        'continuous-state': 'every state fluent is real',
        'mixed-state': 'some state fluents are real and some are not',
    }),
    'action': ('the types of the action fluents', {
        'discrete-action': 'every action fluent is bool, int or an object (enum) type',
        'continuous-action': 'every action fluent is real',
        'mixed-action': 'some action fluents are real and some are not',
    }),
}

# tag: facet
TAG_FACET: Dict[str, str] = {
    tag: facet for facet, (_, tags) in FACETS.items() for tag in tags
}

# the distributions of RDDL that draw a random value (KronDelta and DiracDelta are deterministic)
DISTRIBUTIONS = (
    'Uniform', 'Bernoulli', 'Discrete', 'UnnormDiscrete', 'Normal', 'Poisson', 'Exponential',
    'Weibull', 'Gamma', 'Binomial', 'NegativeBinomial', 'Beta', 'Geometric', 'Pareto', 'Student',
    'Gumbel', 'Laplace', 'Cauchy', 'Gompertz', 'ChiSquare', 'Kumaraswamy', 'MultivariateNormal',
    'MultivariateStudent', 'Dirichlet', 'Multinomial'
)


def list_tags() -> Dict[str, str]:
    '''All tags with their descriptions, facet by facet.'''
    return {tag: desc for (_, tags) in FACETS.values() for tag, desc in tags.items()}


def normalize_tags(tags) -> List[str]:
    '''Tags given as a list, or as a comma-separated string (as in a manifest).'''
    if tags is None:
        return []
    if isinstance(tags, str):
        tags = tags.split(',')
    return sorted({str(tag).strip() for tag in tags if str(tag).strip()})


def check_known(tags: Iterable[str], where: str) -> None:
    '''Raises an error if a tag is not in the vocabulary, suggesting the closest ones.'''
    import difflib
    for tag in tags:
        if tag not in TAG_FACET:
            close = difflib.get_close_matches(tag, list(TAG_FACET), n=3, cutoff=0.4)
            hint = f' Did you mean: {", ".join(close)}?' if close else ''
            raise RDDLRepoTagNotExistError(
                f'Tag <{tag}> of {where} is not in the vocabulary of rddlrepository.core.tags.{hint}')


def missing_facets(tags: Iterable[str]) -> List[str]:
    '''The facets that do not have exactly one tag among the given ones.'''
    counts = {facet: 0 for facet in FACETS}
    for tag in tags:
        if tag in TAG_FACET:
            counts[TAG_FACET[tag]] += 1
    return [facet for facet, n in counts.items() if n != 1]


def kind(ranges: List[str], prefix: str) -> str:
    '''discrete-, continuous- or mixed-(prefix) for fluents with the given ranges.'''
    real = ['real' == r for r in ranges]
    if real and all(real):
        return f'continuous-{prefix}'
    if any(real):
        return f'mixed-{prefix}'
    return f'discrete-{prefix}'


# ==========================================================================
# DERIVED TAGS, FROM THE TEXT OF THE DOMAIN
# ==========================================================================

_DECLARATION = re.compile(r'^\s*[^\s:(]+\s*(?:\([^)]*\))?\s*:\s*\{\s*([a-z\-]+)\s*,\s*([^,}\s]+)')
_RANDOM = re.compile(r'(?<![\w\-@$?])(?:' + '|'.join(DISTRIBUTIONS) + r')\s*(?:\(|\[|_\s*\{)')


def _block(text: str, keyword: str) -> str:
    '''The text between the braces that follow the first occurrence of keyword.'''
    m = re.search(r'(?<![\w\-])' + re.escape(keyword) + r'(?![\w\-])[^{]*\{', text)
    if m is None:
        raise ValueError(f'no {keyword} block')
    depth, start = 1, m.end()
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i]
    raise ValueError(f'the {keyword} block is not closed')


def scan_tags(rddl: str) -> List[str]:
    '''The derived tags of a domain, from the text of its RDDL file.'''
    text = re.sub(r'//[^\n]*', '', rddl)
    domain = _block(text, 'domain')
    pvariables = _block(domain, 'pvariables')
    fluents = {}
    for declaration in pvariables.split(';'):
        m = _DECLARATION.match(declaration)
        if m is not None:
            fluents.setdefault(m.group(1), []).append(m.group(2))
    rest = domain.replace(pvariables, ' ')
    return sorted([
        'partially-observable' if fluents.get('observ-fluent') else 'fully-observable',
        'stochastic' if _RANDOM.search(rest) else 'deterministic',
        kind(fluents.get('state-fluent', []), 'state'),
        kind(fluents.get('action-fluent', []), 'action'),
    ])
