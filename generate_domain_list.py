'''Builds the list of domains in README.md, grouped by context: a small picture, the name, the instances
and a few words about each domain.

    python generate_domain_list.py                  # rewrite the list in README.md
    python generate_domain_list.py --thumbnails     # also render the pictures that are missing
    python generate_domain_list.py --thumbnails --all   # render all pictures again

The list is written between the DOMAINS:START and DOMAINS:END markers of README.md. Each picture is
the domain's visualizer on its first instance after a few random steps, shrunk to a small PNG of a few
kB in docs/thumbnails, fitted whole into a box of the same size for all. Rendering needs pyRDDLGym; rewriting the list does not.
'''
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from rddlrepository.core.manager import RDDLRepoManager

ROOT = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(ROOT, 'README.md')
THUMBS = 'docs/thumbnails'
START, END = '<!-- DOMAINS:START -->', '<!-- DOMAINS:END -->'

BOX = (200, 150)            # every picture is this size (pixels): the visualizer's picture keeps its
                            # shape inside it, and the rest is filled with its background
COLORS = 96                 # colors kept in a picture
STEPS = 10                  # random steps before the picture is taken
SHOWN = 64                  # width at which pictures are shown (pixels); smaller than BOX, so sharp
SPACER = 16                 # non-breaking spaces that keep the picture column about SHOWN wide
TIMEOUT = 300               # seconds to render one domain

# contexts in the order shown, with a line about each; other contexts follow in alphabetical order
CONTEXTS = {
    'standalone': 'Domains written for this repository.',
    'physics': 'Physical systems and control.',
    'or': 'Operations research.',
    'arcade': 'Arcade, puzzle and card games.',
    'gym': 'RDDL versions of the classic control environments of Gymnasium.',
    'rddlsim': 'Examples from rddlsim, the original Java simulator of RDDL.',
    'ippc2011': 'The International Probabilistic Planning Competition of 2011 '
                '([site](http://users.cecs.anu.edu.au/~ssanner/IPPC_2011/)).',
    'ippc2014': 'The International Probabilistic Planning Competition of 2014 '
                '([site](https://ssanner.github.io/IPPC_2014/)).',
    'ippc2018': 'The International Probabilistic Planning Competition of 2018 '
                '([site](https://ipc2018-probabilistic.bitbucket.io/)).',
    'ippc2023': 'The International Probabilistic Planning Competition of 2023 '
                '([site](https://ipc2023.github.io)).',
}


# words in a problem's name that mark a version of its domain, not the domain itself
VERSIONS = {'MDP', 'POMDP', 'Continuous', 'Discrete', 'Mixed', 'ComplexPhases', 'SimplePhases'}

# a few words about each domain, by problem name or by the name of the domain without its context and
# version (see label); a new domain without one gets its full description and a reminder
SHORT = {
    # standalone
    'BernoulliBandit': 'Bayesian Bernoulli bandit',
    'Elevators': 'Elevators carry passengers between floors',
    'Epidemic': 'Contain an epidemic with interventions',
    'Fishery': 'Harvest competing fish stocks',
    'HVAC': 'Heat the zones of a building',
    'Intruders': 'Detect intruders in an area',
    'Navigation_Continuous': 'Navigate around regions to avoid',
    'PowerGen': 'Generate power to meet demand',
    'Reservoir': 'Manage water in linked reservoirs',
    'TrafficBLX': 'Time traffic signals to cut travel time',
    # physics
    'Bicycle': 'Balance and ride a bicycle',
    'CartPoleChain': 'Swing up a chain of poles on a cart',
    'CartPoleSwingUp': 'Swing up a pole on a cart',
    'Highway': 'Drive through highway traffic',
    'Lander': 'Land a craft on a pad',
    'Pinball': 'Steer a ball around obstacles',
    'Quadcopter': 'Fly a swarm of quadcopters',
    'Reacher': 'Reach a target with a robot arm',
    'Reactor': 'Control a chemical reactor',
    'UAV': 'Fly drones to goal positions',
    # or
    'BinPacking': 'Pack random items into few bins',
    'CarRental': 'Move rental cars between locations',
    'Knapsack': 'Fill knapsacks with valuable items',
    'Option': 'Exercise an American option',
    'StructuralMaintenance': 'Inspect and repair a cracking structure',
    'SupplyChain': 'Ship stock from a factory to warehouses',
    'TSP': 'Travelling salesman tour',
    # arcade
    'Blackjack': 'Draw cards to 21 without going over',
    'BrickBreaker': 'Break bricks with a bouncing ball',
    'Eight': 'Sliding tile puzzle',
    'MazeChase': 'Gather berries in a maze full of wolves',
    'Minesweeper': 'Reveal safe cells, avoid hidden mines',
    'Pong': 'Single player pong',
    'Snake': 'Grow a snake by eating fruit',
    'Sokoban': 'Push boxes onto storage spots',
    'SpaceShooter': 'Shoot down waves of aliens',
    'Sudoku': 'Sudoku number puzzle',
    'Tetris': 'Classic block stacking game',
    'TowerOfHanoi': 'Move a tower of disks between rods',
    'Zombies': 'Avoid infection by zombies',
    # gym
    'Acrobot': 'Swing up a two-link arm',
    'CartPole': 'Balance a pole on a cart',
    'MountainCar': 'Drive a weak car up a hill',
    'Pendulum': 'Swing up and hold a pendulum',
    # rddlsim
    'ComplexSysAdmin': 'Keep an enhanced computer network running',
    'Logistics': 'Deliver packages by truck',
    'Pizza': 'Deliver pizzas',
    'PropDBN': 'Simple propositional network',
    'Sidewalk': 'People walking down a sidewalk',
    'Workforce': 'Run a call center',
    # competitions
    'AcademicAdvising': 'Choose courses for a student to graduate',
    'ChromaticDice': 'Yahtzee-like dice game',
    'CooperativeRecon': 'Rovers survey and photograph objects',
    'CrossingTraffic': 'Cross a grid of moving obstacles',
    'EarthObservation': 'A satellite photographs the Earth',
    'GameOfLife': "Control Conway's Game of Life",
    'Manufacturer': 'Run a manufacturing company',
    'MarsRover': 'Rovers harvest minerals',
    'Navigation': 'Reach a goal across risky cells',
    'PushYourLuck': 'Push-your-luck dice game',
    'RaceCar': 'Race a car around a track',
    'RecSim': 'Recommend content to users',
    'RedFinnedBlueEye': 'Remove an invasive fish species',
    'SkillTeaching': 'Teach skills with hints and quizzes',
    'SysAdmin': 'Keep a computer network running',
    'Tamarisk': 'Contain an invasive plant species',
    'Traffic': 'Control traffic lights on a road network',
    'Traffic_CTM': 'Control traffic lights on a road network',
    'TriangleTireworld': 'Drive to a goal, risking flat tires',
    'Wildfire': 'Fight spreading wildfires',
    'WildlifePreserve': 'Send rangers to stop poachers',
}


def thumbnail_path(name):
    return f'{THUMBS}/{name}.png'


# ===== the pictures =====

def render_one(name):
    '''Renders the picture of one domain (run in its own process).'''
    import warnings
    warnings.filterwarnings('ignore')
    import matplotlib
    matplotlib.use('Agg')
    from PIL import Image
    from pyRDDLGym.core.env import RDDLEnv
    from pyRDDLGym.core.policy import RandomAgent

    info = RDDLRepoManager().get_problem(name)
    viz_class = info.get_visualizer()
    if viz_class is None:
        return
    env = RDDLEnv(info.get_domain(), info.get_instance(info.list_instances()[0]))
    viz = viz_class(env.model)
    state, _ = env.reset()
    image = viz.render(env.state)
    try:
        agent = RandomAgent(env, seed=0)
        for _ in range(min(STEPS, env.horizon - 1)):
            _, _, done, truncated, _ = env.step(agent.sample_action(state))
            if done or truncated:
                break

            # every state is shown, for pictures that draw what happened before, such as a route
            image = viz.render(env.state)
    except Exception:
        pass                        # a picture of the first state is good enough
    image = image.convert('RGB')

    # crop the plain margin around the picture, keeping a little of it
    from PIL import ImageChops
    background = Image.new('RGB', image.size, image.getpixel((0, 0)))
    box = ImageChops.difference(image, background).convert('L').point(lambda v: 255 if v > 3 else 0).getbbox()
    if box:
        pad = max(4, (box[2] - box[0]) // 50)
        image = image.crop((max(box[0] - pad, 0), max(box[1] - pad, 0),
                            min(box[2] + pad, image.width), min(box[3] + pad, image.height)))

    # shrunk to fit the box, keeping its shape, the rest of the box filled with its background
    fill = image.getpixel((0, 0))
    image.thumbnail(BOX, Image.LANCZOS)
    boxed = Image.new('RGB', BOX, fill)
    boxed.paste(image, ((BOX[0] - image.width) // 2, (BOX[1] - image.height) // 2))
    image = boxed.quantize(colors=COLORS, method=Image.Quantize.MEDIANCUT)
    image.save(os.path.join(ROOT, thumbnail_path(name)), optimize=True)


def render_all(names, redo):
    os.makedirs(os.path.join(ROOT, THUMBS), exist_ok=True)
    todo = [n for n in names if redo or not os.path.exists(os.path.join(ROOT, thumbnail_path(n)))]

    def run(name):
        try:
            p = subprocess.run([sys.executable, __file__, '--one', name], cwd=ROOT, capture_output=True,
                               text=True, timeout=TIMEOUT)
            ok = p.returncode == 0
            return name, ok, '' if ok else p.stderr.strip().splitlines()[-1:]
        except subprocess.TimeoutExpired:
            return name, False, ['timed out']

    with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) // 2)) as pool:
        for name, ok, err in pool.map(run, todo):
            print(('rendered  ' if ok else 'FAILED    ') + name + ('' if ok else f'  {err}'))


# ===== the list =====

def instance_text(names):
    '''1, 2, ..., 10 as 1-10; anything else as a list.'''
    if all(n.isdigit() for n in names):
        nums = sorted(int(n) for n in names)
        if len(nums) > 2 and nums == list(range(nums[0], nums[-1] + 1)):
            return f'{nums[0]}&ndash;{nums[-1]}'
    return ', '.join(names)


def escape(text):
    return (text or '').replace('&', '&amp;').replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')


def entries(manager, context):
    '''The problems of a context, with numbered variants of one domain (A_V1_c, A_V2_c, ...) as one entry.'''
    out, variants = [], {}
    for name in sorted(manager.list_problems_by_context(context), key=str.lower):
        info = manager.get_problem(name)
        m = re.fullmatch(r'(.*)_V(\d+)_(' + re.escape(context) + ')', name)
        if m:
            key = (m.group(1), m.group(3))
            if key not in variants:
                variants[key] = dict(names=[], desc=info.desc, viz=info.viz, picture=name)
                out.append(variants[key])
            variants[key]['names'].append((int(m.group(2)), name))
        else:
            out.append(dict(names=[(None, name)], desc=info.desc, viz=info.viz, picture=name,
                            instances=instance_text(info.list_instances())))
    for e in variants.values():
        e['names'].sort()
        e['picture'] = e['names'][0][1]
        e['instances'] = 'the variant number'
    return out


def label(name, context):
    '''The short description of a problem: by its name, or by the name of its domain without the context
    and the version in it (MDP, POMDP, Continuous, ...); the full description if there is none.'''
    if name in SHORT:
        return SHORT[name]
    base = re.sub('_' + re.escape(context) + '$', '', name)
    parts = [p for p in base.split('_') if p not in VERSIONS and not re.fullmatch(r'V\d+', p)]
    return SHORT.get('_'.join(parts))


def identifier(name):
    '''A name in code, which may wrap after its underscores on a narrow screen.'''
    return '<code>' + name.replace('_', '_<wbr>') + '</code>'


def row(e, context):
    names = [n for _, n in e['names']]
    title = identifier(names[0]) if len(names) == 1 else f'{identifier(names[0])} to {identifier(names[-1])}'
    path = thumbnail_path(e['picture'])
    # only the width is given, so that the browser always keeps the shape of the picture
    picture = f'<img src="{path}" width="{SHOWN}" alt="">' if os.path.exists(os.path.join(ROOT, path)) else ''
    text = label(names[0], context)
    if text is None:
        print(f'no short description for {names[0]}: add one to SHORT')
        text = ' '.join((e['desc'] or '').split())
    return f'| {picture} | {title} | {e["instances"]} | {escape(text).replace("|", "&#124;")} |'


def domain_list(manager):
    contexts = manager.list_contexts()
    order = [c for c in CONTEXTS if c in contexts] + sorted(c for c in contexts if c not in CONTEXTS)
    total = sum(len(manager.list_problems_by_context(c)) for c in order)
    lines = [START, '',
             f'**{total} problems in {len(order)} contexts:** '
             + ' &middot; '.join(f'[{c}](#{c}) ({len(manager.list_problems_by_context(c))})' for c in order), '']
    # the header of the picture column is a row of non-breaking spaces about as wide as a picture: GitHub
    # cannot narrow the column below it, so on a small screen the pictures keep their size
    spacer = '&nbsp;' * SPACER
    for context in order:
        lines += [f'### {context}', '']
        if context in CONTEXTS:
            lines += [CONTEXTS[context], '']
        lines += [f'| {spacer} | Domain | Instances | Description |', '|:-:|:--|:--|:--|']
        lines += [row(e, context) for e in entries(manager, context)]
        lines += ['', '<sub>[back to contexts](#domain-gallery)</sub>', '']
    lines.append(END)
    return '\n'.join(lines)


def write_readme(manager):
    text = open(README, encoding='utf-8').read()
    if START not in text or END not in text:
        raise SystemExit(f'README.md needs the markers {START} and {END}')
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    with open(README, 'w', encoding='utf-8', newline='\n') as f:
        f.write(head + domain_list(manager) + tail)


def main():
    args = sys.argv[1:]
    if args[:1] == ['--one']:
        render_one(args[1])
        return
    manager = RDDLRepoManager(rebuild=True)
    if '--thumbnails' in args:
        names = [e['picture'] for c in manager.list_contexts() for e in entries(manager, c)]
        render_all(names, redo='--all' in args)
    write_readme(manager)
    print('README.md updated')


if __name__ == '__main__':
    main()
