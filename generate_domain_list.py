'''Builds the list of domains in README.md, grouped by context, with a small picture of each domain.

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
COLUMNS = 4                 # domains in a row of the list
SHOWN = 150                 # width at which pictures are shown (pixels); smaller than BOX, so sharp.
                            # COLUMNS of them must fit GitHub's README column with room to spare, or
                            # GitHub narrows the columns unevenly and some pictures shrink
LINE = 18                   # longest line of a name (characters), no wider than a picture in GitHub's
                            # code font, so that every column of every context is a picture wide
BLANK = 'docs/thumbnails/_blank.png'    # an empty box for a domain without a visualizer
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
    shown = env.state
    try:
        agent = RandomAgent(env, seed=0)
        for _ in range(min(STEPS, env.horizon - 1)):
            _, _, done, truncated, _ = env.step(agent.sample_action(state))
            if done or truncated:
                break
            shown = env.state
    except Exception:
        pass                        # a picture of the first state is good enough
    image = viz.render(shown).convert('RGB')

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


def blank():
    '''A transparent picture of the size of a box, so that a row stays aligned without a picture.'''
    path = os.path.join(ROOT, BLANK)
    if not os.path.exists(path):
        from PIL import Image
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.new('LA', BOX, (0, 0)).save(path, optimize=True)
    return BLANK


def wrap(name):
    '''A name in lines of at most LINE characters, broken after underscores (or in words).'''
    parts = [p + '_' for p in name.split('_')]
    parts[-1] = parts[-1][:-1]
    # a part that is still too long is broken before its capital letters
    parts = [q for p in parts for q in (re.findall(r'[A-Z]?[^A-Z]*', p)[:-1] if len(p) > LINE else [p])]
    lines, line = [], ''
    for part in parts:
        if line and len(line + part) > LINE:
            lines.append(line)
            line = part
        else:
            line += part
    lines.append(line)
    return '<br>'.join(f'<code>{line}</code>' for line in lines)


def filler():
    '''An empty cell as wide as the others, for the end of a short row.'''
    return f'<td><img src="{blank()}" width="{SHOWN}" alt=""></td>'


def cell(e):
    names = [wrap(n) for _, n in e['names']]
    if len(names) == 1:
        title = names[0]
    else:
        title = f'{names[0]}<br>to<br>{names[-1]}'
    path = thumbnail_path(e['picture'])
    desc = escape(' '.join((e['desc'] or '').split()))
    if not os.path.exists(os.path.join(ROOT, path)):
        path, desc = blank(), ''     # a domain without a visualizer has no picture
    # only the width is given, so that the browser always keeps the shape of the picture
    picture = f'<img src="{path}" width="{SHOWN}" alt="{desc}" title="{desc}">'
    return (f'<td align="center" valign="top">{picture}<br>{title}<br>'
            f'<sub>instances {e["instances"]}</sub></td>')


def domain_list(manager):
    contexts = manager.list_contexts()
    order = [c for c in CONTEXTS if c in contexts] + sorted(c for c in contexts if c not in CONTEXTS)
    total = sum(len(manager.list_problems_by_context(c)) for c in order)
    lines = [START, '',
             f'**{total} problems in {len(order)} contexts:** '
             + ' &middot; '.join(f'[{c}](#{c}) ({len(manager.list_problems_by_context(c))})' for c in order), '']
    for context in order:
        lines += [f'### {context}', '']
        if context in CONTEXTS:
            lines += [CONTEXTS[context], '']
        es = entries(manager, context)
        lines.append('<table>')
        for i in range(0, len(es), COLUMNS):
            row = [cell(e) for e in es[i:i + COLUMNS]]
            row += [filler()] * (COLUMNS - len(row))     # every table has COLUMNS columns
            lines.append('<tr>' + ''.join(row) + '</tr>')
        lines += ['</table>', '', '<sub>[back to contexts](#domain-gallery)</sub>', '']
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
