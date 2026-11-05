import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz

PROVIDER_COLORS = ('#1565c0', '#ef6c00', '#2e7d32', '#6a1b9a', '#00838f', '#ad1457', '#5d4037')
NO_PROVIDER_COLOR = '#b0bec5'
ZOOM_COLOR = '#90a4ae'
LEGEND_COLOR = '#37474f'

# the part of viridis that stands out against a white background
SATISFACTION_COLORMAP = ListedColormap(plt.cm.viridis(np.linspace(0, 0.85, 256)))

# items are drawn around their provider with this standard deviation in the domain, so
# the view extends this many standard deviations to include nearly all of them
ITEM_MARGIN = 3.0

# text smaller than this is not readable, so it is left out
MIN_FONT_SIZE = 5


class RecSimVisualizer(BaseViz):
    '''Draws the consumers, providers and current items in the space of features, a zoomed
    view of the region around the providers where items appear, and how the next items
    are shared between providers.

    The map shows the first two features, and any others are not shown. A provider that
    starts with no satisfaction can never have an item, so it is not drawn. The views
    are fixed, and the drawing only depends on the current state.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(10.4, 6.2),
                 wait_time=100) -> None:
        self._model = model
        self._wait_time = wait_time

        nonfluents = model.ground_vars_with_values(model.non_fluents)
        init_state = model.ground_vars_with_values(model.state_fluents)
        features = model.type_to_objects['feature']
        self._shown = features[:2]
        self._consumers = model.type_to_objects['consumer']
        self._items = model.type_to_objects['item']
        providers = model.type_to_objects['provider']

        # the share of a provider is proportional to its satisfaction, which only grows
        # from items that it has: starting from zero, it can never have a share
        self._providers = [p for p in providers if init_state[
            RDDLPlanningModel.ground_var('provider-satisfaction', [p])] > 0] or providers
        self._colors = {p: PROVIDER_COLORS[i % len(PROVIDER_COLORS)]
                        for (i, p) in enumerate(self._providers)}

        self._consumer_points = self._get_points(nonfluents, 'CONSUMER-AFFINITY', self._consumers)
        self._provider_points = self._get_points(nonfluents, 'PROVIDER-COMPETENCE', self._providers)
        self._norm = Normalize(0.0, nonfluents['MAX-AFFINITY'] or 1.0, clip=True)

        # the map and the color scale share the left, the zoomed view is at the top right
        # and the shares are below it
        fig_width, fig_height = figure_size
        left, gap, side_width, bottom, top = 0.65, 1.25, 3.3, 0.8, 0.55
        share_height, side_gap = 1.3, 0.55
        plot_size = fig_height - bottom - top
        side_left = left + plot_size + gap
        zoom_height = plot_size - share_height - side_gap
        self.fig = plt.figure(figsize=figure_size)
        self._map_ax = self.fig.add_axes(
            [left / fig_width, bottom / fig_height, plot_size / fig_width, plot_size / fig_height])
        bar_axis = self.fig.add_axes(
            [(left + plot_size + 0.15) / fig_width, bottom / fig_height,
             0.13 / fig_width, plot_size / fig_height])
        self._zoom_ax = self.fig.add_axes(
            [side_left / fig_width, (bottom + share_height + side_gap) / fig_height,
             side_width / fig_width, zoom_height / fig_height])
        self._share_ax = self.fig.add_axes(
            [side_left / fig_width, bottom / fig_height,
             side_width / fig_width, share_height / fig_height])
        self._setup_map(features)
        self._setup_zoom(init_state)
        self._setup_shares(share_height)
        scale = self.fig.colorbar(ScalarMappable(self._norm, SATISFACTION_COLORMAP), cax=bar_axis)
        scale.set_label('consumer satisfaction', fontsize=9)
        scale.ax.tick_params(labelsize=8, colors='#616161')
        scale.outline.set_edgecolor('#b0bec5')

        self.fig.legend(handles=[
            Line2D([], [], marker='o', color='none', markerfacecolor=SATISFACTION_COLORMAP(0.6),
                   markeredgecolor='white', markersize=9, label='Consumer'),
            Line2D([], [], marker='D', color='none', markerfacecolor=LEGEND_COLOR,
                   markeredgecolor='white', markersize=8, label='Provider'),
            Line2D([], [], marker='x', color='none', markeredgecolor=LEGEND_COLOR,
                   markeredgewidth=1.5, label='Item')
        ], loc='lower center', ncol=3, frameon=False, fontsize=8.5, columnspacing=1.0,
           handlelength=1.4)

    def _get_points(self, nonfluents, var, objects):
        points = np.array([[nonfluents[RDDLPlanningModel.ground_var(var, [o, f])]
                            for f in self._shown] for o in objects], dtype=float)
        return points.reshape((len(objects), len(self._shown)))

    def _xy(self, points):

        # with a single feature, everything lies along a line
        y = points[:, 1] if points.shape[1] > 1 else np.zeros(len(points))
        return points[:, 0], y

    def _setup_map(self, features):
        ax = self._map_ax
        ax.set_xlabel(features[0], fontsize=9)
        if len(features) > 1:
            ax.set_ylabel(features[1], fontsize=9)
        else:
            ax.set_yticks([])
        ax.tick_params(labelsize=8, colors='#616161')
        for spine in ax.spines.values():
            spine.set_color('#b0bec5')

        # the view is square, around the consumers and providers
        xs, ys = zip(*(self._xy(points) for points in (self._consumer_points, self._provider_points)))
        xs, ys = np.concatenate(xs), np.concatenate(ys)
        half = (max(np.ptp(xs), np.ptp(ys)) if len(xs) else 0.0) / 2 + ITEM_MARGIN
        x_center = (np.max(xs) + np.min(xs)) / 2 if len(xs) else 0.0
        y_center = (np.max(ys) + np.min(ys)) / 2 if len(ys) else 0.0
        ax.set_xlim(x_center - half, x_center + half)
        ax.set_ylim(y_center - half, y_center + half)
        ax.set_aspect('equal')

        if len(features) > 2:
            ax.text(0.01, 0.99, f'First 2 of {len(features)} features', transform=ax.transAxes,
                    ha='left', va='top', fontsize=8, color='#616161')

    def _setup_zoom(self, init_state):
        ax = self._zoom_ax
        ax.set_title('Zoom on providers and items', fontsize=10)
        ax.tick_params(labelsize=8, colors='#616161')
        if len(self._shown) < 2:
            ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(ZOOM_COLOR)
            spine.set_linestyle('--')

        # items appear around their provider, after starting from their initial positions
        starts, _ = self._get_items(init_state)
        xs = np.concatenate((self._xy(self._provider_points)[0], starts[0]))
        ys = np.concatenate((self._xy(self._provider_points)[1], starts[1]))
        half = max(np.ptp(xs), np.ptp(ys)) / 2 + ITEM_MARGIN
        x_center, y_center = (np.max(xs) + np.min(xs)) / 2, (np.max(ys) + np.min(ys)) / 2
        ax.set_xlim(x_center - half, x_center + half)
        ax.set_ylim(y_center - half, y_center + half)
        ax.set_aspect('equal')

        # show the region of the zoomed view on the main map
        self._map_ax.add_patch(patches.Rectangle(
            (x_center - half, y_center - half), 2 * half, 2 * half, facecolor='none',
            edgecolor=ZOOM_COLOR, linestyle='--', linewidth=1.2, zorder=1))

    def _setup_shares(self, plot_size):
        ax = self._share_ax
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(self._providers))
        ax.set_title('Share of next items', fontsize=10)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(['0%', '50%', '100%'], fontsize=8, color='#616161')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#b0bec5')
        ax.tick_params(axis='x', colors='#616161')

        # the first provider is on the top row
        self._rows = {p: len(self._providers) - 1 - i for (i, p) in enumerate(self._providers)}
        row_inches = plot_size / len(self._providers)
        font_size = min(9, 72 * 0.8 * row_inches)
        if font_size >= MIN_FONT_SIZE:
            ax.set_yticks([self._rows[p] + 0.5 for p in self._providers])
            ax.set_yticklabels(self._providers, fontsize=font_size)
        else:
            ax.set_yticks([])
        ax.tick_params(axis='y', length=0, colors='#616161')

    def _get_items(self, state):
        xs, ys, colors = [], [], []
        for i in self._items:
            location = [float(state[RDDLPlanningModel.ground_var('item-feature', [i, f])])
                        for f in self._shown]
            xs.append(location[0])
            ys.append(location[1] if len(location) > 1 else 0.0)
            owner = [p for p in self._providers if state[
                RDDLPlanningModel.ground_var('item-by', [p, i])]]
            colors.append(self._colors[owner[0]] if owner else NO_PROVIDER_COLOR)
        return (np.array(xs), np.array(ys)), colors

    def _draw_items(self, ax, positions, colors):

        # items change on every step, so they are on top with an outline to stand out
        items = ax.scatter(*positions, marker='x', s=50, c=colors, linewidths=2.0, zorder=5)
        items.set_path_effects([patheffects.withStroke(linewidth=3.5, foreground='white')])
        return [items]

    def _draw_providers(self, ax, shares):
        artists = []
        for (j, p) in enumerate(self._providers):
            x, y = self._xy(self._provider_points[[j]])
            artists.extend(ax.plot(
                x, y, marker='D', markersize=7 + 10 * np.sqrt(shares[p]), color=self._colors[p],
                markeredgecolor='white', linestyle='none', zorder=3))
        return artists

    def _draw_shares(self, shares):
        return [self._share_ax.add_patch(patches.Rectangle(
            (0, self._rows[p] + 0.15), shares[p], 0.7, facecolor=self._colors[p],
            edgecolor='none')) for p in self._providers]

    def _draw_consumers(self, ax, satisfaction):
        xs, ys = self._xy(self._consumer_points)
        return [ax.scatter(
            xs, ys, s=55, c=satisfaction, cmap=SATISFACTION_COLORMAP, norm=self._norm,
            edgecolors='white', linewidths=0.8, alpha=0.75, zorder=4)]

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def render(self, state):
        satisfaction = [float(state[RDDLPlanningModel.ground_var('consumer-satisfaction', [c])])
                        for c in self._consumers]
        provider_values = {p: float(state[RDDLPlanningModel.ground_var('provider-satisfaction', [p])])
                           for p in self._providers}

        # same rule as the domain: the share of next items follows provider satisfaction
        total = sum(provider_values.values())
        shares = {p: (value / total if total > 0 else 0.0) for (p, value) in provider_values.items()}

        positions, colors = self._get_items(state)
        artists = self._draw_shares(shares)
        for ax in (self._map_ax, self._zoom_ax):
            artists.extend(self._draw_items(ax, positions, colors))
            artists.extend(self._draw_providers(ax, shares))
            artists.extend(self._draw_consumers(ax, satisfaction))

        self.fig.suptitle(f'Total consumer satisfaction: {sum(satisfaction):.1f}',
                          fontsize=13, fontweight='bold')

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
