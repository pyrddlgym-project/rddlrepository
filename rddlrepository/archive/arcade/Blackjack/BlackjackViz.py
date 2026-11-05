import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.visualizer.viz import BaseViz


class BlackjackVisualizer(BaseViz):
    '''Draws the player and dealer hands on a card table.

    The state only holds the hand totals, so the cards are worked out from the
    change in each total between renders. This requires render to be called on
    every step, otherwise the cards drawn in between merge into one. All cards
    worth 10 points (10, J, Q, K) have the same effect on the state, so they
    are all drawn as 10.
    '''

    def __init__(self, model: RDDLPlanningModel,
                 figure_size=(6, 5.4),
                 card_size=(0.12, 0.26),
                 wait_time=100) -> None:
        self._model = model
        self._figure_size = figure_size
        self._card_size = card_size
        self._wait_time = wait_time

        self._hands = {'player': [], 'dealer': []}
        self._hard = {'player': 0, 'dealer': 0}
        self._newest = None
        self._last_state = None

        self.fig = plt.figure(figsize=self._figure_size)
        self.ax = self.fig.add_axes([0.02, 0.02, 0.96, 0.89])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.ax.set_facecolor('#c8e6c9')
        for spine in self.ax.spines.values():
            spine.set_color('#b0bec5')

    def convert2img(self, fig):
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        data = data[:, :, :3]
        return Image.fromarray(data)

    def _update_hands(self, state):

        # rendering the same state again must not change the hands
        stage = str(state['current-stage'])
        last_state = (stage,
                      int(state['player-value']), bool(state['player-soft']),
                      int(state['dealer-value']), bool(state['dealer-soft']))
        if last_state == self._last_state:
            return
        self._last_state = last_state

        self._newest = None
        for who in self._hands:

            # total with aces as 1 rises by exactly the value of the card drawn:
            # the hand value alone does not, e.g. soft 15 + K is still 15
            hard = int(state[f'{who}-value']) - 10 * bool(state[f'{who}-soft'])

            # a new episode has started
            if stage == 'p1' or hard < self._hard[who]:
                self._hands[who] = []
                self._hard[who] = 0

            drawn = hard - self._hard[who]
            self._hard[who] = hard
            if drawn > 0:
                label = 'A' if drawn == 1 else (str(drawn) if drawn <= 10 else '?')
                self._hands[who].append(label)
                self._newest = (who, len(self._hands[who]) - 1)

    def _status(self, state):
        stage = str(state['current-stage'])
        if stage == 'done':
            player, dealer = int(state['player-value']), int(state['dealer-value'])
            if player > 21:
                return 'Player busts', '#c62828'
            elif dealer > 21:
                return 'Dealer busts, player wins', '#2e7d32'
            elif player > dealer:
                return 'Player wins', '#2e7d32'
            elif player == dealer:
                return 'Push', 'black'
            else:
                return 'Dealer wins', '#c62828'
        elif stage == 'pn':
            return "Player's turn", 'black'
        elif stage in ('d2', 'dn'):
            return "Dealer's turn", 'black'
        else:
            return 'Dealing', 'black'

    def _draw_card(self, x, y, zorder, label=None, highlight=False):
        w, h = self._card_size
        artists = [self.ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0,rounding_size=0.015', zorder=zorder,
            facecolor='white' if label else '#1565c0',
            edgecolor='#ff8f00' if highlight else '#37474f',
            linewidth=3.0 if highlight else 1.0))]
        if label:

            # the rank in the corner stays visible when cards overlap
            artists.append(self.ax.text(
                x + 0.012, y + h - 0.012, label, ha='left', va='top',
                fontsize=11, fontweight='bold', zorder=zorder))
            artists.append(self.ax.text(
                x + w / 2, y + h / 2 - 0.015, label, ha='center', va='center',
                fontsize=20, fontweight='bold', zorder=zorder))
        return artists

    def _draw_hand(self, who, y, face_down, state):
        total = int(state[f'{who}-value'])
        soft = ' (soft)' if state[f'{who}-soft'] else ''
        title = f'{who.capitalize()}: {total}{soft}' if self._hands[who] else who.capitalize()
        artists = [self.ax.text(
            0.04, y + self._card_size[1] + 0.025, title,
            fontsize=10, fontweight='bold', va='bottom')]

        # cards are spaced evenly, overlapping when they would not fit on the table
        hand = self._hands[who] + [None] * face_down
        step = self._card_size[0] + 0.02
        if len(hand) > 1:
            step = min(step, (0.92 - self._card_size[0]) / (len(hand) - 1))
        for (i, label) in enumerate(hand):
            artists.extend(self._draw_card(
                0.04 + i * step, y, i, label, self._newest == (who, i)))
        return artists

    def render(self, state):
        self._update_hands(state)

        # the dealer's second card stays face down until the player stands
        face_down = (str(state['current-stage']) in ('pn', 'd2')
                     and len(self._hands['dealer']) == 1)

        # draw the hands and the status of the game
        status, color = self._status(state)
        artists = self._draw_hand('dealer', 0.56, int(face_down), state)
        artists.extend(self._draw_hand('player', 0.08, 0, state))
        self.ax.set_title(status, fontsize=13, fontweight='bold', color=color)

        self.fig.canvas.draw()
        img = self.convert2img(self.fig)
        for artist in artists:
            artist.remove()
        del artists

        return img
