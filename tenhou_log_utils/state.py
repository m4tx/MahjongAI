import copy
from collections import Counter
from typing import Iterable, Union, List

import numpy as np

import logging

_LG = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN)


TILE_EAST = 27
TILE_SOUTH = 28
TILE_WEST = 29
TILE_NORTH = 30
WIND_TILES = [TILE_EAST, TILE_SOUTH, TILE_WEST, TILE_NORTH]


class GameState:
    def __init__(self, hands, red_fives, dora_indicators, scores, kyoku, dealer):
        self.hands = hands
        self.red_fives = red_fives
        self.discards = [[], [], [], []]
        self.stolen_tiles = [[], [], [], []]
        self.dora_indicators = dora_indicators
        self.riichi = [False] * 4
        self.scores = scores
        self.kyoku = kyoku  # 0 - 7 (round number)
        self.dealer = dealer

        self.last_discard = None
        self.last_action = None
        self.finished = False
        self.gains = None

    def get_player_state(self, player, last_state):
        discards = [self.discards[(player + i) % 4] for i in range(4)]
        stolen_tiles = [self.stolen_tiles[(player + i) % 4] for i in range(4)]
        riichi = [self.riichi[(player + i) % 4] for i in range(4)]

        return PlayerState(
            hand=self.hands[player],
            red_fives=self.red_fives[player],
            discards=discards,
            stolen_tiles=stolen_tiles,
            dora_indicators=self.dora_indicators,
            riichi=riichi,
            rank=self._get_player_rank(player),
            kyoku=self.kyoku,
            round_wind=self._get_round_wind(),
            own_wind=(player - self.dealer + 4) % 4,
            last_player_state=last_state,
        )

    def make_draw(self, player, tile) -> 'GameState':
        new_gs = copy.deepcopy(self)
        new_gs.hands[player].append(tile)
        new_gs.last_action = None
        return new_gs

    def make_discard(self, player, tile) -> 'GameState':
        new_gs = copy.deepcopy(self)
        new_gs.hands[player].remove(tile)
        new_gs.hands[player].sort()
        new_gs.discards[player].append(tile)
        new_gs.last_discard = tile
        new_gs.last_action = 'discard', player, tile
        return new_gs

    def make_riichi(self, player) -> 'GameState':
        new_gs = copy.deepcopy(self)
        new_gs.riichi[player] = True
        new_gs.last_action = 'riichi', player
        return new_gs

    def make_dora(self, tile) -> 'GameState':
        new_gs = copy.deepcopy(self)
        new_gs.dora_indicators.append(tile)
        new_gs.last_action = None
        return new_gs

    def make_call(self, player, tiles: List[int]) -> 'GameState':
        new_gs = copy.deepcopy(self)

        new_gs.stolen_tiles[player] += tiles
        to_remove = tiles.copy()
        to_remove.remove(self.last_discard)
        for tile in to_remove:
            new_gs.hands[player].remove(tile)

        new_gs.last_action = 'call', player, tiles
        return new_gs

    def make_kan(self, player, tiles: List[int]) -> 'GameState':
        tile = tiles[0]
        new_gs = copy.deepcopy(self)
        while new_gs.stolen_tiles[player].count(tile) < 4:
            new_gs.stolen_tiles[player].append(tile)
        while tile in new_gs.hands[player]:
            new_gs.hands[player].remove(tile)

        new_gs.last_action = None
        return new_gs

    def make_finish(self, scores, gains) -> 'GameState':
        new_gs = copy.deepcopy(self)
        new_gs.scores = scores
        new_gs.finished = True
        new_gs.gains = gains

        new_gs.last_action = None
        return new_gs

    def can_call_pon(self, player, discarded_tile):
        return self.hands[player].count(discarded_tile) >= 2

    def can_call_chii(self, player, discarded_tile):
        hand = self.hands[player]
        rv = []

        if discarded_tile + 1 in hand and discarded_tile + 2 in hand and tiles_in_the_same_family(discarded_tile, discarded_tile + 1, discarded_tile + 2):
            rv.append(1)

        if discarded_tile - 1 in hand and discarded_tile + 1 in hand and tiles_in_the_same_family(discarded_tile - 1, discarded_tile, discarded_tile + 1):
            rv.append(2)

        if discarded_tile - 2 in hand and discarded_tile - 1 in hand and tiles_in_the_same_family(discarded_tile - 2, discarded_tile - 1, discarded_tile):
            rv.append(3)

        return rv

    def can_call_riichi(self, player):
        hand = self.hands[player]
        return can_call_riichi(hand)

    def _get_round_wind(self):
        return 0 if self.kyoku < 4 else 1

    def _get_player_rank(self, player):
        scores = list(enumerate(self.scores))
        scores.sort(key=lambda x: x[1])
        for i in range(4):
            if scores[i][0] == player:
                return i
        assert False

    def __str__(self):
        s = []

        for i in range(4):
            s.append(f'Player {i} hand: {convert_hand_2(self.hands[i])}')
        for i in range(4):
            s.append(f'Player {i} red fives: {convert_hand_2(self.red_fives[i])}')
        for i in range(4):
            s.append(f'Player {i} discards: {convert_hand_2(self.discards[i])}')
        for i in range(4):
            s.append(f'Player {i} stolen tiles: {convert_hand_2(self.stolen_tiles[i])}')
        s.append(f'Dora indicators: {convert_hand_2(self.dora_indicators)}')
        s.append(f'Riichis: {self.riichi}')
        s.append(f'Scores: {self.scores}')
        s.append(f'Kyoku: {self.kyoku}')
        s.append(f'Dealer: {self.dealer}')

        return '\n'.join(s)




class PlayerState:
    def __init__(self, hand, red_fives, discards, stolen_tiles, dora_indicators, riichi, rank, kyoku, round_wind, own_wind, last_player_state):
        self.hand = hand
        self.red_fives = red_fives
        self.discards = discards
        self.stolen_tiles = stolen_tiles
        self.dora_indicators = dora_indicators
        self.riichi = riichi
        self.rank = rank
        self.kyoku = kyoku
        self.round_wind = round_wind
        self.own_wind = own_wind
        self.last_player_state = last_player_state

        self.data = None
        self.next_plane = 0

    def make_data(self):
        if self.data is not None:
            return self.data

        self.data = np.zeros((43, 34, 4), dtype=np.int8)

        self.fill_hand()
        self.fill_red_fives()
        self.fill_discards()
        self.fill_stolen_tiles()
        self.fill_dora_indicators()
        self.fill_riichi()
        self.fill_rank()
        self.fill_kyoku()
        self.fill_round_wind()
        self.fill_own_wind()

        if self.last_player_state is not None:
            self.fill_last_hand()
            self.fill_last_discards()
            self.fill_last_stolen_tiles()
            self.fill_last_dora_indicators()
            self.fill_last_riichi()

        return self.data

    def fill_hand(self):
        self.fill_plane_tiles(self.data[self.next_plane], self.hand)
        self.next_plane += 1

    def fill_last_hand(self):
        self.fill_plane_tiles(self.data[self.next_plane], self.last_player_state.hand)
        self.next_plane += 1

    def fill_red_fives(self):
        self.fill_plane_tiles_rows(self.data[self.next_plane], self.red_fives)
        self.next_plane += 1

    def fill_discards(self):
        arr = self.data[self.next_plane:self.next_plane + 4]
        for i in range(4):
            self.fill_plane_tiles(arr[i], self.discards[i])
        self.next_plane += 4

    def fill_last_discards(self):
        arr = self.data[self.next_plane:self.next_plane + 4]
        for i in range(4):
            self.fill_plane_tiles(arr[i], self.last_player_state.discards[i])
        self.next_plane += 4

    def fill_stolen_tiles(self):
        arr = self.data[self.next_plane:self.next_plane + 4]
        for i in range(4):
            self.fill_plane_tiles(arr[i], self.stolen_tiles[i])
        self.next_plane += 4

    def fill_last_stolen_tiles(self):
        arr = self.data[self.next_plane:self.next_plane + 4]
        for i in range(4):
            self.fill_plane_tiles(arr[i], self.last_player_state.stolen_tiles[i])
        self.next_plane += 4

    def fill_dora_indicators(self):
        self.fill_plane_tiles(self.data[self.next_plane], self.dora_indicators)
        self.next_plane += 1

    def fill_last_dora_indicators(self):
        self.fill_plane_tiles(self.data[self.next_plane], self.last_player_state.dora_indicators)
        self.next_plane += 1

    def fill_riichi(self):
        arr = self.data[self.next_plane:self.next_plane + 4]
        for i in range(4):
            self.fill_plane_value(arr[i], self.riichi[i])
        self.next_plane += 4

    def fill_last_riichi(self):
        arr = self.data[self.next_plane:self.next_plane + 4]
        for i in range(4):
            self.fill_plane_value(arr[i], self.last_player_state.riichi[i])
        self.next_plane += 4

    def fill_rank(self):
        arr = self.data[self.next_plane:self.next_plane + 4]
        self.fill_plane_value(arr[self.rank], 1)
        self.next_plane += 4

    def fill_kyoku(self):
        kyoku = min(self.kyoku, 7)
        arr = self.data[self.next_plane:self.next_plane + 8]
        self.fill_plane_value(arr[kyoku], 1)
        self.next_plane += 8

    def fill_round_wind(self):
        self.fill_plane_tiles_rows(self.data[self.next_plane], [WIND_TILES[self.round_wind]])
        self.next_plane += 1

    def fill_own_wind(self):
        self.fill_plane_tiles_rows(self.data[self.next_plane], [WIND_TILES[self.own_wind]])
        self.next_plane += 1

    def fill_plane_tiles(self, arr, tiles: Iterable[int]):
        for tile in tiles:
            for i in range(4):
                if arr[tile][i] == 0:
                    arr[tile][i] = 1
                    break

    def fill_plane_tiles_rows(self, arr, tile_set: Iterable[int]):
        for tile in tile_set:
            for i in range(4):
                arr[tile][i] = 1

    def fill_plane_value(self, arr, value: Union[bool, int]):
        arr.fill(value)

    def __str__(self):
        s = []

        s.append(f'Player hand: {convert_hand_2(self.hand)}')
        s.append(f'Player red fives: {convert_hand_2(self.red_fives)}')
        for i in range(4):
            s.append(f'Player discards: {convert_hand_2(self.discards[i])}')
        for i in range(4):
            s.append(f'Player stolen tiles: {convert_hand_2(self.stolen_tiles[i])}')
        s.append(f'Dora indicators: {convert_hand_2(self.dora_indicators)}')
        s.append(f'Riichis: {self.riichi}')
        s.append(f'Rank: {self.rank}')
        s.append(f'Kyoku: {self.kyoku}')
        s.append(f'Round wind: {_tile2text(WIND_TILES[self.round_wind])}')
        s.append(f'Own wind: {_tile2text(WIND_TILES[self.own_wind])}')

        return '\n'.join(s)


def tiles_in_the_same_family(*tiles):
    return len(set([x // 9 for x in tiles if x < 27])) == 1


def can_call_riichi(hand):
    if len(hand) != 14:
        return False

    for removed_tile in hand:
        for added_tile in range(34):
            new_hand = hand.copy()
            new_hand.remove(removed_tile)
            new_hand.append(added_tile)
            new_hand.sort()
            if _riichi_test(new_hand, False):
                return True

    return False


def _riichi_test(hand, had_pair):
    if len(hand) == 0:
        return True
    if len(hand) == 1:
        return False

    if list(Counter(hand).values()) == [2] * 7:
        return True

    if hand == [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33] or len(set(hand).intersection({0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33})) == 12:
        return True

    if hand[0] == hand[1] and not had_pair:
        if _riichi_test(hand[2:], True):
            return True

    if len(hand) == 2:
        return False

    if hand[0] == hand[1] == hand[2]:
        if _riichi_test(hand[3:], had_pair):
            return True

    if hand[0] + 1 in hand and hand[0] + 2 in hand and tiles_in_the_same_family(hand[0], hand[0] + 1, hand[0] + 2):
        new_hand = hand.copy()
        new_hand.remove(hand[0])
        new_hand.remove(hand[0] + 1)
        new_hand.remove(hand[0] + 2)
        new_hand.sort()
        if _riichi_test(new_hand, had_pair):
            return True

    return False


def _tile2text(tile):
    tile_unicodes = [
        # M
        u'\U0001f007',
        u'\U0001f008',
        u'\U0001f009',
        u'\U0001f00a',
        u'\U0001f00b',
        u'\U0001f00c',
        u'\U0001f00d',
        u'\U0001f00e',
        u'\U0001f00f',
        # P
        u'\U0001f019',
        u'\U0001f01a',
        u'\U0001f01b',
        u'\U0001f01c',
        u'\U0001f01d',
        u'\U0001f01e',
        u'\U0001f01f',
        u'\U0001f020',
        u'\U0001f021',
        # S
        u'\U0001f010',
        u'\U0001f011',
        u'\U0001f012',
        u'\U0001f013',
        u'\U0001f014',
        u'\U0001f015',
        u'\U0001f016',
        u'\U0001f017',
        u'\U0001f018',
        # Z
        u'\U0001f000',
        u'\U0001f001',
        u'\U0001f002',
        u'\U0001f003',
        u'\U0001f006',
        u'\U0001f005',
        u'\U0001f004',
    ]
    return tile_unicodes[tile]



def convert_hand_2(tiles):
    """Convert hands (int) into unicode characters for print."""
    return u' '.join([_tile2text(tile) for tile in tiles])



def _tile2unicode(tile):
    tile_unicodes = [
        # M
        u'\U0001f007',
        u'\U0001f008',
        u'\U0001f009',
        u'\U0001f00a',
        u'\U0001f00b',
        u'\U0001f00c',
        u'\U0001f00d',
        u'\U0001f00e',
        u'\U0001f00f',
        # P
        u'\U0001f019',
        u'\U0001f01a',
        u'\U0001f01b',
        u'\U0001f01c',
        u'\U0001f01d',
        u'\U0001f01e',
        u'\U0001f01f',
        u'\U0001f020',
        u'\U0001f021',
        # S
        u'\U0001f010',
        u'\U0001f011',
        u'\U0001f012',
        u'\U0001f013',
        u'\U0001f014',
        u'\U0001f015',
        u'\U0001f016',
        u'\U0001f017',
        u'\U0001f018',
        # Z
        u'\U0001f000',
        u'\U0001f001',
        u'\U0001f002',
        u'\U0001f003',
        u'\U0001f006',
        u'\U0001f005',
        u'\U0001f004',
    ]
    return u'{} {}'.format(tile_unicodes[tile//4], tile % 4)


def convert_hand(tiles):
    """Convert hands (int) into unicode characters for print."""
    return u' '.join([_tile2unicode(tile) for tile in tiles])


################################################################################
def _print_shuffle(data):
    _LG.info('Shuffle:')
    _LG.info('  Seed: %s', data['seed'])
    _LG.info('  Ref: %s', data['ref'])


################################################################################
def _print_go(data):
    _LG.info('Lobby%s:', '' if data['lobby'] < 0 else ' %s' % data['lobby'])
    _LG.info('  Table: %s', data['table'])
    for key, value in data['config'].items():
        _LG.info('    %s: %s', key, value)


################################################################################
def _print_resume(data):
    index, name = data['index'], data['name']
    _LG.info('Player %s (%s) has returned to the game.', index, name)


def _print_un(data):
    _LG.info('Players:')
    _LG.info('  %5s: %3s, %8s, %3s, %s', 'Index', 'Dan', 'Rate', 'Sex', 'Name')
    for i, datum in enumerate(data):
        dan, rate = datum['dan'], datum['rate']
        name, sex = datum['name'], datum['sex']
        _LG.info('  %5s: %3s, %8.2f, %3s, %s', i, dan, rate, sex, name)


################################################################################
def _print_taikyoku(data):
    _LG.info('Dealer: %s', data['oya'])


################################################################################
def _print_scores(scores):
    for i, score in enumerate(scores):
        _LG.info('  %6s: %6s', i, score)


def _print_init(data):
    # GameState()

    # dora = convert_hand([data['dora']])
    # field_ = data['round'] // 4
    # repeat = field_ // 4
    # round_ = data['round'] % 4 + 1
    # field = ['Ton', 'Nan', 'Xia', 'Pei'][field_ % 4]
    # _LG.info('Initial Game State:')
    # if repeat:
    #     _LG.info('  Round: %s %s %s Kyoku', repeat, field, round_)
    # else:
    #     _LG.info('  Round: %s %s Kyoku', field, round_)
    # _LG.info('  Combo: %s', data['combo'])
    # _LG.info('  Reach: %s', data['reach'])
    # _LG.info('  Dice 1: %s', data['dices'][0])
    # _LG.info('  Dice 2: %s', data['dices'][1])
    # _LG.info('  Dora Indicator: %s', dora)
    # _LG.info('  Initial Scores:')
    # _print_scores(data['scores'])
    # _LG.info('  Dealer: %s', data['oya'])
    # _LG.info('  Initial Hands:')
    # for i, hand in enumerate(data['hands']):
    #     _LG.info('  %5s: %s', i, convert_hand(sorted(hand)))

    hands = [[i // 4 for i in sorted(hand)] for hand in data['hands']]
    dora_indicators = [data['dora'] // 4]
    gs = GameState(hands, [[]] * 4, dora_indicators, data['scores'], data['round'], int(data['oya']))
    return gs


################################################################################
def _print_draw(data, gs):
    tile = _tile2unicode(data['tile'])
    _LG.info('Player %s: Draw    %s', data['player'], tile)
    return gs.make_draw(data['player'], data['tile'] // 4)


################################################################################
def _print_discard(data, gs):
    tile = _tile2unicode(data['tile'])
    _LG.info('Player %s: Discard %s', data['player'], tile)
    return gs.make_discard(data['player'], data['tile'] // 4)


################################################################################
def _print_call(caller, callee, call_type, mentsu, gs):
    tiles = u''.join([_tile2unicode(tile) for tile in mentsu])
    if call_type == 'KaKan' or caller == callee:
        from_ = u''
    else:
        from_ = u' from player {}'.format(callee)
    _LG.info(u'Player %s: %s%s: %s', caller, call_type, from_, tiles)

    tiles_36 = [tile // 4 for tile in mentsu]
    if call_type == 'KaKan' or caller == callee:
        return gs.make_kan(caller, tiles_36)
    else:
        return gs.make_call(caller, tiles_36)

################################################################################
def _print_reach(data, gs):
    if data['step'] == 1:
        _LG.info(u'Player %s: Reach', data['player'])
        return gs.make_riichi(data['player'])
    elif data['step'] == 2:
        return gs
        # _LG.info(u'Player %s made deposite.', data['player'])
        # if 'scores' in data:
        #     _LG.info(u'New scores:')
        #     _print_scores(data['scores'])
    else:
        raise NotImplementedError('Unexpected step value: {}'.format(data))


################################################################################
def _print_ba(ba):
    _LG.info('  Ten-bou:')
    _LG.info('    Combo: %s', ba['combo'])
    _LG.info('    Reach: %s', ba['reach'])


def _print_result(result):
    _LG.info('  Result:')
    for score, uma in zip(result['scores'], result['uma']):
        _LG.info('    %6s: %6s', score, uma)


def _print_agari(data, gs):
    limit = [
        'No limit',
        'Mangan',
        'Haneman',
        'Baiman',
        'Sanbaiman',
        'Yakuman',
    ]

    yaku_name = [
        # 1 han
        'Tsumo',
        'Reach',
        'Ippatsu',
        'Chankan',
        'Rinshan-kaihou',
        'Hai-tei-rao-yue',
        'Hou-tei-rao-yui',
        'Pin-fu',
        'Tan-yao-chu',
        'Ii-pei-ko',
        # Ji-kaze
        'Ton',
        'Nan',
        'Xia',
        'Pei',
        # Ba-kaze
        'Ton',
        'Nan',
        'Xia',
        'Pei',
        'Haku',
        'Hatsu',
        'Chun',
        # 2 han
        'Double reach',
        'Chii-toi-tsu',
        'Chanta',
        'Ikki-tsuukan',
        'San-shoku-dou-jun',
        'San-shoku-dou-kou',
        'San-kan-tsu',
        'Toi-Toi-hou',
        'San-ankou',
        'Shou-sangen',
        'Hon-rou-tou',
        # 3 han
        'Ryan-pei-kou',
        'Junchan',
        'Hon-itsu',
        # 6 han
        'Chin-itsu',
        # mangan
        'Ren-hou',
        # yakuman
        'Ten-hou',
        'Chi-hou',
        'Dai-sangen',
        'Suu-ankou',
        'Suu-ankou Tanki',
        'Tsu-iisou',
        'Ryu-iisou',
        'Chin-routo',
        'Chuuren-poutou',
        'Jyunsei Chuuren-poutou 9',
        'Kokushi-musou',
        'Kokushi-musou 13',
        'Dai-suushi',
        'Shou-suushi',
        'Su-kantsu',
        # kensyou
        'Dora',
        'Ura-dora',
        'Aka-dora',
    ]
    _LG.info('Player %s wins.', data['winner'])
    if 'loser' in data:
        _LG.info('  Ron from player %s', data['loser'])
    else:
        _LG.info('  Tsumo.')
    _LG.info('  Hand: %s', convert_hand(sorted(data['hand'])))
    _LG.info('  Machi: %s', convert_hand(data['machi']))
    _LG.info('  Dora Indicator: %s', convert_hand(data['dora']))
    if data['ura_dora']:
        _LG.info('  Ura Dora: %s', convert_hand(data['ura_dora']))
    _LG.info('  Yaku:')
    for yaku, han in data['yaku']:
        _LG.info('      %-20s (%2d): %2d [Han]', yaku_name[yaku], yaku, han)
    if data['yakuman']:
        for yaku in data['yakuman']:
            _LG.info('      %s (%s)', yaku_name[yaku], yaku)
    _LG.info('  Fu: %s', data['ten']['fu'])
    _LG.info('  Score: %s', data['ten']['point'])
    if data['ten']['limit']:
        _LG.info('    - %s', limit[data['ten']['limit']])
    _print_ba(data['ba'])
    _LG.info('  Scores:')
    for cur, gain in zip(data['scores'], data['gains']):
        _LG.info('    %6s: %6s', cur, gain)

    if 'result' in data:
        _print_result(data['result'])

    scores = [x + y for x, y in zip(data['scores'], data['gains'])]
    return gs.make_finish(scores, data['gains'])




###############################################################################
def _print_dora(data, gs):
    _LG.info('New Dora Indicator: %s', convert_hand([data['hai']]))
    return gs.make_dora(data['hai'] // 4)


###############################################################################
def _print_ryuukyoku(data, gs):
    reason = {
        'nm': 'Nagashi Mangan',
        'yao9': '9-Shu 9-Hai',
        'kaze4': '4 Fu',
        'reach4': '4 Reach',
        'ron3': '3 Ron',
        'kan4': '4 Kan',
    }

    _LG.info('Ryukyoku:')
    if 'reason' in data:
        _LG.info('  Reason: %s', reason[data['reason']])
    for i, hand in enumerate(data['hands']):
        if hand is not None:
            _LG.info('Player %s: %s', i, convert_hand(sorted(hand)))
    _LG.info('  Scores:')
    for cur, gain in zip(data['scores'], data['gains']):
        _LG.info('    %6s: %6s', cur, gain)
    _print_ba(data['ba'])
    if 'result' in data:
        _print_result(data['result'])

    scores = [x + y for x, y in zip(data['scores'], data['gains'])]
    return gs.make_finish(scores, data['gains'])


################################################################################
def _print_bye(data):
    _LG.info('Player %s has left the game.', data['index'])


################################################################################
def print_node(tag, data, gs):
    """Print XML node of tenhou mjlog parsed with `parse_node` function.

    Parameters
    ----------
    tag : str
        Tags such as 'GO', 'DORA', 'AGARI' etc...

    data: dict
        Parsed info of the node
    """
    _LG.debug('%s: %s', tag, data)
    if tag == 'GO':
        pass
    elif tag == 'UN':
        pass
    if tag == 'TAIKYOKU':
        pass
    elif tag == 'SHUFFLE':
        pass
    elif tag == 'INIT':
        return _print_init(data), None
    elif tag == 'DORA':
        return _print_dora(data, gs), None
    elif tag == 'DRAW':
        return _print_draw(data, gs), 'draw'
    elif tag == 'DISCARD':
        return _print_discard(data, gs), 'discard'
    elif tag == 'CALL':
        return _print_call(**data, gs=gs), 'call'
    elif tag == 'REACH':
        return _print_reach(data, gs), 'riichi'
    elif tag == 'AGARI':
        return _print_agari(data, gs), 'finish'
    elif tag == 'RYUUKYOKU':
        return _print_ryuukyoku(data, gs), 'finish'
    elif tag == 'BYE':
        pass
    elif tag == 'RESUME':
        pass
    else:
        raise NotImplementedError('{}: {}'.format(tag, data))


def process_round(round,
                  x_data_discard, y_data_discard,
                  x_data_pon, y_data_pon,
                  x_data_chii, y_data_chii,
                  x_data_riichi, y_data_riichi):
    gs = None
    gs_list = []
    for node in round:
        result = print_node(node['tag'], node['data'], gs)
        if result is not None:
            new_gs, action = result

            assert new_gs is not None
            gs = new_gs

            if action is None:
                continue

            # if action == 'call':
            #     print(len(gs_list))

            gs_list.append(gs)

            # if action is not None:
                # print('=' * 30)
                # player_state = gs.get_player_state(0)
                # print(player_state)
                # print(player_state.make_data())

            if action == 'finish':
                gains = gs_list[-1].gains
                # formidable_opponents = [i for i, gain in enumerate(gains) if gain > -1500]
                formidable_opponents = [0, 1, 2, 3]
                # print(formidable_opponents)

                process_discards(gs_list, formidable_opponents, x_data_discard, y_data_discard)
                process_pons(gs_list, formidable_opponents, x_data_pon, y_data_pon)
                process_chiis(gs_list, formidable_opponents, x_data_chii, y_data_chii)
                process_riichis(gs_list, formidable_opponents, x_data_riichi, y_data_riichi)

                gs_list.clear()


def process_discards(gs_list, formidable_opponents, x_data, y_data):
    for player in formidable_opponents:
        for index, game_state in enumerate(gs_list):
            if index < 2:
                continue

            if (game_state.last_action is not None and
                    game_state.last_action[0] == 'discard' and
                    game_state.last_action[1] == player and
                    not game_state.riichi[player]):
                last_state_1 = gs_list[index - 1]
                last_player_state_2 = gs_list[index - 2].get_player_state(player, None)
                data = last_state_1.get_player_state(player, last_player_state_2).make_data()
                x_data.append(data)
                y_data.append(game_state.last_action[2])


def process_pons(gs_list: List[GameState], formidable_opponents, x_data, y_data):
    for player in formidable_opponents:
        for index, game_state in enumerate(gs_list[:-1]):
            if (game_state.last_action is not None and
                    game_state.last_action[0] == 'discard' and
                    game_state.last_action[1] != player and
                    not game_state.riichi[player] and
                    game_state.can_call_pon(player, game_state.last_action[2])):
                next_state = gs_list[index + 1]
                data = next_state.get_player_state(player, game_state.get_player_state(player, None)).make_data()
                x_data.append(data)
                called_pon = (next_state.last_action is not None and
                              next_state.last_action[0] == 'call' and
                              next_state.last_action[1] == player)
                y_data.append(1 if called_pon else 0)


def process_chiis(gs_list: List[GameState], formidable_opponents, x_data, y_data):
    for index, game_state in enumerate(gs_list[:-1]):
        for player in formidable_opponents:
            if (game_state.last_action is not None and
                    game_state.last_action[0] == 'discard' and
                    game_state.last_action[1] == (player + 3) % 4 and
                    not game_state.riichi[player] and
                    game_state.can_call_chii(player, game_state.last_action[2])):
                next_state = gs_list[index + 1]
                data = next_state.get_player_state(player, game_state.get_player_state(player, None)).make_data()
                x_data.append(data)
                called_chii = (next_state.last_action is not None and
                               next_state.last_action[0] == 'call' and
                               next_state.last_action[1] == player and
                               len(set(next_state.last_action[2])) > 1)

                if called_chii:
                    called_tiles = sorted(next_state.last_action[2])
                    discarded_tile = game_state.last_action[2]
                    if called_tiles == [discarded_tile, discarded_tile + 1, discarded_tile + 2]:
                        y_data.append(1)
                    elif called_tiles == [discarded_tile - 1, discarded_tile, discarded_tile + 1]:
                        y_data.append(2)
                    elif called_tiles == [discarded_tile - 2, discarded_tile - 1, discarded_tile]:
                        y_data.append(3)
                else:
                    y_data.append(0)


def process_riichis(gs_list: List[GameState], formidable_opponents, x_data, y_data):
    for player in formidable_opponents:
        for index, game_state in enumerate(gs_list[:-1]):
            if index < 1:
                continue

            if (game_state.last_action is not None and
                    game_state.last_action[1] == player and
                    game_state.can_call_riichi(player)):
                last_player_state_1 = gs_list[index - 1].get_player_state(player, None)
                data = game_state.get_player_state(player, last_player_state_1).make_data()

                x_data.append(data)
                called_richii = game_state.last_action[0] == 'riichi'
                y_data.append(1 if called_richii else 0)
