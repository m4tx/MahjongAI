from collections import Counter


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


def tiles_in_the_same_family(*tiles):
    return len(set([x // 9 for x in tiles if x < 27])) == 1


hand = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7]
print(can_call_riichi(hand))

hand = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 5]
print(can_call_riichi(hand))

hand = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 32, 5]
print(can_call_riichi(hand))

hand = [0, 0, 2, 2, 3, 4, 4, 4, 5, 6, 7, 6, 7, 8]
print(can_call_riichi(hand))

hand = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 4]
print(can_call_riichi(hand))
