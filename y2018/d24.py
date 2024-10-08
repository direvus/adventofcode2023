"""Advent of Code 2018

Day 24: Immune System Simulator 20XX

https://adventofcode.com/2018/day/24
"""
import logging  # noqa: F401
from copy import deepcopy

from util import timing


class Group:
    def __init__(self):
        self.name = ''
        self.army = ''
        self.count = 0
        self.hp = 0
        self.weaknesses = set()
        self.immunities = set()
        self.type = ''
        self.damage = 0
        self.initiative = 0

    @property
    def groupid(self):
        return f'{self.army} {self.name}'

    @property
    def power(self) -> int:
        return self.count * self.damage

    def calc_damage(self, defender: 'Group') -> int:
        if self.type in defender.immunities:
            return 0
        elif self.type in defender.weaknesses:
            return self.power * 2
        return self.power

    def select_target(self, enemies: list['Group']):
        targets = filter(lambda x: self.type not in x.immunities, enemies)
        targets = list(targets)
        if not targets:
            return None
        targets.sort(key=lambda x: (
                self.calc_damage(x), x.power, x.initiative), reverse=True)
        return targets[0]

    def take_damage(self, attacker: 'Group') -> int:
        damage = attacker.calc_damage(self)
        kills = min(self.count, damage // self.hp)
        self.count -= kills
        return kills


def parse_group(army: str, name: str, line: str) -> Group:
    line = line.strip()
    group = Group()
    group.army = army
    group.name = name
    if ' (' in line:
        prefix, rem = line.split(' (')
        brackets, suffix = rem.split(')')
        words = (prefix + suffix).split()
    else:
        words = line.split()
        brackets = ''
    group.count = int(words[0])
    group.hp = int(words[4])
    group.damage = int(words[12])
    group.type = words[13]
    group.initiative = int(words[17])

    # damage modifiers
    if not brackets:
        return group

    modifiers = brackets.split('; ')
    for mod in modifiers:
        effect, _, rem = mod.split(maxsplit=2)
        types = set(rem.split(', '))
        if effect == 'weak':
            group.weaknesses = types
        elif effect == 'immune':
            group.immunities = types
    return group


class Army:
    def __init__(self, name: str):
        self.name = name
        self.groups = []

    def clean(self):
        self.groups = list(filter(lambda x: x.count > 0, self.groups))

    def get_total_units(self):
        return sum(x.count for x in self.groups)

    def boost(self, amount: int):
        for g in self.groups:
            g.damage += amount


def parse_army(name: str, stream) -> Army:
    army = Army(name)
    n = 1
    for line in stream:
        line = line.strip()
        if line == '':
            break
        group = parse_group(name, n, line)
        army.groups.append(group)
        n += 1
    return army


def parse(stream) -> list[Army]:
    result = []
    line = stream.readline().strip()
    while line:
        name = line.rstrip(':')
        result.append(parse_army(name, stream))
        line = stream.readline().strip()
    return result


def do_round(armies: list[Army]) -> bool:
    # Set up groups
    targets = {}
    targeted = set()
    groups = {}
    for army in armies:
        for group in army.groups:
            groups[group.groupid] = group

    # Select targets
    targetors = list(groups.values())
    targetors.sort(key=lambda x: (x.power, x.initiative), reverse=True)
    for group in targetors:
        choices = filter(
                lambda x: x.army != group.army and x.groupid not in targeted,
                targetors)
        target = group.select_target(choices)
        if target:
            targets[group.groupid] = target.groupid
            targeted.add(target.groupid)

    if not targets:
        # Stalemate
        return False

    # Attack
    attackers = list(targets.keys())
    attackers.sort(key=lambda x: groups[x].initiative, reverse=True)
    total_kills = 0
    for attackerid in attackers:
        attacker = groups[attackerid]
        if attacker.count <= 0:
            continue
        targetid = targets[attackerid]
        target = groups[targetid]
        kills = target.take_damage(attacker)
        total_kills += kills

    if total_kills == 0:
        # Stalemate
        return False

    for army in armies:
        army.clean()
        if len(army.groups) == 0:
            return False
    return True


def fight(armies: list[Army]) -> tuple:
    active = True
    while active:
        active = do_round(armies)

    survivors = [x for x in armies if len(x.groups) > 0]
    if len(survivors) > 1:
        return False, sum(x.get_total_units() for x in survivors)

    winner = survivors[0]
    return (winner.name == 'Immune System', winner.get_total_units())


def find_min_boost(initial_armies: list[Army], boost: int) -> int:
    win = False
    while not win:
        armies = deepcopy(initial_armies)
        for army in armies:
            if army.name == 'Immune System':
                army.boost(boost)
        win, units = fight(armies)
        boost += 1
    return units


def run(stream, test: bool = False):
    with timing("Part 1"):
        armies = parse(stream)
        initial = deepcopy(armies)
        win, result1 = fight(armies)

    with timing("Part 2"):
        initial_boost = 1570 if test else 58
        result2 = find_min_boost(initial, initial_boost)

    return (result1, result2)
