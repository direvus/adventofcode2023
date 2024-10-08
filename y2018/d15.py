"""Advent of Code 2018

Day 15: Beverage Bandits

https://adventofcode.com/2018/day/15
"""
import logging  # noqa: F401
from collections import defaultdict
from copy import deepcopy

from util import get_manhattan_distance, timing, INF, PriorityQueue


def get_adjacent(position: tuple) -> set[tuple]:
    y, x = position
    return {(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)}


class Unit:
    def __init__(
            self, key: int, side: bool, position: tuple,
            power: int = 3, health: int = 200):
        self.key = key
        self.position = position
        self.side = side
        self.power = power
        self.health = health

    def __str__(self):
        side = 'Elf' if self.side else 'Goblin'
        return f'{side} #{self.key}'


class Game:
    def __init__(self, grid: str = ''):
        self.walls = set()
        self.occupied = set()
        self.units = defaultdict(dict)
        self.initial_units = {}
        if grid:
            self.parse(grid.split('\n'))

    def parse(self, stream):
        y = 0
        unitkey = 0
        for line in stream:
            line = line.strip()
            for x, ch in enumerate(line):
                pos = (y, x)
                match ch:
                    case '#':
                        self.walls.add(pos)
                    case 'E' | 'G':
                        side = ch == 'E'
                        unit = Unit(unitkey, side, pos)
                        self.units[side][unitkey] = unit
                        self.occupied.add(pos)
                        unitkey += 1
            y += 1
        self.initial_units = deepcopy(self.units)

    def reset(self):
        self.units = deepcopy(self.initial_units)
        self.occupied = (
                {x.position for x in self.elves} |
                {x.position for x in self.goblins})

    @property
    def elves(self):
        return self.units[True].values()

    @property
    def goblins(self):
        return self.units[False].values()

    @property
    def total_health(self) -> int:
        """Return the total health of all remaining units."""
        return (
                sum(x.health for x in self.elves) +
                sum(x.health for x in self.goblins))

    def to_string(self) -> str:
        lines = []
        maxy = max(x[0] for x in self.walls)
        maxx = max(x[1] for x in self.walls)
        elves = {x.position for x in self.elves}
        goblins = {x.position for x in self.goblins}
        for y in range(maxy + 1):
            line = []
            for x in range(maxx + 1):
                p = (y, x)
                if p in self.walls:
                    ch = '#'
                elif p in elves:
                    ch = 'E'
                elif p in goblins:
                    ch = 'G'
                else:
                    ch = ' '
                line.append(ch)
            lines.append(''.join(line))
        return '\n'.join(lines) + '\n'

    def find_targets(self, unit: Unit) -> list:
        targets = self.units[not unit.side].values()
        return list(targets)

    def get_neighbours(self, position: tuple) -> set:
        return get_adjacent(position) - (self.walls | self.occupied)

    def find_path(
            self, start: tuple, goal: tuple,
            limit: int | float = INF,
            ) -> int | None:
        """Find the shortest path from `start` to `goal`.

        Return the number of steps in the shortest path, or None if the goal is
        not reachable.

        If the path cost exceeds `limit`, give up and return None.
        """
        q = PriorityQueue()
        q.push(start, get_manhattan_distance(start, goal))
        dist = defaultdict(lambda: INF)
        dist[start] = 0
        explored = set()

        while q:
            cost, node = q.pop()
            if node == goal:
                return cost

            if cost > limit:
                return None

            for n in self.get_neighbours(node):
                score = dist[node] + 1
                if score < dist[n]:
                    dist[n] = score
                    f = score + get_manhattan_distance(n, goal)
                    q.set_priority(n, f)
            explored.add(node)
        return None

    def select_move(self, start: tuple, goals: set) -> tuple | None:
        """Select a target square for unit movement.

        If the unit is currently at `start`, and its possible destinations are
        listed in `goals`, select the best goal to move towards, or None if
        none of those goals can be reached.
        """
        if len(goals) == 0:
            return None
        if len(goals) == 1:
            dest = tuple(goals)[0]
            cost = self.find_path(start, dest)
            return dest if cost is not None else None

        best = INF
        result = None
        # Sort the goals by manhattan distance and reading order first, then
        # use A Star to determine the shortest path to each one. To save on
        # cycles, once we have found a shortest path, abandon any goals that
        # can't be resolved in fewer steps. In the worst case, the goal with
        # the shortest manhattan will be unreachable, then we will end up
        # exploring the entire space.
        goals = list(goals)
        goals.sort(key=lambda x: (get_manhattan_distance(start, x), x))
        for goal in goals:
            cost = self.find_path(start, goal, best)
            if cost is not None:
                if cost < best or (cost == best and goal < result):
                    result = goal
                    best = cost
        return result

    def select_step(self, start: tuple, goal: tuple) -> tuple:
        """Select the next step towards the selected goal.

        Considering all of the open squares adjacent to `start`, choose the one
        that has the shortest distance to the goal, breaking ties in favour of
        the earliest reading order.
        """
        best = INF
        choices = []
        for n in self.get_neighbours(start):
            cost = self.find_path(n, goal, best)
            if cost is not None and cost <= best:
                best = cost
                choices.append((cost, n))
        choices.sort()
        return choices[0][1]

    def do_move(self, unit: Unit, targets: list[Unit]) -> bool:
        """Perform this unit's movement.

        If one of the targets is already adjacent to the unit, then no movement
        is needed, return without moving.

        Otherwise, consider all of the open squares that are adjacent to each
        target. If there are none available, return without moving.

        Eliminate destination squares that are not reachable. If no valid
        destinations remain, return without moving.

        Finally, select the destination square that can be reached in the
        fewest steps, breaking ties in reading order. Move one step towards
        that square along the shortest path, again breaking ties by
        preferencing squares with a lower reading order.

        Return whether the unit moved.
        """
        adjacent = get_adjacent(unit.position)
        target_positions = {x.position for x in targets}
        if target_positions & adjacent:
            return False

        target_adjacencies = set()
        for pos in target_positions:
            target_adjacencies |= get_adjacent(pos)
        target_adjacencies -= (self.walls | self.occupied)
        if not target_adjacencies:
            logging.debug(f"Not moving {unit}, nowhere to go")
            return False

        goal = self.select_move(unit.position, target_adjacencies)
        if goal is None:
            logging.debug(f"Not moving {unit}, cannot reach")
            return False

        pos = self.select_step(unit.position, goal)
        logging.debug(f"Moving {unit} {unit.position} -> {pos}")
        self.occupied.discard(unit.position)
        self.occupied.add(pos)
        unit.position = pos
        return True

    def do_attack(self, unit: Unit, targets: list[Unit]) -> int:
        """Perform this unit's attack.

        Select the adjacent target that has the lowest health, breaking ties by
        reading order. Deal damage equal to the unit's attack power, removing
        the target if its health goes to zero or lower.

        Return the amount of damage the unit inflicted.
        """
        adjacent = get_adjacent(unit.position)
        targets = {x for x in targets if x.position in adjacent}
        if not targets:
            return 0

        if len(targets) == 1:
            target = next(iter(targets))
        else:
            choices = list(targets)
            choices.sort(key=lambda x: (x.health, x.position))
            target = choices[0]

        logging.debug(
                f"{unit} attacks {target} at {target.position}")
        target.health -= unit.power
        if target.health <= 0:
            logging.debug(f"{target} is defeated.")
            del self.units[target.side][target.key]
            self.occupied.discard(target.position)
        return unit.power

    def do_round(self) -> bool:
        """Do a round of combat.

        Return whether the game is still active and should continue after this
        round.
        """
        units = list(self.elves) + list(self.goblins)
        units.sort(key=lambda x: x.position)
        for unit in units:
            if unit.key not in self.units[unit.side]:
                # The unit must have been defeated in this round
                continue
            targets = self.find_targets(unit)
            if not targets:
                logging.debug(f"{unit} has nothing to fight, game over.")
                return False
            self.do_move(unit, targets)
            self.do_attack(unit, targets)
        return True

    def run(self, complete: bool = True) -> int:
        """Run the game.

        When `complete` is true, continue executing rounds of combat until one
        side is completely defeated. Otherwise, stop the game immediately any
        time an individual elf is defeated.

        Return the number of rounds that were completed.
        """
        rounds = 0
        active = True
        num_elves = len(self.initial_units[True])
        while active:
            logging.debug(f"--- BEGIN ROUND {rounds + 1} ---")
            active = self.do_round()
            if not complete and len(self.elves) < num_elves:
                logging.debug("Ending early because an elf has died")
                return rounds
            if active:
                rounds += 1
        return rounds

    def set_elf_power(self, power: int):
        for unit in self.elves:
            unit.power = power

    def find_attack_power(self) -> int:
        """Find the lowest attack power for a total elf victory.

        A total elf victory means that the elves win and none of them are
        killed.

        Return the number of rounds of the successful game.
        """
        power = 3
        win = False
        target = len(self.initial_units[True])
        while not win:
            power += 1
            self.reset()
            self.set_elf_power(power)
            rounds = self.run(False)
            result = len(self.elves)
            logging.info(
                    f"At {power} power, {result}/{target} elves "
                    f"survive after {rounds} rounds")
            win = result == target
        return rounds


def run(stream, test: bool = False):
    with timing("Part 1"):
        game = Game()
        game.parse(stream)
        rounds = game.run()
        result1 = rounds * game.total_health

    with timing("Part 2"):
        rounds = game.find_attack_power()
        result2 = rounds * game.total_health

    return (result1, result2)
