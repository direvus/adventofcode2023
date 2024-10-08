"""Advent of Code 2017

Day 21: Fractal Art

https://adventofcode.com/2017/day/21
"""
import logging  # noqa: F401
from collections import defaultdict
from itertools import product

from util import timing


INITIAL_PATTERN = '.#./..#/###'


class Grid:
    size: int
    rows: list

    def __init__(self, source):
        """Initialise a square grid.

        The grid can be specified either with a string pattern (sequences of .
        and #, each row delimited by a /), or another grid to copy, or a list
        of lists of 1s and 0s, or a size integer to create a grid with all
        cells off.
        """
        if isinstance(source, str):
            rows = source.split('/')
            self.size = len(rows)
            self.rows = [[int(c == '#') for c in row] for row in rows]
        elif isinstance(source, Grid):
            self.size = source.size
            self.rows = [list(row) for row in source.rows]
        elif isinstance(source, int) and source > 0:
            self.size = source
            self.rows = [list([0] * source) for _ in range(source)]
        else:
            self.size = len(source)
            self.rows = [list(row) for row in source]

    @property
    def count(self):
        """Return the total number of 'on' cells in the grid."""
        return sum(sum(row) for row in self.rows)

    @property
    def string(self):
        """Return a compact string representation of the grid."""
        return ''.join(''.join(str(c) for c in row) for row in self.rows)

    def print(self):
        for row in self.rows:
            print(''.join('#' if c else '.' for c in row))

    def rotate(self, count: int = 1):
        """Return a new grid, rotated clockwise `count` steps."""
        count %= 4
        if count == 0:
            rows = self
        elif count == 1:
            rows = [
                [self.rows[y][x] for y in reversed(range(self.size))]
                for x in range(self.size)]
        elif count == 2:
            rows = [reversed(row) for row in self.rows]
            rows.reverse()
        elif count == 3:
            rows = [
                [self.rows[y][x] for y in range(self.size)]
                for x in reversed(range(self.size))]
        return Grid(rows)

    def flip(self):
        """Return a new grid, flipped horizontally."""
        rows = [reversed(row) for row in self.rows]
        return Grid(rows)

    def get_subgrid_strings(self):
        step = 2 if self.size % 2 == 0 else 3
        result = []
        for y in range(0, self.size, step):
            line = []
            for x in range(0, self.size, step):
                line.append(''.join(
                        ''.join(str(c) for c in row[x:x + step])
                        for row in self.rows[y:y + step]))
            result.append(line)
        return result


class Transform:
    def __init__(self, source, dest):
        self.source = Grid(source)
        self.dest = Grid(dest)


class TransformSet:
    def __init__(self):
        self.transforms = defaultdict(lambda: defaultdict(list))
        self.subgrids = {}

    def add(self, item: Transform):
        self.transforms[item.source.size][item.source.count].append(item)

    def prepare_subgrids(self, grid: Grid):
        """Build a mapping of this grid through its matching transform.

        This function generates every possible rearrangement of the grid by
        rotation or flipping, and if it finds a matching transformation rule,
        maps all of those rearrangement strings to the destination of that rule
        in self.subgrids.
        """
        size = grid.size
        count = grid.count
        txs = self.transforms[size][count]
        if not txs:
            return

        # Prepare all flips and rotations for this grid as strings
        s = grid.string
        strings = {s}
        strings |= {grid.rotate(i).string for i in range(1, 4)}
        flip = grid.flip()
        strings.add(flip.string)
        strings |= {flip.rotate(i).string for i in range(1, 4)}

        for tx in txs:
            src = tx.source.string
            if src in strings:
                for s in strings:
                    self.subgrids[s] = tx.dest
                return

    def prepare_all_subgrids(self) -> None:
        """Build a mapping of every size 2 and 3 grid through their transform.

        This function generates every possible grid of size 2 and 3, and maps
        it to the destination of its matching transform, if we have one.

        The result is stored in `self.subgrids` as a dict mapping the string
        form of the source subgrid to its transform output Grid.
        """
        for r in product(range(2), repeat=4):
            g = Grid([list(r[:2]), list(r[2:])])
            if g.string not in self.subgrids:
                self.prepare_subgrids(g)

        for r in product(range(2), repeat=9):
            g = Grid([list(r[j:j + 3]) for j in range(0, 9, 3)])
            if g.string not in self.subgrids:
                self.prepare_subgrids(g)

    def transform(self, grid: Grid | str) -> Grid:
        if isinstance(grid, Grid):
            grid = grid.string
        if grid not in self.subgrids:
            raise ValueError(f"Grid {grid} not found in cached transforms")
        return Grid(self.subgrids[grid])

    def enhance(self, grid: Grid, count: int = 1) -> Grid:
        """Perform `count` rounds of enhancement on the grid.

        For each round of enhancement, split the grid up into subgrids of size
        2 or 3, apply the matching transform rule to each subgrid, and create a
        new grid by joining the transformed subgrids back together.

        Return the final Grid result.
        """
        for i in range(count):
            strings = grid.get_subgrid_strings()
            rows = []
            j = 0
            for line in strings:
                for s in line:
                    sub = self.transform(s)
                    for k in range(len(sub.rows)):
                        if len(rows) <= j + k:
                            rows.append(sub.rows[k])
                        else:
                            rows[j + k].extend(sub.rows[k])
                j += k + 1
            grid = Grid(rows)
        return grid


def parse(stream) -> TransformSet:
    result = TransformSet()
    for line in stream:
        line = line.strip()
        parts = line.split(' => ')
        tx = Transform(*parts)
        result.add(tx)
    return result


def run(stream, test: bool = False):
    start = Grid(INITIAL_PATTERN)

    with timing("Part 1"):
        transforms = parse(stream)
        transforms.prepare_all_subgrids()

        count = 2 if test else 5
        grid = transforms.enhance(start, count)
        result1 = grid.count

    with timing("Part 2"):
        # Further transformations not possible with the test input
        result2 = 0
        if not test:
            # Part 2 requires 18 iterations, but we've already done 5
            count = 18 - 5
            grid = transforms.enhance(grid, count)
            result2 = grid.count

    return (result1, result2)
