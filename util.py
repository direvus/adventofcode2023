import heapq
import logging
import time
from collections import namedtuple
from contextlib import contextmanager
from enum import Enum, auto
from functools import total_ordering


@contextmanager
def timing(message: str = None) -> int:
    start = time.perf_counter_ns()
    if message:
        logging.info(f"[........] ==> {message}")
    try:
        yield start
    finally:
        end = time.perf_counter_ns()
        dur = end - start
        logging.info(f"[{dur//1000:8d}] <== {message}")


@total_ordering
class Direction(Enum):
    NORTH = auto()
    WEST = auto()
    SOUTH = auto()
    EAST = auto()

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __str__(self):
        if self == Direction.NORTH:
            return 'N'
        if self == Direction.EAST:
            return 'E'
        if self == Direction.WEST:
            return 'W'
        return 'S'


Point = namedtuple('point', ['y', 'x'])


VECTORS = {
        Direction.NORTH: (-1,  0),
        Direction.SOUTH:  (1,  0),
        Direction.EAST:   (0,  1),
        Direction.WEST:   (0, -1),
        }


def move(
        point: Point,
        direction: Direction,
        count: int = 1,
        ) -> Point:
    v = tuple(x * count for x in VECTORS[direction])
    return Point(point[0] + v[0], point[1] + v[1])


def minmax(a, b):
    if a < b:
        return a, b
    return b, a


class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.finder = {}
        self.deleted = set()

    def __len__(self):
        return len(self.queue)

    def __bool__(self):
        return bool(self.finder)

    def push(self, node, priority):
        entry = (priority, node)
        heapq.heappush(self.queue, entry)
        self.finder[node] = entry

    def has_node(self, node):
        return node in self.finder

    def has_position(self, position):
        return position in {(n.y, n.x) for n in self.finder.keys()}

    def set_priority(self, node, priority):
        if node in self.finder:
            self.deleted.add(id(self.finder[node]))
        self.push(node, priority)

    def pop(self):
        while self.queue:
            entry = heapq.heappop(self.queue)
            if id(entry) not in self.deleted:
                del self.finder[entry[1]]
                return entry
            self.deleted.discard(id(entry))
        raise KeyError('Cannot pop from empty priority queue')
