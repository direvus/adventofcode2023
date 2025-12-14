"""matrix.py

Utility module for matrix operations
"""
from fractions import Fraction
from operator import add, sub


def num_leading_zeros(row: list) -> int:
    for i, n in enumerate(row):
        if n != 0:
            return i
    return i + 1


def get_non_echelon_row(matrix: list) -> int:
    """Return the index of the first non-echelon row of the matrix.

    The first non-echelon row is the first row encountered (starting from the
    top row) that does not have more leading zeroes than the previous row.

    Return None if all rows are in echelon form.
    """
    prev = -1
    for i, row in enumerate(matrix):
        z = num_leading_zeros(row)
        if z == len(row):
            return None
        if z <= prev:
            return i
        prev = z
    return None


def scale_iter(v, f):
    return tuple((x * f for x in v))


def row_echelon(matrix):
    """Perform reduction of a matrix to row echelon form.

    The matrix is modified in-place and returned.
    """
    while True:
        matrix.sort(key=num_leading_zeros)

        i = get_non_echelon_row(matrix)
        if i is None:
            break
        j = num_leading_zeros(matrix[i])
        upper = matrix[i - 1]
        for i in range(i, len(matrix)):
            lead = matrix[i][j]
            if lead == 0:
                break
            factor = Fraction(0 - lead, upper[j])
            scaled = scale_iter(upper, factor)
            matrix[i] = tuple(map(add, matrix[i], scaled))
    return matrix


def row_reduce(matrix):
    """Perform reduction of a matrix to reduced row echelon form.

    The matrix is modified in-place and returned.
    """
    # First, ensure that the matrix is in basic row echelon form
    row_echelon(matrix)

    # Starting from the second row, select the column containing the leading
    # cell, and ensure that all cells above it in the same column are zero, by
    # adding a multiple of the current row to it (type 3 transform).
    for i in range(1, len(matrix)):
        z = num_leading_zeros(matrix[i])
        if z >= len(matrix[i]) - 1:
            continue
        lead = matrix[i][z]
        for j in range(i):
            cell = matrix[j][z]
            if cell == 0:
                continue
            factor = Fraction(cell, lead)
            scaled = scale_iter(matrix[i], factor)
            matrix[j] = tuple(map(sub, matrix[j], scaled))
    return matrix


def find_free(matrix) -> set:
    """Find the free variables in a reduced-row matrix.

    Free variables are columns that do not have a pivot (leading non-zero
    value).

    Return a set of column indices for the free variables.
    """
    columns = set(range(len(matrix[0]) - 1))
    return columns - set(map(num_leading_zeros, matrix))


def solve_values(matrix: list, values: dict) -> dict:
    """Set values for variables in the matrix and solve the remainder.

    The original input matrix is not modified, but the values dict is.
    """
    width = len(matrix[0]) - 1
    for row in reversed(matrix):
        z = num_leading_zeros(row)
        if z >= width:
            continue
        aug = row[width]
        for j in range(z + 1, width):
            if j in values:
                aug -= row[j] * values[j]
        values[z] = Fraction(aug, row[z])
    return values


def solve_reduced(matrix: list) -> list:
    """Perform back-substitution to solve a matrix in reduced row echelon form.

    This function is only meant to be used on a matrix that is known to have a
    single valid solution.

    Return the list of values in the solution as Fractions.
    """
    width = len(matrix[0]) - 1
    solution = [None] * width
    for row in reversed(matrix):
        z = num_leading_zeros(row)
        if z >= width:
            continue
        aug = row[width]
        for j in range(z + 1, width):
            if solution[j] is not None:
                aug -= row[j] * solution[j]
        solution[z] = Fraction(aug, row[z])
    return solution


def solve_gaussian(matrix: list) -> list:
    """Perform a Gaussian elimination on an augmented matrix.

    Return the list of values in the solution as Fractions.
    """
    row_reduce(matrix)
    return solve_reduced(matrix)
