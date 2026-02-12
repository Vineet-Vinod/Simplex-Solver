"""
Test suite designed to expose bugs in the Simplex solver.

Known bugs targeted:
1. Division by zero in min-ratio test when pivot column has zero entries (line 33-36)
2. Unbounded check uses strict < instead of <= 0, missing zero-column cases (line 29)
3. Min-ratio test doesn't skip rows with negative pivot column entries, producing wrong ratios
4. Negative RHS / negative pivot column entry gives a positive ratio, fooling the min-ratio test
5. Already optimal (all objective coeffs <= 0) — should be a no-op but worth verifying
6. Single constraint LP — the for-loop on line 34 is range(2,2) so only row 1 is considered
"""

import pytest
from simplex import Simplex


def _make(rows):
    """Helper: convert list-of-lists of numbers into list-of-lists of strings (as the solver expects)."""
    return [[str(v) for v in row] for row in rows]


# ── 1. Basic sanity: the provided example from problem.txt ──────────────────

def test_basic_example():
    """
    Maximize 5x1 + 6x2  s.t.  3x1+4x2<=108, 5x1+4x2<=140, x1,x2>=0
    Known optimal: x1=12, x2=18 => obj=168... let's just verify solve completes.
    Actually: pivoting gives obj=170 (x1=20, x2=10+).
    We trust the LP formulation; just check solve doesn't crash.
    """
    tableau = _make([
        [5, 6, 0, 0, 0],
        [3, 4, 1, 0, 108],
        [5, 4, 0, 1, 140],
    ])
    s = Simplex(tableau)
    s.solve()
    assert pytest.approx(-s.mat[0][-1], abs=1e-6) == 170.0


# ── 2. Already optimal — no pivoting needed ─────────────────────────────────

def test_already_optimal():
    """All objective coefficients are <= 0 so solve() should do nothing."""
    tableau = _make([
        [-1, -2, 0, 0, -10],
        [1,  1,  1, 0, 5],
        [2,  1,  0, 1, 8],
    ])
    s = Simplex(tableau)
    s.solve()
    assert pytest.approx(-s.mat[0][-1], abs=1e-6) == 10.0


# ── 3. Single constraint LP ─────────────────────────────────────────────────

def test_single_constraint():
    """
    Maximize 3x1 + 2x2  s.t.  x1+x2<=10
    Optimal at x1=10, x2=0 => obj=30
    With only one constraint row the inner for-loop (range(2, 2)) never runs.
    """
    tableau = _make([
        [3, 2, 0, 0],
        [1, 1, 1, 10],
    ])
    s = Simplex(tableau)
    s.solve()
    assert pytest.approx(-s.mat[0][-1], abs=1e-6) == 30.0


# ── 4. Zero in pivot column → division by zero ─────────────────────────────

def test_zero_in_pivot_column():
    """
    BUG: When a constraint row has 0 in the entering variable's column,
    the min-ratio test (line 33/35) divides by zero → ZeroDivisionError.

    Maximize 2x1 + x2  s.t.  x1<=4, x2<=6, x1,x2>=0
    Row 2 has 0 in the x1 column, so pivoting on x1 triggers the bug.
    Expected optimal: x1=4, x2=6 => obj=14
    """
    tableau = _make([
        [2, 1, 0, 0, 0],
        [1, 0, 1, 0, 4],   # x1 column = 1
        [0, 1, 0, 1, 6],   # x1 column = 0  ← triggers div-by-zero
    ])
    s = Simplex(tableau)
    s.solve()  # should not raise ZeroDivisionError
    assert pytest.approx(-s.mat[0][-1], abs=1e-6) == 14.0


# ── 5. Negative entry in pivot column → wrong min-ratio ─────────────────────

def test_negative_entry_in_pivot_column():
    """
    BUG: The min-ratio test considers rows with negative pivot-column entries.
    A negative entry produces a negative ratio which is always < any valid
    positive ratio, causing the solver to pick the wrong pivot row.

    Maximize x1 + x2
    s.t.   x1       <= 4
          -x1 + x2  <= 3
                 x2  <= 6
    Optimal: x1=4, x2=6 => obj=10 (but not x2=7 which violates x2<=6)

    When entering x1, row 2 has coefficient -1, giving ratio 3/(-1) = -3,
    which the solver incorrectly picks as minimum.
    """
    tableau = _make([
        [1,  1, 0, 0, 0, 0],
        [1,  0, 1, 0, 0, 4],
        [-1, 1, 0, 1, 0, 3],
        [0,  1, 0, 0, 1, 6],
    ])
    s = Simplex(tableau)
    s.solve()
    assert pytest.approx(-s.mat[0][-1], abs=1e-6) == 10.0


# ── 6. Unbounded LP with zero (not negative) column ─────────────────────────

def test_unbounded_zero_column():
    """
    BUG: The unbounded check (line 29) uses `all(row[argma] < 0 ...)`.
    A column of all zeros (<=0 but not <0) is also unbounded but passes
    the check, leading to division by zero instead of reporting unbounded.

    Maximize x1 + x2  s.t.  x2 <= 5
    x1 is free to go to infinity → unbounded.
    Pivot column for x1 has all zeros in constraint rows.
    """
    tableau = _make([
        [1, 1, 0, 0],
        [0, 1, 1, 5],
    ])
    s = Simplex(tableau)
    # Should detect unbounded; currently divides by zero
    try:
        s.solve()
    except ZeroDivisionError:
        pytest.fail("Solver raised ZeroDivisionError instead of detecting unbounded LP")


# ── 7. Clearly unbounded LP (negative column) ───────────────────────────────

def test_unbounded_negative_column():
    """
    Maximize x1  s.t.  -x1 <= 5
    x1 can grow without bound. The solver should detect this.
    """
    tableau = _make([
        [1,  0, 0],
        [-1, 1, 5],
    ])
    s = Simplex(tableau)
    s.solve()
    # After solve, objective should still show the problem is unbounded.
    # The solver prints "Unbounded LP" and breaks — the objective row
    # will still have a positive coefficient.
    assert any(v > 0 for v in s.mat[0][:-1]), "Should have stopped with positive obj coefficients (unbounded)"


# ── 8. Three constraints, multiple pivots ────────────────────────────────────

def test_three_constraints():
    """
    Maximize 5x1 + 4x2 + 3x3
    s.t.  6x1 + 4x2 + 2x3 <= 240
          3x1 + 2x2 + 5x3 <= 270
          5x1 + 6x2 + 5x3 <= 420
    Known optimal: obj = 220
    """
    tableau = _make([
        [5, 4, 3, 0, 0, 0, 0],
        [6, 4, 2, 1, 0, 0, 240],
        [3, 2, 5, 0, 1, 0, 270],
        [5, 6, 5, 0, 0, 1, 420],
    ])
    s = Simplex(tableau)
    s.solve()
    assert pytest.approx(-s.mat[0][-1], abs=1e-6) == 220.0


# ── 9. Degenerate LP — zero RHS can cause cycling or wrong pivot ─────────────

def test_degenerate_zero_rhs():
    """
    BUG: When RHS is 0 and pivot column entry is positive, the ratio is 0.
    But if another row has a negative pivot column entry, 0/negative = 0
    which ties and the solver may pick the wrong row.

    Maximize x1  s.t.  x1 <= 0, x1 + x2 <= 1
    Optimal: x1=0, x2=0 => obj=0
    """
    tableau = _make([
        [1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 1],
    ])
    s = Simplex(tableau)
    s.solve()
    assert pytest.approx(-s.mat[0][-1], abs=1e-6) == 0.0


# ── 10. Large coefficients — numeric stability stress test ───────────────────

def test_large_coefficients():
    """
    Maximize 1000x1 + 2000x2  s.t.  x1<=1e6, x2<=1e6
    Optimal: x1=1e6, x2=1e6 => obj=3e9
    """
    tableau = _make([
        [1000, 2000, 0, 0, 0],
        [1,    0,    1, 0, 1e6],
        [0,    1,    0, 1, 1e6],
    ])
    s = Simplex(tableau)
    s.solve()
    assert pytest.approx(-s.mat[0][-1], rel=1e-6) == 3e9


# ── 11. Pivot column entry is exactly zero in first constraint row ───────────

def test_pivot_column_zero_first_row():
    """
    BUG: Line 33 unconditionally computes mi = mat[1][-1] / mat[1][argma].
    If mat[1][argma] == 0, this is an immediate ZeroDivisionError before
    the loop even starts.

    Maximize x1 + x2  s.t.  x2<=3, x1+x2<=5
    When pivoting on x1, row 1 (x2<=3) has 0 in the x1 column.
    """
    tableau = _make([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 3],   # x1 coeff = 0 in first constraint
        [1, 1, 0, 1, 5],
    ])
    s = Simplex(tableau)
    s.solve()  # ZeroDivisionError at line 33
    assert pytest.approx(-s.mat[0][-1], abs=1e-6) == 5.0
