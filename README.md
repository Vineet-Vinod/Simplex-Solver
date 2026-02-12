# Simplex Solver

A simplex solver implemented in Python.

Note that the Solver is NOT meant for production use. There are floating point errors (should store numerator and denominator separately to handle this) that are possible and in rare cases, there may be pivot selection stability errors (the textbook simplex method runs into these errors). However, for a textbook implementation, these are acceptable tradeoffs.
