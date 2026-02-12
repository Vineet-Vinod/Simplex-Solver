from typing import List
from math import inf
import sys

class Simplex():
    def __init__(self, eqns: List[List[str]]):
        try:
            self.mat: List[List[float]] = [[float(entry) for entry in eqn] for eqn in eqns]
        except:
            print('Could not parse the matrix. Please check and try again')
            sys.exit(1)
        
        self.eqns: int = len(self.mat)
        self.vars: int = len(self.mat[0])

    def solve(self) -> None:
        """
        3 cases
        1. Feasible and bounded
        2. Feasible and unbounded - check at the start of every loop and report if unbounded
        3. Infeasible - cannot bring to standard form as standard form gives A solution - don't need to handle
        """
        while any(val > 0 for val in self.mat[0][:-1]): # Can keep simplexing
            ma, argma = self.mat[0][0], 0
            for xi, val in enumerate(self.mat[0][1:-1], 1):
                if val > ma:
                    ma = val
                    argma = xi
            
            if all(row[argma] <= 0 for row in self.mat[1:]): # Unbounded check
                print("Unbounded LP - no solution")
                break

            mi, argmi = inf, 0 # self.mat[1][-1] / self.mat[1][argma], 1
            for yi in range(1, self.eqns):
                if self.mat[yi][argma] and \
                   self.mat[yi][-1] / self.mat[yi][argma] > 0 \
                   and mi > self.mat[yi][-1] / self.mat[yi][argma]:
                    mi = self.mat[yi][-1] / self.mat[yi][argma]
                    argmi = yi
            
            print(f"Pivoting around ({argma}, {argmi})")
            div = self.mat[argmi][argma]
            for i in range(self.vars): self.mat[argmi][i] /= div
            for i in range(self.eqns):
                if i != argmi:
                    mult = self.mat[i][argma] / self.mat[argmi][argma]
                    for j in range(self.vars):
                        self.mat[i][j] -= mult * self.mat[argmi][j]

    def solution(self) -> str:
        soln: List[float] = []
        for xi in range(self.vars - 1):
            if sum(row[xi] == 0.0 for row in self.mat) == self.eqns - 1:
                print(f"x{xi} is a basic variable")
                for row in self.mat:
                    if row[xi]:
                        soln.append(str(row[-1] / row[xi]))
            else:
                print(f"x{xi} is a free variable")
                soln.append("0")
        
        return f"Solution is: ({", ".join(soln)}) and the objective value is {-self.mat[0][-1]}"
