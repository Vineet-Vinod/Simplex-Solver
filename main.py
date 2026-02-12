from simplex import Simplex


def main(filename: str) -> None:
    """
    Expect filename to contain lines with comma separated values
    representing entries in the matrix - not trying to convert
    word problems into LPs (idt this is possible programmatically),
    nor trying to convert eqns in set notation to simplex standard form
    as it involves parsing + getting to the form which is more an exercise
    in boilerplate programming than actual logic which is not the point here
    (AI can do that really well and we can check its work)
    """
    with open(filename, "r") as file:
        eqns = [line.strip().split(",") for line in file.readlines()]
    solver = Simplex(eqns)
    solver.solve()
    print(solver.solution())


if __name__ == "__main__":
    main("problem.txt")
