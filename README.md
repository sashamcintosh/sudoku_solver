# Vision Based Sudoku Solver
Created by [Sasha McIntosh](@sashamcintosh) & Michael Saltzman
For a full write up of this project, see [documentation/documentation.md](./documentation/documentation.md)

Running from the command line example:
```
$ cd sudoku_solver
$ python solve.py puzzles/img1.py
```

Hit enter to step through the program's image processing. The program writes to the console the following 2d arrays:
1. The Extracted Sudoku: Using digit recognition based on a training set, what the percived values for each cell of the sudoku.
ex:
```
Extracted Sudoku:
[[0 2 3 1 1 3 6 0 0]
 [0 0 5 5 1 1 0 4 0]
 [1 0 0 0 1 2 0 0 7]
 [0 1 0 0 1 0 7 0 0]
 [7 0 1 0 4 0 5 0 7]
 [0 0 1 0 7 0 0 8 0]
 [0 0 0 4 0 0 0 0 4]
 [0 1 0 0 0 7 7 0 0]
 [0 0 7 1 0 0 5 1 0]]
```

2. The Actual Sudoku: Taken from the [solutions.py](solutions.py) file, the actual values at each cell of the sudoku. The accuracy of the digit recognition is based on the comparison of these two matrices.
ex:
```
Actual Sudoku:
[[0 1 5 0 0 3 8 0 0]
 [0 0 3 5 0 0 0 4 0]
 [6 0 0 0 0 2 0 0 7]
 [0 4 0 0 6 0 7 0 0]
 [2 0 8 0 4 0 5 0 3]
 [0 0 6 0 3 0 0 9 0]
 [9 0 0 3 0 0 0 0 2]
 [0 8 0 0 0 9 6 0 0]
 [0 0 1 6 0 0 4 7 0]]
Digit Recognition Accuracy: 65.4320987654
```

3. The Solution: The program's solution of the sudoku based on the "actual sudoku" matrix.
ex:
```
Solution:
[[7 1 5 4 9 3 8 2 6]
 [8 2 3 5 7 6 9 4 1]
 [6 9 4 1 8 2 3 5 7]
 [3 4 9 2 6 5 7 1 8]
 [2 7 8 9 4 1 5 6 3]
 [1 5 6 8 3 7 2 9 4]
 [9 6 7 3 5 4 1 8 2]
 [4 8 2 7 1 9 6 3 5]
 [5 3 1 6 2 8 4 7 9]]
```











