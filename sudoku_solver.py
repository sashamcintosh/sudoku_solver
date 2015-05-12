#################################
# 
# Sasha McIntosh - sam2270
# Michael Saltzman - mjs2287
# 
# Visual Interfaces to Computers
# Final Project
# 
# sudoku_solver2.py
# 
#################################

# based on a sdoku solver found at
# http://www.mathworks.com/company/newsletters/articles/solving-sudoku-with-matlab.html

import math
import numpy as np

def sudoku(X):
	# Fill in all "singletons". 
	# C is a cell array of candidate vectors
	# s is the first cell, if any, with one candidate. 
	# e is the first cell, if any, with no candidates. 
	C, s, e = candidates(X)
	while s.size>0 and e.size==0:
		s0, s1 = s[0], s[1]
		X[s0][s1] = C[s0][s1][0]
		C, s, e = candidates(X)

	# Return for impossible puzzles. 
	if e.size>0:
		return False

	# Recursive backtracking. 
	if np.any(X==0):
		Y = X[:,:]
		z = np.transpose( np.where(X==0) )[0]   # The first unfilled cell.
		z0, z1 = z[0], z[1]
		for r in C[z[0]][z[1]]:			# Iterate over candidates.
			X = Y[:,:]

			X[z0][z1] = r 				# Insert a tentative value.

			X = sudoku(X.copy())		# Recursive call.
			if np.any(X>0) and not np.any(X==0): 	# Found a solution.
				return X

	return X

# find the correct upper left corner of cube 
def tri(k):
	return int( 3*math.floor(k/3) )

def candidates(X):
	C = np.empty((9,9))
	C.fill(None)
	C = C.tolist()

	for i in range(0,9):
		for j in range(0,9):

			if X[i, j] == 0:
				z = range(0,10) #leading zero will be removed later

				# eliminate possibilities based on row
				for e in X[ i, np.nonzero(X[i,:])[0] ]:
					z[e] = 0
				# eliminate possibilities based on column
				for e in X[ np.nonzero(X[:,j])[0], j]:
					z[e] = 0
				# eliminate possibilities based on cube
				Ti = tri(i)
				Tj = tri(j)
				for e0, e1 in np.transpose( np.nonzero(X[ Ti:Ti+3, Tj:Tj+3 ]) ).tolist():
					z[ X[Ti+e0][Tj+e1] ] = 0

				C[i][j] = np.nonzero(z)[0].tolist()

	# maintain the number of candidates for each cell
	L = np.zeros((9,9)).astype(int)
	for i in range(0,9):
		for j in range(0,9):
			if type(C[i][j])==list:
				L[i,j]= len(C[i][j])

	s = np.transpose( np.where((X==0) & (L==1)) )
	if len(s) > 0:
		s = s[0]
	e = np.transpose( np.where((X==0) & (L==0)) )
	if len(e) > 0:
		e = e[0]

	return C, s, e

def stringToArray(puzzle):
	out = np.zeros((9,9))
	puzzle = list(puzzle)

	for i in range(0,9):
		for j in range(0,9):
			out[i,j] = puzzle.pop(0)

	return out.astype(int)

def main():
	puzzle = '000060030240000100007002008001400309700319002306007500500700800002000013070020000'
	#puzzle = '020030040600000003004000500000806000800010006000705000007000600400000008030040020'
	puzzle = stringToArray(puzzle)

	print puzzle
	print sudoku(puzzle)


main()























