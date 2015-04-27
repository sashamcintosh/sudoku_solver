#################################
# 
# Sasha McIntosh - sam2270
# Michael Saltzman - mjs
# 
# Visual Interfaces to Computers
# Final Project
# 
# sudoku_solver.py
# 
#################################

from config import *
from sudoku import *
import cv2

def display(name, img):
    # display image
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get(filename):
    img = cv2.imread(filename, 1)
    gray_img = cv2.imread(filename, 0)
    
    return img, gray_img

def clean(img):
    # absolute threshold, binary image with white foreground
    ret, binary_img = cv2.threshold(img,THRESH,255,cv2.THRESH_BINARY)
    return ret, binary_img

def main():
    # import image
    img, gray_img = get('puzzles/img3.png')
    display('Grayscale',gray_img)

    # process clean image
    ret, binary_img = clean(gray_img)
    display('Binary',binary_img)

    # grid edge and cell detection

    # extract numbers
    # function formating
    grid1  = '003020600900305001001806400008102900700000008006708200002609500800203009005010300'
    grid2  = '4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......'
    hard1  = '.....6....59.....82....8....45........3........6..3.54...325..6..................'

    # solve and print sudoku
    #print 'Sudoku:'
    #display(grid_values(grid1))

    #print 'Solved Sudoku:'
    #display(solve(grid1))



main()