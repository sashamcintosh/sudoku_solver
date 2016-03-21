# Vision Based Sudoku Solver
Created by Sasha McIntosh & Michael Saltzman
for Visual Interfaces to Computers
on Tuesday, May 12th 2015

## 1. Motivation
Sudoku is a number­puzzle game that has become very popular in recent times and is typically featured in newspapers, magazines, and stand­alone books. The sudoku puzzle consists of a 9x9 square grid. The goal of the puzzle is to fill the grid with numbers from 1­9. Each row and column can only contain one instance of each number. Furthermore, each 3x3 grid within the puzzle can only contain one instance of each number. Typically, some of the numbers are given as a starting point. Fewer starting numbers indicates a higher difficulty. The goal is then to fill in the puzzle, given these constraints.

Sudoku is a number­puzzle that is well known and understood, and widely played. It has a fixed set of rules, inputs and logic making it a reasonable vision and machine learning challenge. For these reasons we chose to construct a system that solves a sudoku puzzle, given an image of the puzzle. Captured as an image in its physical form, sudoku puzzle extraction provides a reasonable set of challenges. Challenges vary based on the physical form and can include glare, shadow, warped bounding lines, blurring and fading.

## 2. Program Input
For this project, we limited the input space significantly to control various factors. We used 4 images to test. Each image was printed on white paper and the sudoku puzzle consists of black ink with common font faces. No handwritten puzzles were used. Each image was captured using an iPhone 5s and the camera was placed directly over the paper to control the rotation and angle of the photo. The room was well­lit to reduce shadowing. Each image was then scaled down by 1⁄4 to reduce the quality for contouring. (higher quality photos generally resulted in worse performance).

## 3. Grid Extraction
### Implementation
First, we preprocess the image. This involves converting the image to grayscale, applying a Gaussian filter to blur the image and remove noise, and finally applying an adaptive threshold to binarize the image. Applying global thresholds did not work in this case as the images did have a bit of shadowing. Global thresholds resulted in very choppy binary images.

```python
img=cv2.imread(img_name)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(5,5),0)
thresh=cv2.adaptiveThreshold(gray,255,1,1,11,2)
```













