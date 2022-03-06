#include <iostream>
#include <vector>
#include <math.h>
#include <windows.h>

#define SIZE 9

int grid[SIZE][SIZE] = {
   {3, 0, 6, 5, 0, 8, 4, 0, 0},
   {5, 2, 0, 0, 0, 0, 0, 0, 0},
   {0, 8, 7, 0, 0, 0, 0, 3, 1},
   {0, 0, 3, 0, 1, 0, 0, 8, 0},
   {9, 0, 0, 8, 6, 3, 0, 0, 5},
   {0, 5, 0, 0, 9, 0, 6, 0, 0},
   {1, 3, 0, 0, 0, 0, 2, 5, 0},
   {0, 0, 0, 0, 0, 0, 0, 7, 4},
   {0, 0, 5, 2, 0, 6, 3, 0, 0}
 };

HANDLE hCon = GetStdHandle(STD_OUTPUT_HANDLE);

bool isPresentInColumn(int column, int number)
{
    for (unsigned int i = 0; i < SIZE; ++i) {
        if (grid[i][column] == number) {
            return true;
        }
    }
    return false;
}

bool isPresentInRow(int row, int number)
{
    for (unsigned int i = 0; i < SIZE; ++i) {
        if (grid[row][i] == number) {
            return true;
        }
    }
    return false;
}

bool isPresentInBox(int boxStartRow, int boxStartColumn, int number)
{
    for (unsigned int i = 0; i < std::sqrt(SIZE); ++i) {
        for (unsigned int k = 0; k < std::sqrt(SIZE); ++k) {
            if (grid[i + boxStartRow][k + boxStartColumn] == number) {
                return true;
            }
        }
    }
    return false;
}

//TODO pay attention here
bool findEmptyCellFrom(int &row, int &column)
{
    for (row = 0; row < SIZE; ++row) {
        for (column = 0; column < SIZE; ++column) {
            if (grid[row][column] == 0) {
                return true;
            }
        }
    }
    return false;
}

bool isValidPlace(int row, int column, int number)
{
    return !isPresentInColumn(column, number) &&
           !isPresentInRow(row, number) &&
           !isPresentInBox(row - (row % (int)std::sqrt(SIZE)), column - (column % (int)std::sqrt(SIZE)), number);
}

bool solveSudoku(int row = 0, int column = 0)
{
    if (!findEmptyCellFrom(row, column)) {
        return true;
    }
    for (int num = 1; num <= SIZE; ++num) {
        if (isValidPlace(row, column, num)) {
            grid[row][column] = num;
            if (solveSudoku(row, column)) {
                return true;
            }
            grid[row][column] = 0;
        }
    }
    return false;
}

void printGrid()
{
    for (unsigned int i = 0; i < SIZE; ++i) {
        for (unsigned int k = 0; k < SIZE; ++k) {
            if (grid[i][k] == 0) {
                SetConsoleTextAttribute(hCon, 4);
            } else {
                SetConsoleTextAttribute(hCon, 3);
            }
            std::cout << " " << grid[i][k];
            SetConsoleTextAttribute(hCon, 7);
        
            if (k != 0 && k != SIZE - 1 && (k + 1) % (int)std::sqrt(SIZE) == 0) {
                std::cout << " |";
            }
        }
        std::cout << std::endl;

        if (i != 0 && i != SIZE - 1 && (i + 1) % (int)std::sqrt(SIZE) == 0) {
            for (unsigned int k = 0; k < SIZE + std::sqrt(SIZE) - 1; ++k) {
                std::cout << " -";
            }
            std::cout << std::endl;
        }
    }
}

int main()
{
    if (SIZE != std::pow(std::sqrt(SIZE), 2)) {
        std::cout << "Provided grid cannot be solved. Cannot build boxes." << std::endl;
    }

    printGrid();
    std::cout << "---" << std::endl;
    if (solveSudoku()) {
        printGrid();
    } else {
        std::cout << "This sudoku cannot be solved." << std::endl;
    }

    return 0;
}

