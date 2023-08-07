import copy


# function for iterating мшф array and counting "ones", after one element
# was found, all connected elements would be excluded
def get_islands_number(islands_grid):
    if len(islands_grid) == 0:
        return 0
    n = len(islands_grid)
    m = len(islands_grid[0])
    isl_count = 0
    for i in range(n):
        for j in range(m):
            if islands_grid[i][j] == 1:
                isl_count += 1
            mark_calculated_islands(i, j, n, m, islands_grid)
    return isl_count


# function to recursively exclude all connected elements
def mark_calculated_islands(i, j, n, m, islands_grid):
    if i < 0 or j < 0 or i >= n or j >= m:
        return

    if islands_grid[i][j] == 0:
        return
    else:
        islands_grid[i][j] = 0

    # rules that define connectivity
    # for diagonal connectivity there would be another four
    mark_calculated_islands(i+1, j, n, m, islands_grid)
    mark_calculated_islands(i, j+1, n, m, islands_grid)
    mark_calculated_islands(i-1, j, n, m, islands_grid)
    mark_calculated_islands(i, j-1, n, m, islands_grid)


# final island calculation and printing output
def calculate_islands(islands_grid):
    # create deepcopy to prevent changing original numpy array
    grid = copy.deepcopy(islands_grid)
    print('Islands array:')
    print(islands_grid)
    print(f'Calculated number of islands: {get_islands_number(grid)}')
    print()


if __name__ == "__main__":
    test_map_1 = [[0, 1, 0], [0, 0, 0], [0, 1, 1]]
    test_map_2 = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]
    test_map_3 = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1]]
    test_map_4 = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]

    calculate_islands(test_map_1)
    calculate_islands(test_map_2)
    calculate_islands(test_map_3)
    calculate_islands(test_map_4)
