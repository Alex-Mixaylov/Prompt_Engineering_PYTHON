def longest_increasing_path(matrix):
    """Finds the length of the longest strictly increasing path in a matrix.

    Moves can be made up, down, left, or right (no diagonal moves). Each move must
    go to a neighboring cell with a higher value than the current cell. This function
    uses depth-first search (DFS) with memoization to avoid recomputing path lengths
    for cells that have already been visited.

    Args:
        matrix (List[List[int]]): The matrix of integers where we want to find the
            longest increasing path.

    Returns:
        int: The length of the longest path of strictly increasing values in the matrix.
        If the matrix is empty, returns 0.
    """
    # Check for empty matrix. If there are no rows or no columns, return 0.
    if not matrix or not matrix[0]:
        return 0

    m, n = len(matrix), len(matrix[0])
    # Cache to store the length of the longest increasing path starting from each cell.
    # A value of 0 means the cell's longest path has not been computed yet.
    cache = [[0] * n for _ in range(m)]

    # Directions for moving up, down, left, or right from a given cell.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def dfs(i, j):
        """Depth-first search that returns the length of the longest increasing path from cell (i, j)."""
        # If this cell's longest path has been computed before, return it from the cache.
        if cache[i][j] != 0:
            return cache[i][j]

        # Start with length 1 (the path includes the cell itself).
        max_path_length = 1
        current_value = matrix[i][j]

        # Explore all four possible directions.
        for di, dj in directions:
            ni, nj = i + di, j + dj  # Coordinates of the neighboring cell.
            # Check if the neighbor is within bounds and has a greater value than the current cell.
            if 0 <= ni < m and 0 <= nj < n and matrix[ni][nj] > current_value:
                # Recurse into the neighbor cell to find the path length from there.
                length_from_neighbor = dfs(ni, nj)
                # Update the maximum path length if moving to this neighbor yields a longer path.
                if 1 + length_from_neighbor > max_path_length:
                    max_path_length = 1 + length_from_neighbor

        # Cache the computed longest path length for this cell to avoid re-computation in the future.
        cache[i][j] = max_path_length
        return max_path_length

    # Compute the longest increasing path starting from each cell, track the global maximum.
    longest_path = 0
    for i in range(m):
        for j in range(n):
            current_path_length = dfs(i, j)
            if current_path_length > longest_path:
                longest_path = current_path_length

    return longest_path

# Примеры использования:
matrix1 = [[9, 9, 4],
           [6, 6, 8],
           [2, 1, 1]]
print(longest_increasing_path(matrix1))  # Expected output: 4
# Explanation: The longest increasing path is [1, 2, 6, 9].

matrix2 = [[3, 4, 5],
           [3, 2, 6],
           [2, 2, 1]]
print(longest_increasing_path(matrix2))  # Expected output: 4
# Explanation: The longest increasing path is [3, 4, 5, 6].

matrix3 = [[1]]
print(longest_increasing_path(matrix3))  # Expected output: 1
