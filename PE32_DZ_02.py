def longest_increasing_path(matrix: list[list[int]]) -> int:
    """Возвращает длину наибольшего возрастающего пути в матрице используя DFS с мемоизацией.

    Алгоритм:
    1. Используем динамическое программирование для кэширования результатов
    2. Для каждой клетки вычисляем максимальный путь через итеративный DFS:
       - Первый проход: помечаем клетку как посещенную и добавляем соседей
       - Второй проход: вычисляем максимальное значение пути на основе соседей
    3. Обход всех клеток гарантирует обработку всех возможных путей

    Сложность: O(m*n), где m и n - размерности матрицы. Каждая клетка обрабатывается один раз.

    Args:
        matrix: Двумерный список целых чисел, представляющий матрицу.

    Returns:
        Целое число, длина наибольшего возрастающего пути.

    Examples:
        >>> longest_increasing_path([[9,9,4],[6,6,8],[2,1,1]])
        4
        >>> longest_increasing_path([[3,4,5],[3,2,6],[2,2,1]])
        4
        >>> longest_increasing_path([[1]])
        1
    """
    if not matrix or not matrix[0]:
        return 0

    rows, cols = len(matrix), len(matrix[0])
    dp = [[0] * cols for _ in range(rows)]
    max_length = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(rows):
        for j in range(cols):
            if dp[i][j] == 0:
                stack = [(i, j, False)]
                while stack:
                    x, y, visited = stack.pop()
                    if not visited:
                        if dp[x][y] != 0:
                            continue
                        dp[x][y] = -1  # Временная маркировка
                        stack.append((x, y, True))
                        # Добавляем соседей с БОЛЬШИМИ значениями
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < rows and 0 <= ny < cols:
                                if matrix[nx][ny] > matrix[x][y]:
                                    stack.append((nx, ny, False))
                    else:
                        current_max = 0
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < rows and 0 <= ny < cols:
                                if matrix[nx][ny] > matrix[x][y]:
                                    current_max = max(current_max, dp[nx][ny])
                        dp[x][y] = current_max + 1
                        max_length = max(max_length, dp[x][y])

    return max_length


if __name__ == "__main__":
    # Примеры из задания
    test_cases = [
        ([[9, 9, 4], [6, 6, 8], [2, 1, 1]], 4),
        ([[3, 4, 5], [3, 2, 6], [2, 2, 1]], 4),
        ([[1]], 1),
    ]

    for matrix, expected in test_cases:
        result = longest_increasing_path(matrix)
        print(f"Matrix: {matrix}")
        print(f"Expected: {expected}, Result: {result}")
        print(f"Test {'Passed' if result == expected else 'Failed'}\n")