def longest_increasing_path(matrix):
    """Находит длину самого длинного возрастающего пути в матрице.

    Args:
        matrix (List[List[int]]): Матрица целых чисел размером m x n.

    Returns:
        int: Длина самого длинного возрастающего пути. Возвращает 0, если матрица пуста.

    Examples:
        >>> longest_increasing_path([[9,9,4],[6,6,8],[2,1,1]])
        4  # Путь: [1, 2, 6, 9]
        >>> longest_increasing_path([[3,4,5],[3,2,6],[2,2,1]])
        4  # Путь: [3, 4, 5, 6]
        >>> longest_increasing_path([[1]])
        1
    """
    # Проверка на пустую матрицу или пустую строку
    if not matrix or not matrix[0]:
        return 0

    # Размеры матрицы
    m, n = len(matrix), len(matrix[0])

    # Кэш для хранения длин путей, изначально заполнен нулями
    cache = [[0] * n for _ in range(m)]

    # Возможные направления движения: вверх, вниз, влево, вправо
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def dfs(i, j):
        """Рекурсивно вычисляет длину самого длинного возрастающего пути из ячейки (i, j).

        Args:
            i (int): Индекс строки текущей ячейки.
            j (int): Индекс столбца текущей ячейки.

        Returns:
            int: Длина самого длинного возрастающего пути, начинающегося из (i, j).
        """
        # Если значение уже вычислено, возвращаем его из кэша
        if cache[i][j] != 0:
            return cache[i][j]

        # Максимальная длина пути из соседних ячеек
        max_path = 0
        for di, dj in directions:
            ni, nj = i + di, j + dj
            # Проверяем, что соседняя ячейка в пределах матрицы и значение в ней больше
            if (0 <= ni < m and 0 <= nj < n and
                matrix[ni][nj] > matrix[i][j]):
                max_path = max(max_path, dfs(ni, nj))

        # Длина пути из текущей ячейки — это 1 плюс максимум из соседних путей
        cache[i][j] = max_path + 1
        return cache[i][j]

    # Находим максимальную длину пути, проверяя каждую ячейку как стартовую
    max_length = 0
    for i in range(m):
        for j in range(n):
            max_length = max(max_length, dfs(i, j))

    return max_length

# Примеры для проверки
if __name__ == "__main__":
    # Пример 1
    matrix1 = [[9, 9, 4], [6, 6, 8], [2, 1, 1]]
    print(longest_increasing_path(matrix1))  # Ожидаемый вывод: 4

    # Пример 2
    matrix2 = [[3, 4, 5], [3, 2, 6], [2, 2, 1]]
    print(longest_increasing_path(matrix2))  # Ожидаемый вывод: 4

    # Пример 3
    matrix3 = [[1]]
    print(longest_increasing_path(matrix3))  # Ожидаемый вывод: 1