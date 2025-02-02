def count_numbers_with_three(n: int) -> int:
    """Подсчитывает количество чисел от 1 до n (включительно), содержащих хотя бы одну цифру '3'.

    Args:
        n (int): Верхняя граница диапазона (включительно).

    Returns:
        int: Количество чисел, содержащих цифру '3'.

    Examples:
        >>> count_numbers_with_three(14)
        2  # Числа 3 и 13
    """

    def count_without_three(num: int) -> int:
        """Подсчитывает числа <= num без цифры '3' с использованием динамического программирования по цифрам."""
        s = str(num)
        length = len(s)
        # Мемоизация: [позиция][ограничение][была_ли_тройка]
        memo = [[[0] * 2 for _ in range(10)] for __ in range(length)]

        def dfs(pos: int, tight: bool, has_three: bool) -> int:
            """Рекурсивная функция для подсчёта чисел.

            Args:
                pos (int): Текущая позиция в числе.
                tight (bool): Ограничение на текущую цифру (True = нельзя превышать исходную цифру).
                has_three (bool): Была ли уже встречена цифра '3'.

            Returns:
                int: Количество чисел, удовлетворяющих условиям.
            """
            if pos == length:
                return 0 if has_three else 1  # Возвращаем 1, если '3' не встретилась
            if memo[pos][tight][has_three] != 0:
                return memo[pos][tight][has_three]  # Используем кэш

            limit = int(s[pos]) if tight else 9  # Максимальная допустимая цифра
            total = 0
            for d in range(0, limit + 1):
                new_tight = tight and (d == limit)  # Обновляем флаг ограничения
                new_has_three = has_three or (d == 3)  # Проверяем, появилась ли '3'
                if new_has_three:
                    continue  # Пропускаем числа с '3'
                # Рекурсивно обрабатываем следующую позицию
                total += dfs(pos + 1, new_tight, new_has_three)

            memo[pos][tight][has_three] = total  # Сохраняем результат в кэш
            return total

        return dfs(0, True, False)  # Начинаем с первой позиции, с ограничением

    total_numbers = n
    numbers_without_three = count_without_three(n)
    # Общее количество минус числа без '3' = числа с хотя бы одной '3'
    return total_numbers - numbers_without_three


if __name__ == "__main__":
    result = count_numbers_with_three(2024)
    print(f"Количество чисел с хотя бы одной цифрой '3': {result}")