def sum_range(start, end):
    """Возвращает сумму всех целых чисел от start до end включительно.

    Args:
        start (int): Начальное значение диапазона.
        end (int): Конечное значение диапазона.

    Returns:
        int: Сумма всех целых чисел от start до end включительно.

    Raises:
        ValueError: Если start больше end.
    """
    # Проверяем, что начальное значение не больше конечного
    if start > end:
        raise ValueError("Начальное значение не может быть больше конечного.")

    total = 0  # Инициализируем переменную для накопления суммы

    # Перебираем все числа от start до end включительно
    for num in range(start, end + 1):
        total += num  # Прибавляем текущее число к общей сумме

    return total  # Возвращаем вычисленную сумму


# Пример использования функции
if __name__ == "__main__":
    try:
        result = sum_range(1, 12)
        print("Сумма чисел от 1 до 12:", result)
    except ValueError as e:
        print("Ошибка:", e)
