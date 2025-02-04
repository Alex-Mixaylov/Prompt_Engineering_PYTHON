import math

def square(side):
    """Вычисляет периметр, площадь и диагональ квадрата.

    Args:
        side (float): Длина стороны квадрата.

    Returns:
        tuple: Кортеж, содержащий:
            - периметр квадрата,
            - площадь квадрата,
            - диагональ квадрата.

    Raises:
        ValueError: Если сторона меньше или равна нулю.
    """
    # Проверяем, что сторона квадрата положительна
    if side <= 0:
        raise ValueError("Сторона квадрата должна быть положительным числом.")

    # Вычисляем периметр квадрата: 4 * сторона
    perimeter = 4 * side

    # Вычисляем площадь квадрата: сторона в квадрате
    area = side ** 2

    # Вычисляем диагональ квадрата: сторона * sqrt(2)
    diagonal = side * math.sqrt(2)

    # Возвращаем кортеж из трех значений через запятую
    return perimeter, area, diagonal


# Запуск функции
if __name__ == "__main__":
    try:
        perim, sq_area, diag = square(5)
        print(f"Периметр: {perim}, Площадь: {sq_area}, Диагональ: {diag}")
    except ValueError as error:
        print("Ошибка:", error)
