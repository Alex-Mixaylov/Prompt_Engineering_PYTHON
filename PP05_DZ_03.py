import logging
from functools import lru_cache

# Настройка логгера: вывод в консоль и запись в лог-файл
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Обработчик для записи логов в файл (logfile.log)
file_handler = logging.FileHandler("logfile.log", mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Обработчик для вывода логов в консоль
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def count_numbers_with_three(n: int) -> int:
    """Подсчитывает количество чисел от 1 до n (включительно), содержащих хотя бы одну цифру '3'.

    Args:
        n (int): Верхняя граница диапазона (включительно).

    Returns:
        int: Количество чисел, содержащих цифру '3'.

    Raises:
        Exception: Если произошла ошибка при обработке числа.
    """
    logger.info("Начало вычисления count_numbers_with_three для n = %s", n)
    try:
        # Преобразуем число в строку для удобной работы с его цифрами
        s = str(n)
        length = len(s)
        logger.debug("Преобразовано число в строку: %s, длина: %s", s, length)
    except Exception as e:
        logger.error("Ошибка при преобразовании числа в строку: %s", e)
        raise

    @lru_cache(maxsize=None)
    def dfs(pos: int, tight: bool, has_three: bool) -> int:
        """Рекурсивная функция для подсчёта чисел, учитывая ограничения.

        Args:
            pos (int): Текущая позиция в строковом представлении числа.
            tight (bool): Флаг ограничения, True если цифры до данной позиции совпадают с исходными.
            has_three (bool): Флаг, указывающий, встречалась ли цифра '3' до текущей позиции.

        Returns:
            int: Количество чисел от позиции pos до конца, удовлетворяющих условиям.

        Raises:
            Exception: Если происходит ошибка в рекурсивном вызове.
        """
        logger.debug("Вход в dfs: pos = %s, tight = %s, has_three = %s", pos, tight, has_three)
        try:
            if pos == length:
                # Если дошли до конца числа, возвращаем 1, если цифра '3' встречалась, иначе 0
                result = 1 if has_three else 0
                logger.debug("Базовый случай достигнут на pos = %s, возвращаю %s", pos, result)
                return result

            # Определяем ограничение для текущей цифры
            limit = int(s[pos]) if tight else 9
            logger.debug("На позиции %s, ограничение (limit) = %s", pos, limit)
            total = 0
            for d in range(0, limit + 1):
                logger.debug("На позиции %s, рассматриваю цифру d = %s", pos, d)
                next_tight = tight and (d == limit)
                next_has_three = has_three or (d == 3)
                logger.debug("Следующий вызов dfs: pos = %s, next_tight = %s, next_has_three = %s",
                             pos + 1, next_tight, next_has_three)
                total += dfs(pos + 1, next_tight, next_has_three)
            logger.debug("Выход из dfs: pos = %s, возвращаю total = %s", pos, total)
            return total
        except Exception as e:
            logger.error("Ошибка в рекурсивном вызове на позиции %s: %s", pos, e)
            raise

    try:
        # dfs считает все числа от 0 до n, включая 0.
        result = dfs(0, True, False)
        # Так как функция dfs считает число 0, которое не входит в диапазон от 1 до n,
        # вычтем его, поскольку 0 не содержит '3'
        if n >= 0:
            result -= 1
        logger.info("Окончательный результат count_numbers_with_three: %s", result)
    except Exception as e:
        logger.error("Ошибка при вычислении результата: %s", e)
        raise

    return result

if __name__ == "__main__":
    try:
        result = count_numbers_with_three(2024)
        print(f"Количество чисел с хотя бы одной цифрой '3': {result}")
    except Exception as e:
        logger.critical("Произошла непредвиденная ошибка: %s", e)
