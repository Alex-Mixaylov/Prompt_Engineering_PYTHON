def compress_string(input_str: str) -> str:
    """Сжимает строку, заменяя повторяющиеся символы на символ + количество повторений.

    Если входная строка пустая, возвращает пустую строку.

    Args:
        input_str: Исходная строка для сжатия. Может содержать любые буквы.

    Returns:
        Сжатая строка в формате [символ][количество]. Например, "aab" -> "a2b1".

    Examples:
        >>> compress_string("aabcccccaaa")
        'a2b1c5a3'
        >>> compress_string("abcd")
        'a1b1c1d1'
        >>> compress_string("")
        ''
    """
    if not input_str:
        return ""

    compressed = []  # Список для хранения частей сжатой строки
    current_char = input_str[0]  # Текущий обрабатываемый символ
    count = 1  # Счетчик повторений

    for char in input_str[1:]:
        if char == current_char:
            # Увеличиваем счетчик при совпадении символов
            count += 1
        else:
            # Добавляем в результат предыдущий символ и его счетчик
            compressed.append(f"{current_char}{count}")
            # Начинаем отсчет для нового символа
            current_char = char
            count = 1

    # Добавляем последний обработанный символ
    compressed.append(f"{current_char}{count}")

    # Склеиваем все части в итоговую строку
    return "".join(compressed)

if __name__ == "__main__":
    test_str = "aabcccccaaa"
    print(f"Исходная строка: {test_str}")
    print(f"Сжатая строка: {compress_string(test_str)}")