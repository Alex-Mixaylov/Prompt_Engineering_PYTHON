def caesar_encrypt(text: str, shift: int) -> str:
    """Шифрует текст с использованием шифра Цезаря с заданным сдвигом.

    Args:
        text: Исходный текст для шифрования.
        shift: Величина сдвига (может быть отрицательной).

    Returns:
        Зашифрованная строка. Неалфавитные символы остаются без изменений.

    Examples:
        >>> caesar_encrypt('ABC', 3)
        'DEF'
        >>> caesar_encrypt('xyz!', 5)
        'cde!'
    """

    def shift_char(char: str, offset: int) -> str:
        """Сдвигает символ на указанное количество позиций в алфавите."""
        if not char.isalpha():
            return char  # Не изменяем неалфавитные символы

        # Определяем базовый код для регистра (A=65, a=97)
        base = ord('A') if char.isupper() else ord('a')
        # Сдвиг с учетом зацикливания алфавита (26 букв)
        return chr((ord(char) - base + offset) % 26 + base)

    # Применяем сдвиг к каждому символу
    return ''.join(shift_char(c, shift) for c in text)


def caesar_decrypt(text: str, shift: int) -> str:
    """Дешифрует текст, зашифрованный шифром Цезаря.

    Args:
        text: Зашифрованный текст.
        shift: Сдвиг, который использовался при шифровании.

    Returns:
        Расшифрованная строка.

    Examples:
        >>> caesar_decrypt('DEF', 3)
        'ABC'
        >>> caesar_decrypt('cde!', 5)
        'xyz!'
    """
    # Дешифровка = шифрование с обратным сдвигом
    return caesar_encrypt(text, -shift)


if __name__ == "__main__":
    # Пример использования
    original = "DeepSeek R1 ia a new Chinese AI"
    shift = 6

    encrypted = caesar_encrypt(original, shift)
    decrypted = caesar_decrypt(encrypted, shift)

    print(f"Исходный текст: {original}")
    print(f"Зашифрованный:   {encrypted}")
    print(f"Расшифрованный:  {decrypted}")