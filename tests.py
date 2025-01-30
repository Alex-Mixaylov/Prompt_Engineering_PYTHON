import unittest
from unittest.mock import patch

import PY01_DZ_01 as  dungeon_game
import PY01_DZ_02 as  bank_account


#Тесты для  игры Полземелье

class TestDungeonQuest(unittest.TestCase):

    @patch('builtins.input', side_effect=['1', '1', '1', '1', '1'])  # Последовательность действий игрока
    def test_successful_escape(self, mock_input):
        """
        Тест на успешное завершение подземелья, если игрок всегда выбирает первый вариант.
        """
        with patch('builtins.print') as mock_print:
            dungeon_game.dungeon_quest()
            output_texts = [call[0][0] for call in mock_print.call_args_list]
            self.assertTrue(
                any("Поздравляю, ты победил!" in text for text in output_texts),
                "Финальное сообщение победы не найдено в выводе."
            )

    @patch('builtins.input', side_effect=['2', '1', '1', '2'])  # Альтернативный маршрут
    def test_alternate_path(self, mock_input):
        """
        Тестирование альтернативного пути, где игрок возвращается назад на развилке.
        """
        with patch('builtins.print') as mock_print:
            dungeon_game.dungeon_quest()
            output_texts = [call[0][0] for call in mock_print.call_args_list]
            self.assertTrue(
                any("Дверь не поддается" in text for text in output_texts),
                "Ожидаемое сообщение о закрытой двери не найдено."
            )

    @patch('builtins.input', side_effect=['1', '2', '1', '1', '1'])  # Игрок пропускает ключ, но пытается открыть дверь
    def test_no_key_door_fail(self, mock_input):
        """
        Тестирует ситуацию, когда игрок не находит ключ, но пытается открыть последнюю дверь.
        """
        with patch('builtins.print') as mock_print:
            dungeon_game.dungeon_quest()
            output_texts = [call[0][0] for call in mock_print.call_args_list]
            self.assertTrue(
                any("Дверь не поддается" in text for text in output_texts),
                "Сообщение о невозможности открыть дверь не найдено."
            )

    @patch('builtins.input', side_effect=['1', '1', '1', '2', '1'])  # Игрок находит ключ, но не берет меч
    def test_key_but_no_sword(self, mock_input):
        """
        Тестирует прохождение с ключом, но без меча (не влияет на результат).
        """
        with patch('builtins.print') as mock_print:
            dungeon_game.dungeon_quest()
            output_texts = [call[0][0] for call in mock_print.call_args_list]
            self.assertTrue(
                any("Поздравляю, ты победил!" in text for text in output_texts),
                "Финальное сообщение победы не найдено."
            )

# Тесты для  банковского аккаунта

class TestBankAccount(unittest.TestCase):

    def setUp(self):
        """Создаем новый банковский счет перед каждым тестом."""
        self.account = bank_account.BankAccount()

    def test_deposit_negative_amount(self):
        """Тест попытки внести отрицательную сумму."""
        result = self.account.deposit(-50)
        self.assertEqual(result, "Введите положительную сумму для внесения.")

    def test_withdraw_exceeding_amount(self):
        """Тест попытки снять больше, чем есть на счете."""
        self.account.deposit(50)
        result = self.account.withdraw(100)
        self.assertEqual(result, "Недостаточно средств на счете.")

    def test_withdraw_negative_amount(self):
        """Тест попытки снять отрицательную сумму."""
        result = self.account.withdraw(-30)
        self.assertEqual(result, "Введите положительную сумму для снятия.")

if __name__ == '__main__':
    unittest.main()