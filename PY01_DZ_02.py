class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            return f"Внесено {amount} денег. Новый баланс: {self.balance}"
        return "Введите положительную сумму для внесения."

    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            return f"Снято {amount} денег. Новый баланс: {self.balance}"
        elif amount <= 0:
            return "Введите положительную сумму для снятия."
        return "Недостаточно средств на счете."

    def get_balance(self):
        return self.balance


def bank_menu():
    """ Функция пользовательского интерфейса (CLI), использующая BankAccount. """
    account = BankAccount()

    while True:
        print("\n1. Внести деньги")
        print("2. Снять деньги")
        print("3. Проверить баланс")
        print("4. Выход")

        choice = input("Выберите действие (1-4): ")

        if choice == '1':
            amount = float(input("Введите сумму для внесения: "))
            print(account.deposit(amount))
        elif choice == '2':
            amount = float(input("Введите сумму для снятия: "))
            print(account.withdraw(amount))
        elif choice == '3':
            print("Текущий баланс:", account.get_balance())
        elif choice == '4':
            print("Выход из программы. До свидания!")
            break
        else:
            print("Неверный ввод. Пожалуйста, выберите от 1 до 4.")


if __name__ == "__main__":
    bank_menu()
