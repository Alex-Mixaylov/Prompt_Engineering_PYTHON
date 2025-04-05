import os
import telebot
import io
from voice import get_all_voices, generate_audio
from dotenv import load_dotenv
import logging

load_dotenv()

elevenlabs_api_key = os.getenv("elevenlabs_api_key")
telegram_bot_token = os.getenv("telegram_bot_token")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Инициализация бота
bot = telebot.TeleBot(telegram_bot_token)

# Получаем все голоса из модуля voice.py
voices = get_all_voices()

# Создание клавиатуры для выбора голоса
voice_buttons = telebot.types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
for voice in voices.voices:
    voice_name = voice.name  # Получаем имя голоса
    button = telebot.types.KeyboardButton(voice_name)
    voice_buttons.add(button)

# Словарь для хранения выбранного голоса пользователем
selected_voice = {}

@bot.message_handler(commands=['start'])
def send_welcome(message):
    """
    Обрабатывает команду /start и отправляет приветственное сообщение с выбором голоса.

    Args:
        message (telebot.types.Message): Сообщение от пользователя.
    """
    logger.info(f"Пользователь {message.from_user.id} отправил команду /start")
    bot.reply_to(message,
                 "Привет! Я бот для создания озвучки! Выбери голос, который будет использоваться при создании озвучки:",
                 reply_markup=voice_buttons)

@bot.message_handler(func=lambda message: message.text in [voice.name for voice in voices.voices])
def voice_selected(message):
    """
    Обрабатывает выбор голоса пользователем и запрашивает текст для озвучки.

    Args:
        message (telebot.types.Message): Сообщение от пользователя с выбранным голосом.
    """
    user_id = message.from_user.id
    selected_voice[user_id] = message.text
    logger.info(f"Пользователь {user_id} выбрал голос: {message.text}")
    bot.reply_to(message, f"Вы выбрали голос: {message.text}. Теперь введите текст для озвучки:")

@bot.message_handler(func=lambda message: True)
def generate_voice(message):
    """
    Обрабатывает текстовое сообщение пользователя, генерирует аудио с выбранным голосом и отправляет его пользователю.

    Args:
        message (telebot.types.Message): Сообщение от пользователя с текстом для озвучки.
    """
    user_id = message.from_user.id
    if user_id in selected_voice:
        try:
            # Ищем голос по имени
            voice_name = selected_voice[user_id]
            voice = next(voice for voice in voices.voices if voice.name == voice_name)
            voice_id = voice.voice_id

            # Генерация аудио с выбранным голосом с использованием функции из voice.py
            audio_generator = generate_audio(message.text, voice_id)

            # Запись аудио в байтовый поток
            audio_bytes = io.BytesIO()
            for chunk in audio_generator:
                audio_bytes.write(chunk)

            # Сохраняем аудио в файл и отправляем пользователю
            audio_bytes.seek(0)  # Возвращаемся в начало потока
            bot.send_audio(user_id, audio_bytes)
            logger.info(f"Пользователь {user_id} получил аудио для текста: {message.text}")
        except Exception as e:
            logger.error(f"Ошибка при генерации аудио для пользователя {user_id}: {e}")
            bot.reply_to(message, "Произошла ошибка при генерации аудио. Попробуйте позже.")
    else:
        logger.info(f"Пользователь {user_id} отправил текст без выбора голоса")
        bot.reply_to(message, "Сначала выберите голос командой /start")

if __name__ == '__main__':
    logger.info("Бот запущен и готов к работе")
    bot.polling(none_stop=True)