import os
from dotenv import load_dotenv
from elevenlabs import Voice, VoiceSettings
from elevenlabs.client import ElevenLabs
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

client = ElevenLabs(api_key=os.getenv("elevenlabs_api_key"))

def get_all_voices():
    """
    Получает все доступные голоса от ElevenLabs.

    Returns:
        Voices: Объект, содержащий список всех доступных голосов.
    """
    logger.info("Получение всех голосов")
    voices = client.voices.get_all()
    return voices

def generate_audio(text: str, voice_id: str):
    """
    Генерирует аудио на основе текста и идентификатора голоса.

    Args:
        text (str): Текст, который нужно озвучить.
        voice_id (str): Идентификатор голоса, который будет использоваться для озвучки.

    Returns:
        generator: Генератор, возвращающий байты аудио.
    """
    logger.info(f"Генерация аудио для текста: {text} с голосом: {voice_id}")
    audio = client.generate(
        text=text,
        voice=Voice(
            voice_id=voice_id,
            settings=VoiceSettings(stability=0.75, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
        ),
        model="eleven_multilingual_v2"
    )
    return audio