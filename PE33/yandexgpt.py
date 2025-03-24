import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
api_key = os.getenv("api_key") #здесь API ключ, скопированный из аккаунта
folder_id = os.getenv("folder_id") #Номер каталога


headers = {
    "Authorization": f"Api-Key {api_key}",
    "Content-Type": "application/json"
}

data = {
    "modelUri": f"gpt://{folder_id}/yandexgpt/latest",
    "completionOptions": {
        "stream": False,
        "temperature": 0,
        "maxTokens": "15000"
    },
    "messages": [
        {
            "role": "system",
            "text": "Не используй никакое форматирование и LATEX. Отвечай обычным текстом, разделяй на абзацы. Ты учитель математики. Я даю тебе задачу, тебе нужно ее решить и написать решение и объяснение. В конце напиши ответ."
        },
        {
            "role": "user",
            "text": "Какие примеры зашифрованы: АУ + УА = СОС? Одинаковые буквы обозначают одинаковые цифры, а разные буквы - разные цифры."
        }
    ]
}

response = requests.post(api_url, headers=headers, json=data)

if response.status_code == 200:
    result = response.json()
    print(result)

    try:
        with open("response.txt", "w", encoding="utf-8") as file:
            # Извлекаем текст ответа по вашей структуре
            answer_text = result['result']['alternatives'][0]['message']['text']

            # Декодируем спецсимволы (если есть) и записываем
            file.write(answer_text)

        print("Ответ сохранён в response.txt")

    except KeyError:
        print("Ошибка: Неверная структура ответа API")
    except Exception as e:
        print(f"Ошибка записи: {str(e)}")

else:
    print(f"Error: {response.status_code}, {response.text}")