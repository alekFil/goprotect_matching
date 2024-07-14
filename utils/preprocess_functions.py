import re

import nltk
from num2words import num2words

# Загружаем необходимые ресурсы

nltk.download("punkt")

nltk.download("stopwords")


def simple_preprocess_text(text):
    # Удаляем служебные символы (перенос строки, табуляция и т.д.)
    text = re.sub(r"[\n\t\r]", " ", text)

    # Удаление пунктуации
    text = re.sub(r"[^\w\s]", " ", text)

    # Удаление отдельных букв
    text = re.sub(r"\b[А-ЯЁа-яё]\b", " ", text)

    # Замена букв ё
    text = re.sub(r"[Ёё]", "е", text)

    # Регулярное выражение для поиска различных обозначений номера, включая случаи, когда за ними сразу идут цифры
    pattern = re.compile(r"\b(?:No|no|N|NO|№)(\d*)\b")

    # Замена найденных обозначений на "номер"
    text = pattern.sub(lambda match: f" {match.group(1)}", text)

    # Удаление лишних пробелов
    text = re.sub(r"\s+", " ", text)

    # Удаление пробелов в начале и в конце
    text = text.strip()

    # # Токенизация
    # words = word_tokenize(text.lower(), language="russian")

    # # Удаление стоп-слов
    # stop_words = set(stopwords.words("russian"))
    # filtered_words = [word for word in words if word not in stop_words]

    # return " ".join(filtered_words)
    return text


def replace_numbers_with_text(text):
    # Функция для замены чисел на их текстовое представление
    def num_to_text(match):
        num = match.group(0)
        return num2words(int(num), lang="ru")

    # Регулярное выражение для поиска чисел
    pattern = re.compile(r"\d+")

    # Замена чисел на текст
    return pattern.sub(num_to_text, text)
