# Словарь моделей с описанием
model_descriptions = {
    "exp42": "(Accuracy@1: 0.842) Предобработка текста: выделены регионы, расширенная (ОПФ удалены). Методы: TF-IDF, косинусное сходство",
    "exp39": "(Accuracy@1: 0.783) Предобработка текста: выделены регионы, расширенная (аббр. расшифрованы). Методы: TF-IDF, косинусное сходство",
    "exp31": "(Accuracy@1: 0.690) Предобработка текста: простая. Методы: TF-IDF, косинусное сходство",
    "exp38": "(Accuracy@1: 0.592) Предобработка текста: расширенная (аббр. расшифрованы). Методы: TF-IDF, косинусное сходство",
    "exp37": "(Accuracy@1: 0.554) Предобработка текста: расширенная (аббр. удалены). Методы: TF-IDF, косинусное сходство",
    "exp32": "(Accuracy@1: 0.533) Предобработка текста: простая. Методы: BOW, косинусное сходство",
    "exp33": "(Accuracy@1: 0.342) Предобработка текста: простая. Методы: fuzzywuzzy",
    "exp36": "(Accuracy@1: 0.332) Предобработка текста: простая. Методы: CatBoost",
    "exp34": "(Accuracy@1: 0.136) Предобработка текста: простая. Методы: BERT",
    "exp35": "(Accuracy@1: 0.092) Предобработка текста: простая. Методы: RuBERT",
}

# Создание списка для выбора в Streamlit
model_options = list(model_descriptions.values())
model_name_map = {desc: name for name, desc in model_descriptions.items()}

# Словарь моделей с описанием
model_full_descriptions = {
    # ----------------------------------------------------------------------------------
    "exp31": """
    **Способ предобработки наименований школ**: простая предобработка - удаление 
    служебных символов, пунктуации, отдельных букв, знаков номера, лишних пробелов, 
    пробелов в начале и в конце, замена "ё" на "е". Наименования предобработаны 
    одинаково и для рефенеса (эталонные названия школ) и для валидационной выборки.

    **Регионы**: не учитывались.

    **Способ векторизациии наименований школ**: TF-IDF.

    **Способ сопоставления наименований школ**: косинусное расстояние.

    <span style="background-color: yellow">**Замечания**: Приведение текста 
    к нижнему регистру не влияет на качество сопоставления.</span>
    """,
    # ----------------------------------------------------------------------------------
    "exp32": """
    **Способ предобработки наименований школ**: простая предобработка - удаление 
    служебных символов, пунктуации, отдельных букв, знаков номера, лишних пробелов, 
    пробелов в начале и в конце, замена "ё" на "е". Наименования предобработаны 
    одинаково и для рефенеса (эталонные названия школ) и для валидационной выборки.

    **Регионы**: не учитывались.

    **Способ векторизациии наименований школ**: bag of words (BOW).

    **Способ сопоставления наименований школ**: косинусное расстояние.

    <span style="background-color: yellow">**Замечания**: Приведение текста 
    к нижнему регистру не влияет на качество сопоставления.</span>
    """,
    # ----------------------------------------------------------------------------------
    "exp33": """
    **Способ предобработки наименований школ**: простая предобработка - удаление 
    служебных символов, пунктуации, отдельных букв, знаков номера, лишних пробелов, 
    пробелов в начале и в конце, замена "ё" на "е". Наименования предобработаны 
    одинаково и для рефенеса (эталонные названия школ) и для валидационной выборки.

    **Регионы**: не учитывались.

    **Способ векторизациии наименований школ**: не требуется.

    **Способ сопоставления наименований школ**: fuzzywuzzy - расстояние Левинштейна.
    """,
    # ----------------------------------------------------------------------------------
    "exp34": """
    **Способ предобработки наименований школ**: простая предобработка - удаление 
    служебных символов, пунктуации, отдельных букв, знаков номера, лишних пробелов, 
    пробелов в начале и в конце, замена "ё" на "е". Наименования предобработаны 
    одинаково и для рефенеса (эталонные названия школ) и для валидационной выборки.

    **Регионы**: не учитывались.

    **Способ векторизациии наименований школ**: использовались трансформеры, 
    модель - bert-base-uncased.

    **Способ сопоставления наименований школ**: косинусное расстояние.
    """,
    # ----------------------------------------------------------------------------------
    "exp35": """
    **Способ предобработки наименований школ**: простая предобработка - удаление 
    служебных символов, пунктуации, отдельных букв, знаков номера, лишних пробелов, 
    пробелов в начале и в конце, замена "ё" на "е". Наименования предобработаны 
    одинаково и для рефенеса (эталонные названия школ) и для валидационной выборки.

    **Регионы**: не учитывались.

    **Способ векторизациии наименований школ**: использовались трансформеры, 
    модель - DeepPavlov/rubert-base-cased.

    **Способ сопоставления наименований школ**: косинусное расстояние.

    <span style="background-color: yellow">**Замечания**: Необходимо провести 
    Fine Tuning модели для получения конкурентоспособных результатов.</span>
    """,
    # ----------------------------------------------------------------------------------
    "exp36": """
    **Способ предобработки наименований школ**: простая предобработка - удаление 
    служебных символов, пунктуации, отдельных букв, знаков номера, лишних пробелов, 
    пробелов в начале и в конце, замена "ё" на "е". Наименования предобработаны 
    одинаково и для рефенеса (эталонные названия школ) и для валидационной выборки.

    **Регионы**: не учитывались.

    **Способ векторизациии наименований школ**: TF-IDF.

    **Способ сопоставления наименований школ**: обучена модель классификации совпадений 
    с использованием CatBoost.

    <span style="background-color: yellow">**Замечания**: Сам CatBoost дает точность 
    от 0,81 при предсказании совпадений, однако при перемешивании различных 
    названий школ - ошибается. В итоге общая ошибка растет. Необходимы дополнительные
    экспертименты, поскольку направление работы считаю перспективным.</span>
    """,
    # ----------------------------------------------------------------------------------
    "exp37": """
    **Способ предобработки наименований школ**: расширенная предобработка - 
    работа с аббревиатурами: удалены ВСЕ аббревиатуры; простая предобработка - удаление 
    служебных символов, пунктуации, отдельных букв, знаков номера, лишних пробелов, 
    пробелов в начале и в конце, замена "ё" на "е". Наименования предобработаны 
    одинаково и для рефенеса (эталонные названия школ) и для валидационной выборки.

    **Регионы**: не учитывались.

    **Способ векторизациии наименований школ**: TF-IDF.

    **Способ сопоставления наименований школ**: косинусное расстояние.
    """,
    # ----------------------------------------------------------------------------------
    "exp38": """
    **Способ предобработки наименований школ**: расширенная предобработка - 
    работа с аббревиатурами: расшифрованы аббревиатуры, неизвестные аббревиатуры 
    удалены; простая предобработка - удаление служебных символов, пунктуации, 
    отдельных букв, знаков номера, лишних пробелов, пробелов в начале и в конце, 
    замена "ё" на "е". Наименования предобработаны одинаково и для рефенеса 
    (эталонные названия школ) и для валидационной выборки.

    **Регионы**: не учитывались.

    **Способ векторизациии наименований школ**: TF-IDF.

    **Способ сопоставления наименований школ**: косинусное расстояние.
    """,
    # ----------------------------------------------------------------------------------
    "exp39": """

     <span style="background-color: yellow">**Замечание**: Метрика резко увеличилась
    после предобработки тренировочных данных (удалены ошибочные записки) и референса
    (найдены 2 новых дубликата).</span>
       
    **Способ предобработки наименований школ**: расширенная предобработка - 
    работа с аббревиатурами: расшифрованы аббревиатуры, неизвестные аббревиатуры 
    удалены; простая предобработка - удаление служебных символов, пунктуации, 
    отдельных букв, знаков номера, лишних пробелов, пробелов в начале и в конце, 
    замена "ё" на "е". Наименования предобработаны одинаково и для рефенеса 
    (эталонные названия школ) и для валидационной выборки.

    **Регионы**: выделены с использованием списка регионов.

    **Способ векторизациии наименований школ**: TF-IDF.

    **Способ сопоставления наименований школ**: косинусное расстояние.

    <span style="background-color: yellow">**Замечания**: Планируется поиск 
    регионов с использованием регулярных выражений, а также обучение Named Entity 
    Recognition (распознавание именованных сущностей).</span>
    """,
    # ----------------------------------------------------------------------------------
    "exp42": """

    <span style="background-color: yellow">**Замечание**: Метрика резко увеличилась
    после предобработки тренировочных данных (удалены ошибочные записки) и референса
    (найдены 2 новых дубликата).</span>

    **Способ предобработки наименований школ**: расширенная предобработка - 
    работа с аббревиатурами: расшифрованы аббревиатуры, неизвестные аббревиатуры 
    удалены; все ОПФ - удалены, простая предобработка - удаление служебных символов, 
    пунктуации, отдельных букв, знаков номера, лишних пробелов, пробелов в начале 
    и в конце, замена "ё" на "е". Наименования предобработаны одинаково и для рефенеса 
    (эталонные названия школ) и для валидационной выборки.

    **Регионы**: выделены с использованием списка регионов.

    **Способ векторизациии наименований школ**: TF-IDF.

    **Способ сопоставления наименований школ**: косинусное расстояние.
    """,
    # ----------------------------------------------------------------------------------
}
