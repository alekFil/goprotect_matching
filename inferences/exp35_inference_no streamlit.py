# from utils import metrics as mtc

import os
import sys

import numpy as np
import torch
from fuzzywuzzy import fuzz  # Добавлено для fuzzywuzzy
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

# Добавляем корневую директорию проекта в sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))

from sklearn.metrics.pairwise import cosine_similarity

from utils.load_functions import load_resources
from utils.preprocess_functions import replace_numbers_with_text, simple_preprocess_text

model_name = "exp35"

TOKENIZER = load_resources(model_name, "tokenizer", "pt")
MODEL = load_resources(model_name, "model", "pt")
REFERENCE_VEC = load_resources(model_name, "reference_vec", "joblib")
REFERENCE_ID = load_resources(model_name, "reference_id", "joblib")
REFERENCE_REGION = load_resources(model_name, "reference_region", "joblib")
REFERENCE_NAME = load_resources(model_name, "reference_name", "joblib")


def calculate_similarity(x, y, method="cosine"):
    if method == "cosine":
        return cosine_similarity(x, y)
    elif method == "euclidean":
        return -euclidean_distances(x, y)  # Инвертируем, чтобы максимизировать схожесть
    elif method == "manhattan":
        return -manhattan_distances(x, y)  # Инвертируем, чтобы максимизировать схожесть
    elif method == "fuzzywuzzy":  # Добавлено для fuzzywuzzy
        return fuzz.ratio(x, y) / 100.0  # Нормализуем до диапазона [0, 1]
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def find_matches(
    x_vec,
    x_region,
    reference_id,
    reference_vec,
    reference_region,
    top_k=5,
    threshold=0.9,
    filter_by_region=True,
    empty_region="all",
    similarity_method="cosine",
):
    y_pred = []
    manual_review = []

    for i, x in enumerate(x_vec):
        # Фильтруем reference_vec и reference_id по текущему региону, если включена фильтрация по регионам
        if filter_by_region:
            # Фильтруем reference_vec и reference_id по текущему региону
            current_region = x_region[i]
            region_mask = reference_region == current_region
            filtered_reference_vec = reference_vec[region_mask]
            filtered_reference_id = reference_id[region_mask]

            # Способ обработки, если в текущем регионе нет школ для сравнения
            if empty_region == "all":
                # Если в текущем регионе нет школ для сравнения, используем все школы
                if filtered_reference_vec.shape[0] == 0:
                    filtered_reference_vec = reference_vec
                    filtered_reference_id = reference_id
            else:
                if filtered_reference_vec.shape[0] == 0:
                    # Если в текущем регионе нет школ для сравнения, то помечаем на ручную обработку
                    manual_review.append(x)
                    top_matches = [(None, 0.0)] * top_k
                    y_pred.append(top_matches)
                    continue
        else:
            filtered_reference_vec = reference_vec
            filtered_reference_id = reference_id

        if similarity_method == "fuzzywuzzy":  # Добавлено для fuzzywuzzy
            # Вычисляем сходство с помощью fuzzywuzzy
            similarities = [
                calculate_similarity(x, y, method=similarity_method)
                for y in filtered_reference_vec
            ]
        else:
            # Вычисляем выбранное расстояние
            similarities = calculate_similarity(
                x, filtered_reference_vec, method=similarity_method
            ).flatten()

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        max_similarity = max(similarities)

        # Учитываем пороговое значение для различных методов
        if similarity_method in ["cosine", "fuzzywuzzy"]:
            if max_similarity < threshold:
                manual_review.append(x)
                top_matches = [(None, 0.0)] * top_k
            else:
                top_matches = [
                    (filtered_reference_id[i], similarities[i]) for i in top_indices
                ]
                if len(top_matches) < top_k:
                    top_matches += [(None, 0.0)] * (top_k - len(top_matches))
        else:  # Для других методов расстояний (евклидово и манхэттенское)
            if max_similarity > -threshold:  # Обратим внимание на инверсию
                manual_review.append(x)
                top_matches = [(None, 0.0)] * top_k
            else:
                top_matches = [
                    (filtered_reference_id[i], -similarities[i]) for i in top_indices
                ]
                if len(top_matches) < top_k:
                    top_matches += [(None, 0.0)] * (top_k - len(top_matches))

        y_pred.append(top_matches)

    return y_pred, manual_review


def embed_texts(texts, tokenizer, model):
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    with torch.no_grad():
        outputs = model.bert(**inputs)
    return outputs[0][:, 0, :].numpy()  # Получение скрытых состояний


def predict(data):
    def preprocess(x):
        x = simple_preprocess_text(x)
        x = replace_numbers_with_text(x)
        return x

    vectorized_function = np.vectorize(preprocess)

    x = vectorized_function(np.array(data))

    # Векторизация текста
    x_vec = embed_texts(x.tolist(), TOKENIZER, MODEL)
    x_vec = x_vec.reshape(-1, 1, x_vec.shape[1])

    y_pred, manual_review = find_matches(
        x_vec,
        "санкт петербург",
        REFERENCE_ID,
        REFERENCE_VEC,
        REFERENCE_REGION,
        top_k=5,
        threshold=0.00001,
        filter_by_region=False,
        empty_region="all",  # is ignored if filter_by_region=False
        similarity_method="cosine",
    )

    if y_pred[0][0][0] is not None:
        ref_id = np.where(REFERENCE_ID == y_pred[0][0][0])[0][0]
        answer_id = y_pred[0][0][0]
        answer_name = REFERENCE_NAME[ref_id]
        answer_region = REFERENCE_REGION[ref_id]
        return (
            f"Имя в базе: **{answer_name}**, "
            f"регион - **{answer_region}**, "
            f"id - **{answer_id}**",
            y_pred,
        )
    else:
        answer_id = None
        answer_name = None
        answer_region = None
        return (
            "Школа не распознана, отправлена на ручное распознавание",
            y_pred,
        )
