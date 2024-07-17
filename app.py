import importlib

import streamlit as st

from utils.load_functions import load_evaluations, load_plots
from utils.model_info import model_full_descriptions, model_name_map, model_options
from utils.plot_functions import plot_metrics

# Настройка страницы
st.set_page_config(layout="wide")

# Интерфейс Streamlit
st.title(
    "Текущие результаты разработки модели "
    "для распознавания наименований спортивных школ"
)

# Выбор модели
selected_description = st.selectbox("Выберите модель для распознавания", model_options)
model_name = model_name_map[selected_description]

# Загрузка и отображение оценок качества
if model_name:
    evaluations = load_evaluations(model_name)
    plots = load_plots(model_name)

    # Создание двух столбцов
    col1, col2 = st.columns(2)

    with col1:
        # Секция отображения метрик модели
        st.subheader("Ключевые показатели качества выбранной модели")
        metric_description = {
            "Accuracy@1": "Доля правильно распознаваемых объектов:",
            "Accuracy@3": "Доля объектов, правильно распознаваемых в топ 3:",
            "Accuracy@5": "Доля объектов, правильно распознаваемых в топ 5:",
            "Accuracy@10": "Доля объектов, правильно распознаваемых в топ 10:",
            "auto_error_rate": "Ошибка автоматического распознавания:",
            "manual_processing_rate": "Доля объектов, направляемых на ручное распознавание:",
            "general_error": "Итоговая ошибка (после автоматического и ручного распознавания):",
        }

        st.write(
            f"{metric_description['Accuracy@1']} "
            f"**{evaluations['Accuracy@1'] * 100:.2f}%**"
        )
        st.write(
            f"{metric_description['Accuracy@3']} "
            f"**{evaluations['Accuracy@3'] * 100:.2f}%**"
        )
        st.write(
            f"{metric_description['auto_error_rate']} "
            f"**{evaluations['auto_error_rate'] * 100:.2f}%**"
        )

        # for metric, value in evaluations.items():
        #     st.write(f"{metric_description[metric]} **{value * 100:.2f}%**")

        # Работа с пользовательским вводом
        st.subheader("Проверка работы модели")
        school_name = st.text_input("Введите название школы")

        # Кнопка для предсказания
        if st.button("Распознать наименование"):
            if school_name:
                # Динамическая загрузка файла инференса
                inference_module = importlib.import_module(
                    f"inferences.{model_name}_inference"
                )
                prediction = inference_module.predict([school_name])
                st.subheader("Найденная школа (наиболее вероятное совпадение)")
                st.write(prediction[0])
                st.write(prediction[1])
                # Отображение списка кортежей в свернутом виде
                if prediction[1]:
                    with st.expander(
                        "Нажмите, чтобы показать "
                        "топ-5 результатов (id и вероятность совпадения)"
                    ):
                        for item in prediction[2][0]:
                            st.write(item)
            else:
                st.write("Пожалуйста, введите название школы")

    with col2:
        # Секция отображения описания модели
        st.subheader("Описание используемых подходов")
        st.markdown(
            model_full_descriptions.get(
                model_name, "Описание для этой модели недоступно"
            ),
            unsafe_allow_html=True,
        )

        # Секция отображения графика
        st.subheader("Зависимость ошибок от доли ручного распознавания")
        if plots:
            plot_metrics(plots)
