import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib import ticker
from matplotlib.ticker import PercentFormatter


def plot_metrics(plots):
    # Установка палитры для дальтоников
    sns.set_palette("colorblind")

    # Построение графика
    fig, ax = plt.subplots()

    # График служебный
    # ax.plot(
    #     plots["threshold_values"],
    #     plots["auto_error_rate_values"],
    #     label="Ошибка автоматического распознавания",
    #     linestyle="-",
    #     linewidth=1,
    # )
    # ax.plot(
    #     plots["threshold_values"],
    #     plots["manual_processing_rate_values"],
    #     label="Доля ручного распознавания",
    #     linestyle="--",
    #     linewidth=1,
    # )
    # ax.plot(
    #     plots["threshold_values"],
    #     plots["general_error_values"],
    #     label="Итоговая ошибка",
    #     linestyle=":",
    #     linewidth=1,
    # )
    # ax.set_xlabel("Порог срабатывания")
    # ax.set_ylabel("Доля")
    # ax.set_title("Зависимости ошибок и доли ручного распознавания")
    # ax.legend()

    # # Настройка основных и второстепенных тиков
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))

    # # Добавление сетки для основных и второстепенных тиков с прозрачностью
    # ax.grid(which="major", linestyle="-", linewidth="0.3", color="black", alpha=0.2)
    # ax.grid(which="minor", linestyle=":", linewidth="0.3", color="gray", alpha=0.2)

    # Построение графика для Заказчика

    ax.plot(
        plots["manual_processing_rate_values"],
        plots["auto_error_rate_values"],
        label="Ошибка автоматического распознавания",
        linestyle=":",
    )
    ax.plot(
        plots["manual_processing_rate_values"],
        plots["general_error_values"],
        label="Общая ошибка",
        linestyle="-",
    )
    ax.set_xlabel("Доля ручного распознавания")
    ax.set_ylabel("Ошибка")
    # ax.set_title("Зависимости ошибки")
    ax.legend()

    # Установка формата осей в процентах
    ax.xaxis.set_major_formatter(PercentFormatter(1))

    # Настройка основных и второстепенных тиков
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))

    # Добавление сетки для основных и второстепенных тиков с прозрачностью
    ax.grid(which="major", linestyle="-", linewidth="0.3", color="black", alpha=0.2)
    ax.grid(which="minor", linestyle=":", linewidth="0.3", color="gray", alpha=0.2)

    # Установка максимального значения оси y
    ax.set_ylim(0, 1.0)

    # Отображение графика в Streamlit
    st.pyplot(fig)
