# -*- coding: utf-8 -*-
"""
Модифицированная программа обработки изображений с разбиением на функции,
добавлением логирования, сохранением результатов и специальной обработкой для картины Дали
"""

import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import logging
import sys
import os
import time
from pathlib import Path
import io


# Настройка логирования
def setup_logging():
    """Настройка системы логирования с поддержкой Unicode"""

    # Создаем базовый логгер
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Формат логов
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    # Обработчик для записи в файл
    file_handler = logging.FileHandler("image_processing.log", encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Проверяем, запущен ли скрипт в Windows
    if os.name == 'nt':
        # В Windows необходимо использовать правильную кодировку для консоли
        try:
            # Для поддержки Unicode в консоли Windows
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

            # Настройка вывода с поддержкой Unicode
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')

            # Создаем обработчик для консоли Windows
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        except Exception as e:
            # Если не удалось настроить Unicode для консоли, используем ASCII
            # и добавляем предупреждение в лог-файл
            file_handler.setLevel(logging.INFO)
            logger.warning(f"Не удалось настроить Unicode для консоли: {e}")
            logger.warning("Логи будут записываться только в файл image_processing.log")
    else:
        # Для других ОС (Linux, macOS) обычно не требуется специальная настройка
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def load_image(image_path, logger):
    """Загрузка изображения с диска"""
    logger.info(f"Загрузка изображения из {image_path}")
    try:
        # Проверяем существование файла
        if not os.path.exists(image_path):
            logger.error(f"Файл {image_path} не существует")
            return None

        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Не удалось загрузить изображение из {image_path}")
            return None
        logger.info(f"Изображение загружено успешно, размер: {image.shape}")
        return image
    except Exception as e:
        logger.error(f"Ошибка при загрузке изображения: {e}")
        return None


def resize_image(image, width, height, logger):
    """Изменение размера изображения"""
    logger.info(f"Изменение размера изображения на {width}x{height}")
    try:
        resized = cv2.resize(image, (width, height))
        logger.info(f"Размер изображения изменен успешно, новый размер: {resized.shape}")
        return resized
    except Exception as e:
        logger.error(f"Ошибка при изменении размера изображения: {e}")
        return image


def convert_to_grayscale(image, logger):
    """Конвертация изображения в оттенки серого"""
    logger.info("Конвертация изображения в оттенки серого")
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.info("Изображение успешно конвертировано в оттенки серого")
        return gray
    except Exception as e:
        logger.error(f"Ошибка при конвертации в оттенки серого: {e}")
        return None


def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0, logger=None):
    """Применение гауссова размытия к изображению"""
    if logger:
        logger.info(f"Применение гауссова размытия с размером ядра {kernel_size}")
    try:
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        if logger:
            logger.info("Гауссово размытие применено успешно")
        return blurred
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при применении гауссова размытия: {e}")
        return image


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8), logger=None):
    """Применение адаптивного выравнивания гистограммы (CLAHE)"""
    if logger:
        logger.info(f"Применение CLAHE с clip_limit={clip_limit}, tile_grid_size={tile_grid_size}")
    try:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        equalized = clahe.apply(image)
        if logger:
            logger.info("CLAHE применено успешно")
        return equalized
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при применении CLAHE: {e}")
        return image


def apply_morphology(image, operation, kernel_size=(3, 3), iterations=1, logger=None):
    """Применение морфологических операций к изображению"""
    if logger:
        logger.info(f"Применение морфологической операции {operation} с размером ядра {kernel_size}")
    try:
        kernel = np.ones(kernel_size, np.uint8)
        result = cv2.morphologyEx(image, operation, kernel, iterations=iterations)
        if logger:
            logger.info("Морфологическая операция применена успешно")
        return result
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при применении морфологической операции: {e}")
        return image


def detect_edges_canny(image, low_threshold=30, high_threshold=200, logger=None):
    """Обнаружение краев с помощью алгоритма Canny"""
    if logger:
        logger.info(f"Обнаружение краев с помощью Canny с порогами {low_threshold} и {high_threshold}")
    try:
        edges = cv2.Canny(image, low_threshold, high_threshold)
        if logger:
            logger.info("Края успешно обнаружены с помощью Canny")
        return edges
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при обнаружении краев с помощью Canny: {e}")
        return None


def detect_edges_laplacian(image, ksize=3, logger=None):
    """Обнаружение краев с помощью оператора Лапласа"""
    if logger:
        logger.info(f"Обнаружение краев с помощью оператора Лапласа с размером ядра {ksize}")
    try:
        edges = cv2.Laplacian(image, cv2.CV_8U, ksize=ksize)
        if logger:
            logger.info("Края успешно обнаружены с помощью оператора Лапласа")
        return edges
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при обнаружении краев с помощью оператора Лапласа: {e}")
        return None


def combine_edges(edges1, edges2, logger=None):
    """Комбинирование результатов обнаружения краев"""
    if logger:
        logger.info("Комбинирование результатов обнаружения краев")
    try:
        combined = cv2.bitwise_or(edges1, edges2)
        if logger:
            logger.info("Края успешно скомбинированы")
        return combined
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при комбинировании краев: {e}")
        return edges1  # Возвращаем первый вариант в случае ошибки


def find_contours(edges, retr_mode=cv2.RETR_LIST, approx_method=cv2.CHAIN_APPROX_SIMPLE, logger=None):
    """Поиск контуров на изображении"""
    if logger:
        logger.info("Поиск контуров на изображении")
    try:
        start_time = time.time()
        contours = cv2.findContours(edges.copy(), retr_mode, approx_method)[0]
        elapsed_time = time.time() - start_time
        if logger:
            logger.info(f"Найдено {len(contours)} контуров за {elapsed_time:.2f} секунд")
        return contours
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при поиске контуров: {e}")
        return []


def find_quadrilateral(contours, logger=None):
    """Поиск четырехугольника среди контуров"""
    if logger:
        logger.info("Поиск четырехугольника среди контуров")
    try:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        solution = None
        for c in sorted_contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                solution = approx
                if logger:
                    logger.info("Четырехугольник найден")
                break
        if solution is None and logger:
            logger.warning("Четырехугольник не найден среди контуров")
        return solution
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при поиске четырехугольника: {e}")
        return None


def find_circular_contour(contours, circularity_threshold=0.4, min_radius=5, max_radius=200, min_area=50, logger=None):
    """
    Поиск всех контуров, близких к кругу, по критерию площади с дополнительной фильтрацией.

    Args:
        contours: Список контуров
        circularity_threshold: Порог округлости (0.0-1.0), ниже которого не считаем круглым
        min_radius: Минимальный радиус для фильтрации маленьких контуров
        max_radius: Максимальный радиус для фильтрации больших контуров
        min_area: Минимальная площадь контура
        logger: Объект логгера

    Returns:
        Список круглых контуров
    """
    if logger:
        logger.info(f"Поиск круглых контуров с порогом округлости {circularity_threshold}")

    try:
        circular_contours = []
        all_circularities = []  # Для анализа распределения округлостей

        for i, c in enumerate(contours):
            # Фильтрация по площади
            contour_area = cv2.contourArea(c)
            if contour_area < min_area:
                continue

            # Вычисление округлости
            (x, y), radius = cv2.minEnclosingCircle(c)

            # Фильтрация по радиусу
            if radius < min_radius or radius > max_radius:
                continue

            # Избегаем деления на ноль
            circle_area = np.pi * (radius ** 2)
            if circle_area == 0:
                continue

            circularity = contour_area / circle_area
            all_circularities.append(circularity)

            if logger:
                logger.debug(
                    f"Контур #{i}: площадь={contour_area:.2f}, радиус={radius:.2f}, округлость={circularity:.3f}")

            # Фильтрация по округлости
            if circularity > circularity_threshold:
                circular_contours.append((c, circularity, radius))
                if logger:
                    logger.info(f"Найден круглый контур #{i} с округлостью {circularity:.3f}")

        # Анализ распределения округлостей для отладки
        if all_circularities and logger:
            min_circ = min(all_circularities)
            max_circ = max(all_circularities)
            avg_circ = sum(all_circularities) / len(all_circularities)
            logger.info(f"Статистика округлостей: мин={min_circ:.3f}, макс={max_circ:.3f}, среднее={avg_circ:.3f}")

        # Сортируем по округлости (от большей к меньшей)
        circular_contours.sort(key=lambda x: x[1], reverse=True)

        # Извлекаем только контуры
        circular_contours = [c[0] for c in circular_contours]

        if not circular_contours and logger:
            logger.warning("Круглые контуры не найдены")
        else:
            logger.info(f"Всего найдено {len(circular_contours)} круглых контуров")

        return circular_contours
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при поиске круглых контуров: {e}")
        return []


def detect_circles_hough(gray_image, dp=1, min_dist=20, param1=50, param2=30, min_radius=10, max_radius=100,
                         logger=None):
    """
    Обнаружение кругов на изображении с помощью преобразования Хафа.

    Args:
        gray_image: Изображение в оттенках серого
        dp: Параметр разрешения обратно пропорционален соотношению аккумулятора
        min_dist: Минимальное расстояние между центрами обнаруженных кругов
        param1: Верхний порог для детектора Canny
        param2: Порог для обнаружения центров
        min_radius: Минимальный радиус круга
        max_radius: Максимальный радиус круга
        logger: Объект логгера

    Returns:
        Массив обнаруженных кругов в формате [x, y, radius]
    """
    if logger:
        logger.info(f"Поиск кругов с помощью преобразования Хафа")

    try:
        # Применяем размытие для уменьшения шума
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Обнаружение кругов с помощью преобразования Хафа
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            if logger:
                logger.info(f"Найдено {len(circles)} кругов с помощью преобразования Хафа")
            return circles
        else:
            if logger:
                logger.warning("Круги не обнаружены с помощью преобразования Хафа")
            return []
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при поиске кругов с помощью преобразования Хафа: {e}")
        return []


def draw_contour(image, contour, color=(0, 255, 0), thickness=2, logger=None):
    """Отрисовка контура на изображении"""
    if contour is None:
        if logger:
            logger.warning("Попытка отрисовки пустого контура")
        return image

    if logger:
        logger.info(f"Отрисовка контура с цветом {color} и толщиной {thickness}")
    try:
        image_copy = image.copy()
        cv2.drawContours(image_copy, [contour], -1, color, thickness)
        if logger:
            logger.info("Контур успешно отрисован")
        return image_copy
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при отрисовке контура: {e}")
        return image


def draw_contours(image, contours, color=(0, 255, 0), thickness=2, draw_numbers=True, number_color=(255, 0, 0),
                  logger=None):
    """Отрисовка множества контуров на изображении с возможной нумерацией"""
    if not contours:
        if logger:
            logger.warning("Попытка отрисовки пустого списка контуров")
        return image

    if logger:
        logger.info(f"Отрисовка {len(contours)} контуров с цветом {color} и толщиной {thickness}")
    try:
        image_copy = image.copy()
        for i, contour in enumerate(contours):
            cv2.drawContours(image_copy, [contour], -1, color, thickness)

            # Если нужно нумеровать контуры
            if draw_numbers:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(
                        image_copy, str(i + 1), (cX - 10, cY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, number_color, 1
                    )

        if logger:
            logger.info("Контуры успешно отрисованы")
        return image_copy
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при отрисовке контуров: {e}")
        return image


def draw_hough_circles(image, circles, color=(0, 255, 0), center_color=(0, 0, 255), thickness=2, draw_numbers=True,
                       logger=None):
    """Отрисовка кругов, найденных с помощью преобразования Хафа"""
    if not circles:
        if logger:
            logger.warning("Попытка отрисовки пустого списка кругов")
        return image

    if logger:
        logger.info(f"Отрисовка {len(circles)} кругов, найденных с помощью преобразования Хафа")
    try:
        image_copy = image.copy()
        for i, (x, y, r) in enumerate(circles):
            # Рисуем окружность
            cv2.circle(image_copy, (x, y), r, color, thickness)
            # Рисуем центр
            cv2.circle(image_copy, (x, y), 2, center_color, 3)
            # Добавляем номер
            if draw_numbers:
                cv2.putText(
                    image_copy, str(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
                )

        if logger:
            logger.info("Круги успешно отрисованы")
        return image_copy
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при отрисовке кругов: {e}")
        return image


def display_image(image, title="Image", cmap=None, convert_color=True, logger=None):
    """Отображение изображения с помощью matplotlib"""
    if logger:
        logger.info(f"Отображение изображения с заголовком '{title}'")
    try:
        plt.figure(figsize=(10, 8))
        if convert_color and len(image.shape) == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        plt.show()
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при отображении изображения: {e}")


def save_image(image, output_path, convert_color=False, logger=None):
    """Сохранение изображения на диск"""
    if logger:
        logger.info(f"Сохранение изображения в {output_path}")
    try:
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Для отображения с помощью matplotlib нужно конвертировать обратно в BGR
        if convert_color and len(image.shape) == 3:
            image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_to_save = image

        success = cv2.imwrite(output_path, image_to_save)
        if success:
            if logger:
                logger.info(f"Изображение успешно сохранено в {output_path}")
            return True
        else:
            if logger:
                logger.error(f"Не удалось сохранить изображение в {output_path}")
            return False
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при сохранении изображения: {e}")
        return False


def process_dali_image(image_path, output_dir=None, logger=None):
    """
    Обработка изображения картины Дали с выделением сфер.

    Args:
        image_path: Путь к исходному изображению
        output_dir: Директория для сохранения результатов (если None, используется директория исходного файла)
        logger: Объект логгера

    Returns:
        Словарь с обработанными изображениями
    """
    if logger:
        logger.info(f"Обработка изображения {image_path}")

    # Определяем путь к файлу и директорию для сохранения
    path_obj = Path(image_path)
    if output_dir is None:
        output_dir = path_obj.parent
    file_name = path_obj.stem
    file_ext = path_obj.suffix

    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        if logger:
            logger.error(f"Не удалось загрузить изображение {image_path}")
        return None

    # Сохраняем исходный размер для отображения результатов
    original_size = image.shape[:2][::-1]  # (width, height)

    # Изменение размера для ускорения обработки
    max_size = 800
    if max(original_size) > max_size:
        scale = max_size / max(original_size)
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        image = cv2.resize(image, new_size)
        if logger:
            logger.info(f"Изображение уменьшено до {new_size}")

    # Конвертация в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Создаем CLAHE объект для адаптивного выравнивания гистограммы
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_equalized = clahe.apply(gray)

    # Размытие для уменьшения шума
    blurred = cv2.GaussianBlur(gray_equalized, (5, 5), 0)

    # Адаптивная бинаризация для лучшего выделения контуров
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Морфологические операции для улучшения контуров
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Обнаружение краев различными методами
    edges_canny = cv2.Canny(blurred, 30, 150)
    edges_laplacian = cv2.Laplacian(blurred, cv2.CV_8U)

    # Комбинируем несколько подходов к обнаружению краев
    edges_combined = cv2.bitwise_or(edges_canny, edges_laplacian)

    # Находим контуры
    contours, _ = cv2.findContours(
        edges_combined.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if logger:
        logger.info(f"Найдено {len(contours)} контуров")

    # Обнаружение кругов с помощью метода на основе контуров
    # Настройка параметров специально для картины Дали:
    # - Снижаем порог округлости, так как сферы на картине не идеально круглые
    # - Увеличиваем диапазон допустимых радиусов
    circular_contours = find_circular_contour(
        contours,
        circularity_threshold=0.4,  # Снижаем порог округлости для сфер Дали
        min_radius=5,
        max_radius=200,
        min_area=50,
        logger=logger
    )

    # Для сложных изображений как у Дали, дополнительно используем преобразование Хафа
    hough_circles = detect_circles_hough(
        gray,
        dp=1,
        min_dist=20,
        param1=50,
        param2=30,
        min_radius=10,
        max_radius=100,
        logger=logger
    )

    # Создаем различные выходные изображения для анализа
    results = {}

    # Оригинальное изображение
    results['original'] = image.copy()

    # Обработанное изображение краев
    results['edges'] = cv2.cvtColor(edges_combined, cv2.COLOR_GRAY2BGR)

    # Изображение с контурами
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)
    results['contours'] = contour_image

    # Изображение с выделенными кругами (метод контуров)
    circles_image = image.copy()
    for i, contour in enumerate(circular_contours):
        # Рисуем контур
        cv2.drawContours(circles_image, [contour], -1, (0, 255, 0), 2)

        # Вычисляем и рисуем центр и номер
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(circles_image, (cX, cY), 3, (255, 0, 0), -1)
            cv2.putText(
                circles_image, str(i + 1), (cX - 10, cY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )
    results['circles_contour'] = circles_image

    # Изображение с выделенными кругами (метод Хафа)
    hough_image = image.copy()
    for i, (x, y, r) in enumerate(hough_circles):
        # Рисуем окружность
        cv2.circle(hough_image, (x, y), r, (0, 255, 0), 2)
        # Рисуем центр
        cv2.circle(hough_image, (x, y), 2, (0, 0, 255), 3)
        # Добавляем номер
        cv2.putText(
            hough_image, str(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        )
    results['circles_hough'] = hough_image


    # Комбинированный результат (объединяем оба метода)
    combined_image = image.copy()

    # Добавляем обнаруженные круги методом контуров
    for contour in circular_contours:
        cv2.drawContours(combined_image, [contour], -1, (0, 255, 0), 2)

    # Добавляем обнаруженные круги методом Хафа (другим цветом)
    for (x, y, r) in hough_circles:
        cv2.circle(combined_image, (x, y), r, (255, 0, 0), 2)

    results['combined'] = combined_image

    # Сохраняем результаты
    if logger:
        logger.info("Сохранение результатов")

    saved_paths = {}
    for key, img in results.items():
        output_path = os.path.join(output_dir, f"{file_name}_{key}{file_ext}")
        success = cv2.imwrite(output_path, img)
        if success:
            saved_paths[key] = output_path
            if logger:
                logger.info(f"Сохранено изображение {key} в {output_path}")
        else:
            if logger:
                logger.error(f"Не удалось сохранить изображение {key}")

    return results, saved_paths


def main():
    """Основная функция программы"""
    # Настройка логирования
    logger = setup_logging()
    logger.info("Запуск программы обработки изображения")

    # Настройка matplotlib
    matplotlib.rcParams['figure.figsize'] = (20, 10)

    # Путь к изображению
    image_path = 'content/Dali_Galateya.jpg'

    # Получаем путь и имя файла
    path_obj = Path(image_path)
    file_dir = path_obj.parent
    file_name = path_obj.stem  # имя файла без расширения
    file_ext = path_obj.suffix  # расширение файла

    logger.info(f"Обрабатываем файл {file_name}{file_ext} из директории {file_dir}")

    # Обработка изображения специализированной функцией для Дали
    results, saved_paths = process_dali_image(image_path, logger=logger)

    if results:
        logger.info("Обработка завершена успешно")

        # Отображаем ключевые результаты
        display_image(results['original'], title="Исходное изображение", logger=logger)
        display_image(results['edges'], title="Обнаруженные края", logger=logger)
        display_image(results['circles_contour'], title="Круги (метод контуров)", logger=logger)
        display_image(results['circles_hough'], title="Круги (метод Хафа)", logger=logger)
        display_image(results['combined'], title="Комбинированный результат", logger=logger)
    else:
        logger.error("Обработка завершилась с ошибкой")

    logger.info("Программа успешно завершена")


# Альтернативная функция main() для общего случая (не только для Дали)
def main_general():
    """Основная функция программы для общего случая"""
    # Настройка логирования
    logger = setup_logging()
    logger.info("Запуск программы обработки изображения")

    # Настройка matplotlib
    matplotlib.rcParams['figure.figsize'] = (20, 10)

    # Путь к изображению
    image_path = 'content/Dali_Galateya.jpg'  # Замените на путь к вашему изображению

    # Получаем путь и имя файла
    path_obj = Path(image_path)
    file_dir = path_obj.parent
    file_name = path_obj.stem  # имя файла без расширения
    file_ext = path_obj.suffix  # расширение файла

    logger.info(f"Обрабатываем файл {file_name}{file_ext} из директории {file_dir}")

    # Загрузка и обработка изображения
    original_image = load_image(image_path, logger)
    if original_image is None:
        logger.error("Завершение программы из-за ошибки загрузки изображения")
        return

    # Изменение размера для ускорения обработки
    max_dimension = max(original_image.shape[0], original_image.shape[1])
    if max_dimension > 800:
        scale = 800 / max_dimension
        new_width = int(original_image.shape[1] * scale)
        new_height = int(original_image.shape[0] * scale)
        image = resize_image(original_image, new_width, new_height, logger)
    else:
        image = original_image.copy()

    display_image(image, "Исходное изображение", logger=logger)

    # Словарь для хранения всех изображений и их названий для сохранения
    images_to_save = {}
    images_to_save["original"] = (image, "Исходное изображение")

    # Конвертация в оттенки серого
    gray = convert_to_grayscale(image, logger)
    if gray is None:
        logger.error("Завершение программы из-за ошибки конвертации в оттенки серого")
        return

    # Улучшение контраста с помощью CLAHE
    gray_equalized = apply_clahe(gray, clip_limit=2.0, tile_grid_size=(8, 8), logger=logger)

    display_image(gray_equalized, "Изображение в оттенках серого с CLAHE", cmap='gray', convert_color=False,
                  logger=logger)
    images_to_save["gray"] = (gray_equalized, "Изображение в оттенках серого с CLAHE")

    # Применение размытия
    gray_blurred = apply_gaussian_blur(gray_equalized, (5, 5), 0, logger)
    display_image(gray_blurred, "Размытое изображение", cmap='gray', convert_color=False, logger=logger)
    images_to_save["blurred"] = (gray_blurred, "Размытое изображение")

    # Обнаружение краев несколькими методами
    edges_canny = detect_edges_canny(gray_blurred, 30, 150, logger)
    edges_laplacian = detect_edges_laplacian(gray_blurred, 3, logger)

    # Комбинирование результатов
    edges_combined = combine_edges(edges_canny, edges_laplacian, logger)

    display_image(edges_combined, "Обнаруженные края (комбинированные)", cmap='gray', convert_color=False,
                  logger=logger)
    images_to_save["edges"] = (edges_combined, "Обнаруженные края (комбинированные)")

    # Улучшение краев с помощью морфологических операций
    edges_enhanced = apply_morphology(edges_combined, cv2.MORPH_CLOSE, kernel_size=(3, 3), iterations=1, logger=logger)
    display_image(edges_enhanced, "Улучшенные края", cmap='gray', convert_color=False, logger=logger)
    images_to_save["edges_enhanced"] = (edges_enhanced, "Улучшенные края")

    # Поиск контуров
    contours = find_contours(edges_enhanced, logger=logger)
    if not contours:
        logger.error("Завершение программы из-за отсутствия контуров")
        return

    # Рисуем все контуры
    all_contours_image = image.copy()
    cv2.drawContours(all_contours_image, contours, -1, (0, 255, 0), 1)
    display_image(all_contours_image, "Все контуры", logger=logger)
    images_to_save["all_contours"] = (all_contours_image, "Все контуры")

    # Поиск кругов с помощью двух методов

    # 1. Метод на основе контуров с оптимизированными параметрами
    circular_contours = find_circular_contour(
        contours,
        circularity_threshold=0.4,  # Более низкий порог для нечетких кругов
        min_radius=5,
        max_radius=150,
        min_area=50,
        logger=logger
    )

    if circular_contours:
        contour_circles_image = draw_contours(
            image.copy(),
            circular_contours,
            color=(0, 255, 0),
            thickness=2,
            draw_numbers=True,
            logger=logger
        )
        display_image(contour_circles_image, f"Круги (метод контуров): {len(circular_contours)}", logger=logger)
        images_to_save["circles_contour"] = (contour_circles_image, f"Круги (метод контуров): {len(circular_contours)}")

    # 2. Метод Хафа
    hough_circles = detect_circles_hough(
        gray_equalized,
        dp=1,
        min_dist=20,
        param1=50,
        param2=30,
        min_radius=10,
        max_radius=100,
        logger=logger
    )

    if hough_circles:
        hough_circles_image = draw_hough_circles(
            image.copy(),
            hough_circles,
            color=(255, 0, 0),
            center_color=(0, 0, 255),
            thickness=2,
            draw_numbers=True,
            logger=logger
        )
        display_image(hough_circles_image, f"Круги (метод Хафа): {len(hough_circles)}", logger=logger)
        images_to_save["circles_hough"] = (hough_circles_image, f"Круги (метод Хафа): {len(hough_circles)}")

    # Комбинированный результат
    if circular_contours or hough_circles:
        combined_image = image.copy()

        # Добавляем обнаруженные круги методом контуров
        if circular_contours:
            for contour in circular_contours:
                cv2.drawContours(combined_image, [contour], -1, (0, 255, 0), 2)

        # Добавляем обнаруженные круги методом Хафа (другим цветом)
        if hough_circles:
            for (x, y, r) in hough_circles:
                cv2.circle(combined_image, (x, y), r, (255, 0, 0), 2)

        display_image(combined_image, "Комбинированный результат", logger=logger)
        images_to_save["combined"] = (combined_image, "Комбинированный результат")

    # Сохранение всех обработанных изображений
    logger.info("Сохранение обработанных изображений")

    for key, (img, description) in images_to_save.items():
        # Создаем имя файла: [исходное_имя]_[тип_обработки].jpg
        output_filename = f"{file_name}_{key}{file_ext}"
        output_path = os.path.join(file_dir, output_filename)

        # Сохраняем изображение
        save_success = save_image(img, output_path, logger=logger)
        if save_success:
            logger.info(f"Изображение '{description}' сохранено как {output_filename}")
        else:
            logger.error(f"Не удалось сохранить изображение '{description}'")

    logger.info("Программа успешно завершена")


if __name__ == "__main__":
    # Выберите нужную функцию main в зависимости от типа изображения
    # Для картины Дали:
    main()

    # Для общего случая:
    # main_general()