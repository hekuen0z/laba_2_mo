import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from hashlib import sha1
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau


USE_SMALL_DATASET = False
if USE_SMALL_DATASET:
    DATA_DIR = 'notMNIST_small'
else:
    DATA_DIR = 'notMNIST_large'
LABELS = list('ABCDEFGHIJ')
IMAGE_SIZE = 28
NUM_FEATURES = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = len(LABELS)


def load_and_preprocess_data(root_dir, image_size=IMAGE_SIZE):
    X_all = []
    y_all = []

    print(f"Загрузка изображений и объединение данных из папки: {root_dir}...")

    label_to_index = {label: index for index, label in enumerate(LABELS)}
    total_images_loaded = 0

    for label in LABELS:
        label_index = label_to_index[label]
        class_dir = os.path.join(root_dir, label)

        if not os.path.isdir(class_dir):
            print(f"Внимание! Директория для класса '{label}' не найдена: {class_dir}. Пропуск.")
            continue

        print(f"Загрузка изображений для класса '{label}'...")

        for filename in tqdm(os.listdir(class_dir), desc=f"Класс {label}"):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, filename)

                try:
                    # Преобразование в оттенки серого с последующей нормализацией
                    img = Image.open(img_path).convert('L').resize((image_size, image_size))
                    img_array = np.array(img, dtype=np.float32)
                    normalized_data = (img_array - 128.0) / 128.0
                    X_all.append(normalized_data.flatten())
                    y_all.append(label_index)
                    total_images_loaded += 1

                except Exception as e:
                    continue

    X_all_combined = np.array(X_all)
    y_all_combined = np.array(y_all)

    permutation = np.random.permutation(X_all_combined.shape[0])
    X_all_combined = X_all_combined[permutation]
    y_all_combined = y_all_combined[permutation]

    print(f"Общее количество загруженных изображений: {X_all_combined.shape[0]}")
    return X_all_combined, y_all_combined


def load_and_display(X_combined, y_combined):
    # Обратное преобразование с плоского представления
    X_display = (X_combined * 128.0 + 128.0).astype(np.uint8).reshape(-1, 28, 28)

    num_to_show = 10
    fig, axes = plt.subplots(1, num_to_show, figsize=(15, 1.5))
    random_indices = np.random.choice(X_display.shape[0], num_to_show, replace=False)

    print(f"Отображение {num_to_show} случайных изображений:")
    for i, idx in enumerate(random_indices):
        ax = axes[i]
        ax.imshow(X_display[idx], cmap='gray')
        ax.set_title(LABELS[y_combined[idx]], fontsize=10)
        ax.axis('off')

    plt.suptitle("Примеры изображений из набора notMNIST", y=1.1)
    plt.tight_layout()
    plt.show()


def check_group_balance(y_combined):
    unique_labels, counts = np.unique(y_combined, return_counts=True)
    total_samples = len(y_combined)

    print("Распределение классов:")
    for label_index, count in zip(unique_labels, counts):
        percentage = (count / total_samples) * 100
        print(f"Класс '{LABELS[label_index]}' ({label_index}): {count} изображений ({percentage:.2f}%)")

    plt.figure(figsize=(10, 5))
    plt.bar([LABELS[i] for i in unique_labels], counts, color='skyblue')
    plt.title('Распределение изображений по классам')
    plt.xlabel('Класс (Буква)')
    plt.ylabel('Количество изображений')
    plt.grid(axis='y', alpha=0.5)
    plt.show()

    min_count = counts.min()
    max_count = counts.max()
    print(f"\nМинимальное количество: {min_count}")
    print(f"Максимальное количество: {max_count}")
    if (max_count - min_count) / max_count < 0.1:
        print("Вывод: Классы сбалансированы (разница менее 10%).")
    else:
        print("Вывод: Классы умеренно или сильно несбалансированны.")


def split_data(X_combined, y_combined, use_small_dataset):
    N_TRAIN_REQ = 200000
    N_VALIDATION_REQ = 10000
    N_TEST_REQ = 19000

    total_size = X_combined.shape[0]

    if use_small_dataset or total_size < N_TRAIN_REQ + N_VALIDATION_REQ + N_TEST_REQ:
        # Альтернативное разделение при недостатке данных
        print(f"ВНИМАНИЕ: Используется пропорциональное разделение (~{80}%, ~{10}%, ~{10}%) из-за малого набора данных.")

        test_ratio = 0.1
        valid_ratio_of_temp = 0.1 / (1.0 - test_ratio)

        # Тестовая выборка
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_combined, y_combined, test_size=test_ratio, random_state=42, stratify=y_combined
        )

        # Валидационная выборка
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_temp, y_temp, test_size=valid_ratio_of_temp, random_state=42, stratify=y_temp
        )

    else:
        # Тестовая выборка
        test_ratio = N_TEST_REQ / total_size
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_combined, y_combined, test_size=test_ratio, random_state=42, stratify=y_combined
        )

        # Валидационная выборка
        remaining_size = X_temp.shape[0]
        validation_ratio = N_VALIDATION_REQ / remaining_size
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_temp, y_temp, test_size=validation_ratio, random_state=42, stratify=y_temp
        )

        # Обучающая выборка
        if X_train.shape[0] > N_TRAIN_REQ:
            X_train = X_train[:N_TRAIN_REQ]
            y_train = y_train[:N_TRAIN_REQ]

    print(f"Размер обучающей выборки (X_train, y_train): {X_train.shape[0]}")
    print(f"Размер валидационной выборки (X_valid, y_valid): {X_valid.shape[0]}")
    print(f"Размер контрольной выборки (X_test, y_test): {X_test.shape[0]}")

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def deduplication(X_train, y_train, X_valid, y_valid, X_test, y_test):
    def hash_array(X):
        return [sha1(x.tobytes()).hexdigest() for x in X]

    print("Вычисление хешей для выборок...")
    hash_train = set(hash_array(X_train))
    hash_valid = set(hash_array(X_valid))
    hash_test = set(hash_array(X_test))

    # Пересечение Train и Valid
    duplicates_train_valid = hash_train.intersection(hash_valid)
    print(f"Количество дубликатов между Train и Valid: {len(duplicates_train_valid)}")

    # Пересечение Train и Test
    duplicates_train_test = hash_train.intersection(hash_test)
    print(f"Количество дубликатов между Train и Test: {len(duplicates_train_test)}")

    # Объединенный набор дубликатов для удаления
    duplicates_to_remove = duplicates_train_valid.union(duplicates_train_test)
    num_duplicates = len(duplicates_to_remove)
    print(f"Общее количество дубликатов, которые нужно удалить из Train: {num_duplicates}")

    if num_duplicates > 0:
        print("Удаление дубликатов из обучающей выборки...")

        def get_mask(X, duplicates_set):
            return [sha1(x.tobytes()).hexdigest() not in duplicates_set for x in X]

        mask = get_mask(X_train, duplicates_to_remove)
        X_train_clean = X_train[mask]
        y_train_clean = y_train[mask]

        print(f"Размер Train до очистки: {X_train.shape[0]}")
        print(f"Размер Train после очистки: {X_train_clean.shape[0]}")

        return X_train_clean, y_train_clean, X_valid, y_valid, X_test, y_test
    else:
        print("Дубликатов не найдено. Обучающая выборка остается без изменений.")
        return X_train, y_train, X_valid, y_valid, X_test, y_test


def execute_fcnn_learn(X_train, y_train, X_valid, y_valid, X_test, y_test):
    print("\n--- Базовая FCNN для сравнения с логистической регрессией ---")
    model_base = Sequential([
        Dense(512, activation='relu', input_shape=(NUM_FEATURES,), name='dense_1_relu'),
        Dense(256, activation='tanh', name='dense_2_tanh'),
        Dense(NUM_CLASSES, activation='softmax', name='output_softmax')
    ])

    optimizer = SGD(learning_rate=0.01)
    model_base.compile(optimizer=optimizer,
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    print("Базовая модель построена (512 -> 256 -> 10).")

    NUM_EPOCHS_BASE = 5
    print(f"\nНачинается обучение базовой модели на {NUM_EPOCHS_BASE} эпохах...")
    model_base.fit(
        X_train, y_train,
        epochs=NUM_EPOCHS_BASE,
        batch_size=128,
        validation_data=(X_valid, y_valid),
        verbose=1
    )

    loss_b, acc_b = model_base.evaluate(X_valid, y_valid, verbose=0)

    print("\nРасширение базовой модели с добавлением регуляризации, Dropout и динамической LR ---")

    L2_REG = 0.00001
    initial_lr = 0.002

    model_best = Sequential([
        Dense(1280, activation='relu', kernel_regularizer=l2(L2_REG), input_shape=(NUM_FEATURES,), name='dense_1_reg'),
        Dropout(0.4, name='dropout_1'),

        Dense(768, activation='relu', kernel_regularizer=l2(L2_REG), name='dense_2_reg'),
        Dropout(0.3, name='dropout_2'),

        Dense(384, activation='relu', kernel_regularizer=l2(L2_REG), name='dense_3_reg'),
        Dropout(0.2, name='dropout_3'),

        Dense(256, activation='relu', kernel_regularizer=l2(L2_REG), name='dense_4_reg'),
        Dense(192, activation='relu', name='dense_5_relu'),
        Dense(NUM_CLASSES, activation='softmax', name='output_final')
    ])

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=4,
        min_lr=0.0001,
        verbose=1
    )
    callbacks = [lr_scheduler]

    optimizer_sgd = Adam(learning_rate=initial_lr)

    model_best.compile(optimizer=optimizer_sgd,
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

    print("Продвинутая модель построена (1280 -> 768 -> 384 -> 256 -> 128 -> 10) с L2, Dropout и динамической LR.")

    NUM_EPOCHS = 50
    print(f"\nНачинается обучение продвинутой модели c количеством эпох = {NUM_EPOCHS}...")
    model_best.fit(
        X_train, y_train,
        epochs=NUM_EPOCHS,
        batch_size=256,
        validation_data=(X_valid, y_valid),
        callbacks=callbacks,
        verbose=1
    )

    # Оценка итоговой точности на тестовой выборке
    loss_final, acc_final = model_best.evaluate(X_test, y_test, verbose=0)

    print(f"Точность логистической регрессии: ~85.0%")
    print(f"Достигнутая точность базовой FCNN: {acc_b * 100:.2f}%")
    print(f"Финальная точность FCNN с L2, Dropout и динамической LR: {acc_final * 100:.4f}%")


if __name__ == "__main__":
    X_combined, y_combined = load_and_preprocess_data(DATA_DIR)

    load_and_display(X_combined, y_combined)

    check_group_balance(y_combined)

    X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(
        X_combined, y_combined, USE_SMALL_DATASET
    )

    X_train_clean, y_train_clean, _, _, _, _ = deduplication(
        X_train, y_train, X_valid, y_valid, X_test, y_test
    )

    execute_fcnn_learn(X_train, y_train, X_valid, y_valid, X_test, y_test)