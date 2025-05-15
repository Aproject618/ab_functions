# BootstrapABTest

Библиотека для проведения бутстрэп-анализа A/B-тестов с визуализацией на Plotly.

---

## 🔍 Описание

`BootstrapABTest` — это простой, но гибкий инструмент для статистического сравнения двух групп с помощью бутстрэп-метода. Он позволяет:

- Получить доверительные интервалы для средней метрики в группах.
- Вычислить p-value.
- Построить наглядные распределения.
- Работать как с односторонними, так и с двусторонними тестами.

---

## 📦 Установка

### Вариант 1: установка из исходников
Вариант 1: установка из исходников
```bash
git clone https://github.com/Aproject618/ab_functions.git
cd bootstrap_ab_test
pip install .
```
### Вариант 2: установка напрямую через pip
```bash
python3 -m pip install git+https://github.com/Aproject618/ab_functions.git
```

### 🚀 Быстрый старт
```Python
from bootstrap_ab_test import BootstrapABTest
import pandas as pd

# Примерные данные
df = pd.read_csv("your_data.csv")

bootstrap = BootstrapABTest(
    data=df,
    metric="revenue",
    group_column="group",
    group_names=["control", "test"],
    n=10000,
    alpha=0.05,
    two_sided=True,
    random_state=42
)

bootstrap.run(show_plots=True)
bootstrap.summary()

# Получить результаты вручную:
result_df, boot_diff, p_value = bootstrap.get_results()
```

### 🔧 Аргументы класса
| Аргумент       | Описание                                         |
| -------------- | ------------------------------------------------ |
| `data`         | `pd.DataFrame` с исходными данными               |
| `metric`       | Название колонки метрики                         |
| `group_column` | Название колонки с группами                      |
| `group_names`  | Список из двух названий групп: `[control, test]` |
| `n`            | Кол-во итераций бутстрэпа (по умолчанию 10000)   |
| `alpha`        | Уровень значимости (по умолчанию 0.05)           |
| `two_sided`    | Если `True`, выполняется двусторонний тест       |
| `random_state` | Фиксация random seed для воспроизводимости       |

### 📈 Методы
| Метод                  | Назначение                                                         |
| ---------------------- | ------------------------------------------------------------------ |
| `run(show_plots=True)` | Запустить анализ, при желании вывести графики                      |
| `summary()`            | Напечатать интерпретацию результата                                |
| `get_results()`        | Вернуть: `DataFrame` со средними, `Series` с разностями, `p-value` |

#### 📊 Пример визуализации
    Распределение средних значений для групп.

    Распределение разностей между группами.

Все графики отображаются через Plotly.

### 📜 Лицензия
MIT Licence

### 🧑‍💻 Автор
Ramis Sungatullin