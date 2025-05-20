import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
from typing import List, Optional, Tuple


class BootstrapABTest:
    def __init__(
        self,
        data: pd.DataFrame,
        metric: str,
        group_column: str,
        group_names: List[str],
        n: int = 10000,
        alpha: float = 0.05,
        two_sided: bool = False,
        random_state: Optional[int] = None,
    ):
        """
        Класс для проведения бутстрэп-анализа между двумя группами.

        :param data: DataFrame с вашими данными
        :param metric: имя метрики
        :param group_column: колонка с названиями групп
        :param group_names: список из двух названий групп [control, test]
        :param n: число итераций бутстрэпа
        :param alpha: уровень значимости
        :param two_sided: использовать ли двусторонний тест
        :param random_state: random seed для воспроизводимости
        """
        self.data = data
        self.metric = metric
        self.group_column = group_column
        self.group_names = group_names
        self.n = n
        self.alpha = alpha
        self.two_sided = two_sided
        self.random_state = random_state

        self.control_data = data.loc[data[group_column] == group_names[0], metric].copy()
        self.test_data = data.loc[data[group_column] == group_names[1], metric].copy()

        self.result_df: Optional[pd.DataFrame] = None
        self.boot_diff: Optional[pd.Series] = None
        self.p_value: Optional[float] = None
        self.interpretation: Optional[str] = None
        self.ci_control: Optional[Tuple[float, float]] = None
        self.ci_test: Optional[Tuple[float, float]] = None
        self.ci_diff: Optional[Tuple[float, float]] = None

        if random_state is not None:
            np.random.seed(random_state)

    def _bootstrap(self) -> Tuple[pd.DataFrame, pd.Series]:
        control_means = []
        test_means = []

        for _ in tqdm(range(self.n), desc="Bootstrapping"):
            control_sample = self.control_data.sample(n=len(self.control_data), replace=True)
            test_sample = self.test_data.sample(n=len(self.test_data), replace=True)

            control_means.append(control_sample.mean())
            test_means.append(test_sample.mean())

        df = pd.DataFrame({'control': control_means, 'test': test_means})
        diff = df['test'] - df['control']
        return df, diff

    def _calculate_p_value(self, boot_diff: pd.Series) -> Tuple[float, str]:
        if self.two_sided:
            p_value = (abs(boot_diff) >= abs(boot_diff.mean())).mean()
            interpretation = 'Двусторонний тест'
        else:
            if boot_diff.mean() > 0:
                p_value = (boot_diff <= 0).mean()
                interpretation = 'Тестовая группа статистически значимо лучше'
            else:
                p_value = (boot_diff >= 0).mean()
                interpretation = 'Контрольная группа статистически значимо лучше'

        if p_value >= self.alpha:
            interpretation = f'Разница не является статистически значимой, p-value = {round(p_value, 4)}'
        else:
            interpretation += f', p-value = {round(p_value, 4)}'

        return p_value, interpretation

    def _calculate_confidence_intervals(self):
        """Вычисляет доверительные интервалы для контрольной, тестовой групп и разницы."""
        self.ci_control = np.percentile(self.result_df['control'], [self.alpha/2*100, 100-self.alpha/2*100])
        self.ci_test = np.percentile(self.result_df['test'], [self.alpha/2*100, 100-self.alpha/2*100])
        self.ci_diff = np.percentile(self.boot_diff, [self.alpha/2*100, 100-self.alpha/2*100])

    def _plot_distributions(self):
        fig = go.Figure()
        
        # Добавляем гистограммы
        fig.add_trace(go.Histogram(
            x=self.result_df['control'], 
            name=f'Control ({self.group_names[0]})',
            marker_color='blue',
            opacity=0.5
        ))
        fig.add_trace(go.Histogram(
            x=self.result_df['test'], 
            name=f'Test ({self.group_names[1]})',
            marker_color='red',
            opacity=0.5
        ))
        
        # Добавляем линии для доверительных интервалов
        for ci, color, name in zip(
            [self.ci_control, self.ci_test],
            ['blue', 'red'],
            ['Control CI', 'Test CI']
        ):
            fig.add_vline(x=ci[0], line_dash="dash", line_color=color, annotation_text=f'{name} lower')
            fig.add_vline(x=ci[1], line_dash="dash", line_color=color, annotation_text=f'{name} upper')
            fig.add_vrect(
                x0=ci[0], x1=ci[1],
                fillcolor=color, opacity=0.1,
                line_width=0
            )
        
        fig.update_layout(
            title='Распределения бутстрэп-средних с доверительными интервалами',
            barmode='overlay',
            xaxis_title='Значение метрики',
            yaxis_title='Частота'
        )
        fig.show()

    def _plot_diff(self):
        fig = go.Figure()
        
        # Гистограмма разностей
        fig.add_trace(go.Histogram(
            x=self.boot_diff,
            name='Разность (Test - Control)',
            marker_color='purple',
            opacity=0.7
        ))
        
        # Линии для доверительного интервала разности
        fig.add_vline(x=self.ci_diff[0], line_dash="dash", line_color='black', annotation_text='CI lower')
        fig.add_vline(x=self.ci_diff[1], line_dash="dash", line_color='black', annotation_text='CI upper')
        fig.add_vrect(
            x0=self.ci_diff[0], x1=self.ci_diff[1],
            fillcolor="gray", opacity=0.2,
            line_width=0,
            annotation_text=f'{(1-self.alpha)*100}% доверительный интервал'
        )
        
        # Нулевая линия
        fig.add_vline(x=0, line_color='red', line_width=2)
        
        fig.update_layout(
            title='Распределение разностей с доверительным интервалом',
            xaxis_title='Разность (Test - Control)',
            yaxis_title='Частота'
        )
        fig.show()

    def run(self, show_plots: bool = True) -> None:
        """
        Запускает бутстрэп-анализ.
        :param show_plots: показывать ли графики
        """
        self.result_df, self.boot_diff = self._bootstrap()
        self._calculate_confidence_intervals()
        self.p_value, self.interpretation = self._calculate_p_value(self.boot_diff)

        if show_plots:
            self._plot_diff()
            self._plot_distributions()

    def summary(self) -> None:
        """
        Печатает интерпретацию результата и p-value.
        """
        if self.p_value is None:
            print("Сначала вызовите метод run()")
        else:
            print(self.interpretation)
            print(f"\nДоверительные интервалы (уровень доверия {(1-self.alpha)*100}%):")
            print(f"Контрольная группа ({self.group_names[0]}): [{self.ci_control[0]:.4f}, {self.ci_control[1]:.4f}]")
            print(f"Тестовая группа ({self.group_names[1]}): [{self.ci_test[0]:.4f}, {self.ci_test[1]:.4f}]")
            print(f"Разность (Test - Control): [{self.ci_diff[0]:.4f}, {self.ci_diff[1]:.4f}]")

    def get_results(self) -> Tuple[pd.DataFrame, pd.Series, float, Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Возвращает результат анализа.

        :return: (DataFrame со средними, Series с разностями, p-value, 
                 CI контрольной группы, CI тестовой группы, CI разности)
        """
        if self.result_df is None or self.boot_diff is None or self.p_value is None:
            raise ValueError("Сначала вызовите метод run()")
        return (self.result_df, self.boot_diff, self.p_value, 
                self.ci_control, self.ci_test, self.ci_diff)