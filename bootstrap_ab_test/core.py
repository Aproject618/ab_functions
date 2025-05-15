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

    def _plot_distributions(self):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=self.result_df['control'], name='Control'))
        fig.add_trace(go.Histogram(x=self.result_df['test'], name='Test'))
        fig.update_layout(title='Распределения бутстрэп-средних', barmode='overlay')
        fig.update_traces(opacity=0.7)
        fig.show()

    def _plot_diff(self):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=self.boot_diff))
        fig.update_layout(title='Распределение разностей (test - control)')
        fig.show()

    def run(self, show_plots: bool = True) -> None:
        """
        Запускает бутстрэп-анализ.
        :param show_plots: показывать ли графики
        """
        self.result_df, self.boot_diff = self._bootstrap()
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

    def get_results(self) -> Tuple[pd.DataFrame, pd.Series, float]:
        """
        Возвращает результат анализа.

        :return: (DataFrame со средними, Series с разностями, p-value)
        """
        if self.result_df is None or self.boot_diff is None or self.p_value is None:
            raise ValueError("Сначала вызовите метод run()")
        return self.result_df, self.boot_diff, self.p_value
