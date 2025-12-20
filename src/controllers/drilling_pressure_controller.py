from __future__ import annotations

from time import perf_counter
from typing import Callable, Optional

import pandas as pd

from PySide6.QtCore import QObject, Signal

from src.core.algorithm.DrillingPressureCalculator import (
    DrillingPressureCalculator,
    OperationType,
)
from src.models.drilling_pressure_model import DrillingPressureModel


ProgressCallback = Callable[[int, str], None]


class CalculationWorker(QObject):
    progress = Signal(int, str, float)
    finished = Signal(pd.DataFrame, float)
    error = Signal(str)

    def __init__(self, controller: DrillingPressureController, operation: OperationType):
        super().__init__()
        self._controller = controller
        self._operation = operation

    def run(self) -> None:
        start_time = perf_counter()

        def progress_callback(value: int, message: str) -> None:
            elapsed = perf_counter() - start_time
            self.progress.emit(value, message, elapsed)

        try:
            df = self._controller.perform_calculation(self._operation, progress_callback)
            total_elapsed = perf_counter() - start_time
            self.finished.emit(df, total_elapsed)
        except Exception as exc:  # pylint: disable=broad-except
            self.error.emit(str(exc))


class DrillingPressureController:
    """Controller handling drilling pressure calculations."""

    def __init__(self, model: DrillingPressureModel):
        self._model = model
        self._calculator = DrillingPressureCalculator()

    @property
    def model(self) -> DrillingPressureModel:
        return self._model

    def load_main_data(self, file_path: str) -> None:
        data = self._calculator.read_main_data(file_path)
        if data.empty:
            raise ValueError("未能读取有效的主数据信息")
        self._model.main_data = data
        self._model.clear_result()

    def load_cf_data(self, file_path: str) -> None:
        cf_data = self._calculator.read_collapse_fracture_data(file_path)
        if not cf_data:
            raise ValueError("未能读取有效的坍塌破裂数据")
        self._model.cf_data = cf_data
        self._model.clear_result()

    def update_parameters(self, Dh: float, Dhi: float, fai300: float, fai600: float) -> None:
        params = self._model.params
        params.Dh = Dh
        params.Dhi = Dhi
        params.fai300 = fai300
        params.fai600 = fai600

    def create_worker(self, operation: OperationType) -> CalculationWorker:
        if not self._model.is_ready():
            raise ValueError("数据未加载完整")
        return CalculationWorker(self, operation)

    def perform_calculation(
        self,
        operation: OperationType,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        if not self._model.is_ready():
            raise ValueError("数据未加载完整")

        calculator = self._build_calculator()
        df = self._model.main_data.copy(deep=True)

        self._emit_progress(progress_callback, 5, "准备计算参数")
        df = calculator.calculate_a_values(df)

        self._emit_progress(progress_callback, 25, "已计算a值")
        df = calculator.calculate_v_values(df)

        self._emit_progress(progress_callback, 45, "已计算v值")
        df = calculator.fill_collapse_fracture_data(df, self._model.cf_data)

        self._emit_progress(progress_callback, 65, "已填充坍塌破裂数据")
        df = calculator.calculate_all_pressures(df, operation)

        self._emit_progress(progress_callback, 85, "已计算钻井压力")
        calculator.save_results(df, operation)

        self._model.result = df
        return df

    def _build_calculator(self) -> DrillingPressureCalculator:
        calculator = DrillingPressureCalculator()
        params = self._model.params
        calculator.Dh = params.Dh
        calculator.Dhi = params.Dhi
        calculator.fai300 = params.fai300
        calculator.fai600 = params.fai600
        return calculator

    def _emit_progress(
        self,
        callback: Optional[ProgressCallback],
        value: int,
        message: str,
    ) -> None:
        if callback is not None:
            callback(value, message)
