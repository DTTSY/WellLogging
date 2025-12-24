from enum import Enum
from time import perf_counter

from PySide6.QtCore import QThread, Signal

import calculatefoDrrillingVelocityMpkg
import PYCalculate_for_homorockPkg
import PYCalculate_for_added_homorockPkg

import datetime as dt
import pandas as pd


class CalculateForHomorock:
    def __init__(self):
        try:
            # Initialize Python module created with MATLAB Compiler SDK.
            self.my_PYcalculatefordrillingVelocityPkg = calculatefoDrrillingVelocityMpkg.initialize()
            self.my_PYCalculate_for_homorockPkg = PYCalculate_for_homorockPkg.initialize()
            self.my_PYCalculate_for_added_homorock = PYCalculate_for_added_homorockPkg.initialize()
        except Exception as e:
            raise Exception("Error initializing matlab package\\n:{}".format(e))

    def calculate_for_added_homorock(self, xin_for_added_homorock: pd.DataFrame):
        yOut:pd.DataFrame = self.my_PYCalculate_for_added_homorock.PYCalculate_for_added_homorock(xin_for_added_homorock)
        return yOut

    def calculatefordrillingVelocity(self, xIn: pd.DataFrame):
        yOut:pd.DataFrame = self.my_PYcalculatefordrillingVelocityPkg.PYcalculatefordrillingVelocity(xIn)
        return yOut
    
    def calculate_for_homorock(self, xin_for_homorock: pd.DataFrame):
        yOut:pd.DataFrame = self.my_PYCalculate_for_homorockPkg.PYCalculate_for_homorock(xin_for_homorock)
        return yOut


    def terminate(self):
        self.my_PYcalculatefordrillingVelocityPkg.terminate()
        self.my_PYCalculate_for_homorockPkg.terminate()
        self.my_PYCalculate_for_added_homorock.terminate()


class HomorockTask(Enum):
    ADDED = "added_homorock"
    BASE = "homorock"
    DRILLING_VELOCITY = "drilling_velocity"


class HomorockCalculationThread(QThread):
    """Run homorock related calculations off the UI thread."""
    result_ready = Signal(object, float)
    error_occurred = Signal(str)

    def __init__(self, data: pd.DataFrame, task: HomorockTask, parent=None):
        super().__init__(parent)
        self._data = data
        self._task = task

    def run(self) -> None:
        calculator = None
        result = 'noresult'
        try:
            calculator = CalculateForHomorock()
            start = perf_counter()
            if self._task is HomorockTask.ADDED:
                result = calculator.calculate_for_added_homorock(self._data)
            elif self._task is HomorockTask.BASE:
                result = calculator.calculate_for_homorock(self._data)
            elif self._task is HomorockTask.DRILLING_VELOCITY:
                result = calculator.calculatefordrillingVelocity(self._data)
            else:
                raise ValueError("Unsupported homorock task")
            elapsed = perf_counter() - start
            if not isinstance(result, pd.DataFrame):
                raise ValueError("HomorockTask did not return a DataFrame")
            self.result_ready.emit(result, elapsed)
        except Exception as exc:
            self.error_occurred.emit(f'msg: {result}\n{str(exc)}')
        finally:
            if calculator is not None:
                calculator.terminate()