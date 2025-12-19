import pandas as pd
import numpy as np
import math
import os
import sys
from datetime import datetime
from enum import Enum
from typing import List, Dict, Tuple, Optional

# ==============================
# 配置Excel中的列标题名称
# ==============================

# 1. 深度数据列配置
DEPTH_COLUMN_NAME = "DEP"  # 深度列标题

# 2. 用于计算a值的列配置
COLUMN1_FOR_A_NAME = "HOKHEI"  # 计算a的第一列标题（分子）
COLUMN2_FOR_A_NAME = "WELLTIME"  # 计算a的第二列标题（分母）

# 3. 用于计算v值的列配置
COLUMN1_FOR_V_NAME = "HOKHEI"  # 计算v的第一列标题（分子）
COLUMN2_FOR_V_NAME = "WELLTIME"  # 计算v的第二列标题（分母）

# 4. 参数列配置
DW_COLUMN_NAME = "Dw"  # Excel中Dw参数对应的列标题
DENL_COLUMN_NAME = "MWIN"  # Excel中denL参数对应的列标题

# 5. 坍塌破裂压力文件列配置
COLLAPSE_PRESSURE_COLUMN = "Pc"  # 坍塌压力列
FRACTURE_PRESSURE_COLUMN = "Pf"  # 破裂压力列
COLLAPSE_EQ_DENSITY_COLUMN = "DENmc"  # 坍塌压力当量密度列
FRACTURE_EQ_DENSITY_COLUMN = "DENmf"  # 破裂压力当量密度列


# 操作类型枚举
class OperationType(Enum):
    TRIPPING_OUT = 1  # 起钻
    TRIPPING_IN = 2  # 下钻


class DrillingPressureCalculator:
    def __init__(self):
        self.Dh = 0.0
        self.Dhi = 0.0
        self.fai300 = 0.0
        self.fai600 = 0.0
        self.n = 0.0
        self.K = 0.0
        self.PV = 0.0
        self.tao = 0.0

    def input_filename(self, prompt: str, default_example: str = "data.csv") -> str:
        """输入文件名"""
        print(prompt)
        print("（请确保文件为CSV格式，在同一目录或输入完整路径）")

        while True:
            filename = input(f"文件名（例如: {default_example}，输入q退出）: ").strip()

            if filename.lower() == 'q':
                print("程序退出")
                sys.exit(0)

            # 清理输入，移除可能的逗号
            filename = filename.replace(',', '').replace('，', '')

            # 如果没有扩展名，添加.csv
            if '.' not in filename:
                filename = filename + '.csv'

            # 检查文件是否存在
            if os.path.exists(filename):
                return filename
            else:
                print(f"错误: 文件 '{filename}' 不存在")
                print("请检查文件名或路径，或按q退出")

    def select_operation_type(self) -> OperationType:
        """选择操作类型"""
        print("=" * 40)
        print("请选择操作类型:")
        print("1. 起钻")
        print("2. 下钻")

        while True:
            try:
                choice = input("请输入选择 (1 或 2，输入q退出): ").strip()

                if choice.lower() == 'q':
                    print("程序退出")
                    sys.exit(0)

                choice = int(choice)
                if choice in [1, 2]:
                    break
                print("输入无效，请重新输入 (1 或 2)")
            except ValueError:
                print("输入无效，请重新输入 (1 或 2)")

        op = OperationType.TRIPPING_OUT if choice == 1 else OperationType.TRIPPING_IN
        print(f"已选择: {'起钻' if op == OperationType.TRIPPING_OUT else '下钻'}")
        return op

    def input_numeric(self, prompt: str, min_value: float = None, max_value: float = None,
                      comparison_value: float = None, comparison_type: str = 'lt') -> float:
        """通用数值输入函数"""
        while True:
            try:
                value_str = input(f"{prompt}（输入q退出）: ").strip()

                if value_str.lower() == 'q':
                    print("程序退出")
                    sys.exit(0)

                value = float(value_str)

                # 检查最小值
                if min_value is not None and value <= min_value:
                    print(f"输入无效，请输入大于{min_value}的值")
                    continue

                # 检查最大值
                if max_value is not None and value >= max_value:
                    print(f"输入无效，请输入小于{max_value}的值")
                    continue

                # 检查比较值
                if comparison_value is not None:
                    if comparison_type == 'lt' and value >= comparison_value:
                        print(f"输入无效，请输入小于{comparison_value}的值")
                        continue
                    elif comparison_type == 'gt' and value <= comparison_value:
                        print(f"输入无效，请输入大于{comparison_value}的值")
                        continue

                return value
            except ValueError:
                print("输入无效，请输入一个数字")

    def input_dh_dhi_values(self):
        """输入井筒参数"""
        print("=" * 40)
        print("请输入井筒参数:")

        self.Dh = self.input_numeric("请输入Dh值 (钻杆外径, mm)", min_value=0)

        self.Dhi = self.input_numeric("请输入Dhi值 (钻杆内径, mm)",
                                      min_value=0,
                                      comparison_value=self.Dh,
                                      comparison_type='lt')

        print(f"已输入: Dh = {self.Dh} mm, Dhi = {self.Dhi} mm")
        print("=" * 40)

    def input_fai_values(self):
        """输入流变参数"""
        print("=" * 40)
        print("请输入流变参数:")

        self.fai300 = self.input_numeric("请输入fai300值 (六速粘度计300转读数)", min_value=0)

        self.fai600 = self.input_numeric("请输入fai600值 (六速粘度计600转读数)",
                                         min_value=0,
                                         comparison_value=self.fai300,
                                         comparison_type='gt')

        print(f"已输入: fai300 = {self.fai300} 度, fai600 = {self.fai600} 度")
        print("=" * 40)

    def read_csv_with_encoding(self, filename: str) -> Optional[pd.DataFrame]:
        """尝试多种编码读取CSV文件"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                df = pd.read_csv(filename, encoding=encoding)
                print(f"使用编码 '{encoding}' 成功读取文件: {filename}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                if encoding == encodings[-1]:  # 最后一个编码
                    print(f"尝试所有编码均失败: {str(e)}")
                continue

        return None

    def read_collapse_fracture_data(self, filename: str) -> Dict[int, Tuple[float, float, float, float]]:
        """读取坍塌破裂压力数据"""
        data_map = {}

        df = self.read_csv_with_encoding(filename)
        if df is None:
            print(f"错误: 无法打开坍塌破裂压力文件 {filename}")
            return data_map

        # 清理列名
        df.columns = [str(col).strip().replace(' ', '').replace('\t', '').replace('\r', '').replace('\n', '')
                      for col in df.columns]

        print(f"文件列名: {list(df.columns)}")

        # 查找列
        depth_col = self._find_column(df.columns, DEPTH_COLUMN_NAME,
                                      ["dep", "depth", "井深", "深度"])

        if depth_col is None:
            print("错误: 未能找到深度列")
            return data_map

        collapse_pressure_col = self._find_column(df.columns, COLLAPSE_PRESSURE_COLUMN,
                                                  ["pc", "坍塌", "塌", "collapse"])

        fracture_pressure_col = self._find_column(df.columns, FRACTURE_PRESSURE_COLUMN,
                                                  ["pf", "破裂", "破", "fracture"])

        collapse_eq_density_col = self._find_column(df.columns, COLLAPSE_EQ_DENSITY_COLUMN,
                                                    ["denmc", "坍塌密度", "collapsedensity"])

        fracture_eq_density_col = self._find_column(df.columns, FRACTURE_EQ_DENSITY_COLUMN,
                                                    ["denmf", "破裂密度", "fracturedensity"])

        print(f"找到的列: 深度={depth_col}, 坍塌压力={collapse_pressure_col}, "
              f"破裂压力={fracture_pressure_col}")

        # 处理数据
        for _, row in df.iterrows():
            try:
                depth = int(float(row[depth_col]))

                collapse_pressure = 0.0
                fracture_pressure = 0.0
                collapse_eq_density = 0.0
                fracture_eq_density = 0.0

                if collapse_pressure_col and collapse_pressure_col in df.columns:
                    collapse_pressure = float(row[collapse_pressure_col])

                if fracture_pressure_col and fracture_pressure_col in df.columns:
                    fracture_pressure = float(row[fracture_pressure_col])

                if collapse_eq_density_col and collapse_eq_density_col in df.columns:
                    collapse_eq_density = float(row[collapse_eq_density_col])

                if fracture_eq_density_col and fracture_eq_density_col in df.columns:
                    fracture_eq_density = float(row[fracture_eq_density_col])

                data_map[depth] = (collapse_pressure, fracture_pressure,
                                   collapse_eq_density, fracture_eq_density)
            except Exception as e:
                continue

        print(f"成功读取 {len(data_map)} 行坍塌破裂压力数据")
        return data_map

    def _find_column(self, columns: List[str], exact_name: str, fuzzy_names: List[str]) -> Optional[str]:
        """查找列名"""
        # 精确匹配
        if exact_name in columns:
            return exact_name

        # 模糊匹配（不区分大小写）
        columns_lower = [col.lower() for col in columns]

        # 检查配置的名称
        if exact_name.lower() in columns_lower:
            return columns[columns_lower.index(exact_name.lower())]

        # 检查模糊名称
        for fuzzy_name in fuzzy_names:
            for i, col in enumerate(columns_lower):
                if fuzzy_name in col:
                    return columns[i]

        return None

    def read_main_data(self, filename: str) -> pd.DataFrame:
        """读取主数据"""
        df = self.read_csv_with_encoding(filename)
        if df is None:
            print(f"错误: 无法打开主数据文件 {filename}")
            return pd.DataFrame()

        # 清理列名
        original_columns = list(df.columns)
        df.columns = [str(col).strip().replace(' ', '').replace('\t', '').replace('\r', '').replace('\n', '')
                      for col in df.columns]

        print(f"文件列名: {original_columns}")
        print(f"清理后列名: {list(df.columns)}")

        # 查找必需的列
        column_mapping = {}

        # 深度列（必需）
        depth_col = self._find_column(df.columns, DEPTH_COLUMN_NAME,
                                      ["dep", "depth", "井深", "深度"])
        if depth_col:
            column_mapping[depth_col] = "Depth"
            print(f"找到深度列: {depth_col}")
        else:
            print("错误: 未能找到深度列")
            return pd.DataFrame()

        # Dw列（必需）
        dw_col = self._find_column(df.columns, DW_COLUMN_NAME,
                                   ["dw", "钻杆外径", "钻杆直径"])
        if dw_col:
            column_mapping[dw_col] = "Dw"
            print(f"找到Dw列: {dw_col}")
        else:
            print("错误: 未能找到Dw列")
            return pd.DataFrame()

        # denL列（必需）
        denL_col = self._find_column(df.columns, DENL_COLUMN_NAME,
                                     ["mwin", "denl", "密度", "泥浆", "钻井液"])
        if denL_col:
            column_mapping[denL_col] = "denL"
            print(f"找到denL列: {denL_col}")
        else:
            print("错误: 未能找到denL列")
            return pd.DataFrame()

        # HOKHEI列（用于计算a和v的分子）
        hokhei_col = self._find_column(df.columns, COLUMN1_FOR_A_NAME, ["hokhei", "hook", "大钩"])
        if hokhei_col:
            column_mapping[hokhei_col] = "HOKHEI"
            print(f"找到HOKHEI列: {hokhei_col}")
        else:
            print("错误: 未能找到HOKHEI列")
            return pd.DataFrame()

        # WELLTIME列（用于计算a和v的分母）- 时间格式
        welltime_col = self._find_column(df.columns, COLUMN2_FOR_A_NAME, ["welltime", "time", "时间"])
        if welltime_col:
            column_mapping[welltime_col] = "WELLTIME"
            print(f"找到WELLTIME列: {welltime_col}")
        else:
            print("错误: 未能找到WELLTIME列")
            return pd.DataFrame()

        # WELLDATE列 - 日期列
        welldate_col = self._find_column(df.columns, "WELLDATE", ["welldate", "date", "日期", "datetime"])
        if welldate_col:
            column_mapping[welldate_col] = "WELLDATE"
            print(f"找到WELLDATE列: {welldate_col}")
        else:
            print("警告: 未能找到WELLDATE列，将假设所有数据在同一天")
            df["WELLDATE"] = "2000-01-01"  # 默认日期

        # 重命名列
        df = df.rename(columns=column_mapping)

        # 转换WELLDATE为datetime格式
        print(f"\n转换WELLDATE和WELLTIME格式...")
        df["WELLDATE_dt"] = pd.to_datetime(df["WELLDATE"], errors='coerce')

        # 转换WELLTIME时间格式为秒数（考虑日期）
        df["WELLTIME_total_seconds"] = df.apply(
            lambda row: self._convert_time_to_seconds_with_date(row["WELLDATE_dt"], row["WELLTIME"]),
            axis=1
        )

        # 显示数据预览
        print(f"\n数据预览（前5行）:")
        print(df[["Depth", "Dw", "denL", "HOKHEI", "WELLDATE", "WELLTIME", "WELLTIME_total_seconds"]].head())

        # 显示数据类型和统计信息
        print(f"\n数据类型和统计信息:")
        for col in ["Depth", "Dw", "denL", "HOKHEI", "WELLTIME_total_seconds"]:
            if col in df.columns:
                print(f"  {col}: {df[col].dtype}")
                print(f"    NaN数量: {df[col].isna().sum()}")
                print(f"    最小值: {df[col].min() if not df[col].isna().all() else 'NaN'}")
                print(f"    最大值: {df[col].max() if not df[col].isna().all() else 'NaN'}")

        # 检查是否有数据
        if len(df) == 0:
            print("警告: 读取的数据为空")
            return pd.DataFrame()

        print(f"成功读取 {len(df)} 行主数据")
        return df

    def _convert_time_to_seconds_with_date(self, date_dt, time_str):
        """将日期和时间字符串转换为总秒数（从某个参考点开始）"""
        if pd.isna(time_str) or pd.isna(date_dt):
            return 0.0

        try:
            # 如果date_dt已经是datetime，直接使用
            if isinstance(date_dt, pd.Timestamp):
                base_date = date_dt
            else:
                # 否则尝试解析日期
                base_date = pd.to_datetime(date_dt, errors='coerce')
                if pd.isna(base_date):
                    return 0.0

            # 解析时间字符串
            if isinstance(time_str, str):
                # 尝试解析 HH:MM:SS 格式
                if ':' in time_str:
                    parts = time_str.split(':')
                    if len(parts) == 3:  # HH:MM:SS
                        hours = int(parts[0])
                        minutes = int(parts[1])
                        seconds = int(float(parts[2]))  # 处理可能的小数秒
                    elif len(parts) == 2:  # MM:SS
                        hours = 0
                        minutes = int(parts[0])
                        seconds = int(float(parts[1]))
                    else:
                        # 尝试直接解析为秒数
                        return float(time_str)
                else:
                    # 直接作为秒数处理
                    return float(time_str)
            else:
                # 如果已经是数字，假设是秒数
                hours = 0
                minutes = 0
                seconds = float(time_str)

            # 创建时间对象
            time_obj = pd.Timestamp(year=base_date.year, month=base_date.month, day=base_date.day,
                                    hour=hours, minute=minutes, second=int(seconds))

            # 计算从某个参考点开始的秒数（例如1970-01-01）
            return time_obj.timestamp()

        except Exception as e:
            print(f"时间转换错误: {e}, date_dt={date_dt}, time_str={time_str}")
            return 0.0

    def _convert_time_to_seconds(self, time_str):
        """将时间字符串转换为秒数"""
        if pd.isna(time_str):
            return 0.0

        try:
            # 尝试解析时间格式 HH:MM:SS
            if isinstance(time_str, str):
                parts = time_str.split(':')
                if len(parts) == 3:  # HH:MM:SS
                    hours = float(parts[0])
                    minutes = float(parts[1])
                    seconds = float(parts[2])
                    return hours * 3600 + minutes * 60 + seconds
                elif len(parts) == 2:  # MM:SS
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
                else:
                    # 尝试直接转换为数字
                    return float(time_str)
            else:
                # 如果已经是数字，直接返回
                return float(time_str)
        except:
            return 0.0

    def calculate_a_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算a值 - 根据C++代码逻辑，考虑日期变化"""
        df = df.copy()

        # 确保数据类型正确
        df["HOKHEI"] = pd.to_numeric(df["HOKHEI"], errors='coerce')
        df["WELLTIME_total_seconds"] = pd.to_numeric(df["WELLTIME_total_seconds"], errors='coerce')

        # 检查是否有NaN值
        print(f"\n计算a值前检查:")
        print(f"  HOKHEI NaN数量: {df['HOKHEI'].isna().sum()}")
        print(f"  WELLTIME_total_seconds NaN数量: {df['WELLTIME_total_seconds'].isna().sum()}")

        # 显示前几行的原始值
        print(f"\n原始值预览（前5行）:")
        for i in range(min(5, len(df))):
            print(
                f"  行{i}: HOKHEI={df.iloc[i]['HOKHEI']:.4f}, WELLTIME_total_seconds={df.iloc[i]['WELLTIME_total_seconds']:.4f}")

        # 如果有NaN，填充0
        df['HOKHEI'] = df['HOKHEI'].fillna(0)
        df['WELLTIME_total_seconds'] = df['WELLTIME_total_seconds'].fillna(0)

        # 按照C++代码逻辑计算a值
        # 计算差值
        df["delta_HOKHEI"] = df["HOKHEI"].diff()
        df["delta_WELLTIME"] = df["WELLTIME_total_seconds"].diff()

        # 处理第一行（按照C++逻辑）
        if len(df) > 1:
            df.loc[df.index[0], "delta_HOKHEI"] = df.loc[df.index[1], "HOKHEI"] - df.loc[df.index[0], "HOKHEI"]
            df.loc[df.index[0], "delta_WELLTIME"] = df.loc[df.index[1], "WELLTIME_total_seconds"] - df.loc[
                df.index[0], "WELLTIME_total_seconds"]
        else:
            df.loc[df.index[0], "delta_HOKHEI"] = 0
            df.loc[df.index[0], "delta_WELLTIME"] = 1.0

        # 显示差值
        print(f"\n差值预览（前5行）:")
        for i in range(min(5, len(df))):
            print(
                f"  行{i}: delta_HOKHEI={df.iloc[i]['delta_HOKHEI']:.6f}, delta_WELLTIME={df.iloc[i]['delta_WELLTIME']:.6f}")

        # 处理负的时间差（跨天的情况应该不会出现负值，因为使用了timestamp）
        df["delta_WELLTIME"] = df["delta_WELLTIME"].abs()

        # 如果时间差为0，设置为一个很小的值避免除以0
        df["delta_WELLTIME"] = df["delta_WELLTIME"].apply(lambda x: max(x, 1e-10))

        # 计算a值
        df["a"] = df["delta_HOKHEI"] / (df["delta_WELLTIME"] ** 2)

        # 显示a值
        print(f"\na值预览（前5行）:")
        for i in range(min(5, len(df))):
            print(f"  行{i}: a={df.iloc[i]['a']:.12f}")

        return df

    def calculate_v_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算v值 - 根据C++代码逻辑，考虑日期变化"""
        df = df.copy()

        # 确保数据类型正确
        df["HOKHEI"] = pd.to_numeric(df["HOKHEI"], errors='coerce')
        df["WELLTIME_total_seconds"] = pd.to_numeric(df["WELLTIME_total_seconds"], errors='coerce')

        # 显示前几行的原始值
        print(f"\n计算v值 - 原始值预览（前5行）:")
        for i in range(min(5, len(df))):
            print(
                f"  行{i}: HOKHEI={df.iloc[i]['HOKHEI']:.4f}, WELLTIME_total_seconds={df.iloc[i]['WELLTIME_total_seconds']:.4f}")

        # 如果有NaN，填充0
        df['HOKHEI'] = df['HOKHEI'].fillna(0)
        df['WELLTIME_total_seconds'] = df['WELLTIME_total_seconds'].fillna(0)

        # 计算差值
        df["delta_HOKHEI_v"] = df["HOKHEI"].diff()
        df["delta_WELLTIME_v"] = df["WELLTIME_total_seconds"].diff()

        # 处理第一行（按照C++逻辑）
        if len(df) > 1:
            df.loc[df.index[0], "delta_HOKHEI_v"] = df.loc[df.index[1], "HOKHEI"] - df.loc[df.index[0], "HOKHEI"]
            df.loc[df.index[0], "delta_WELLTIME_v"] = df.loc[df.index[1], "WELLTIME_total_seconds"] - df.loc[
                df.index[0], "WELLTIME_total_seconds"]
        else:
            df.loc[df.index[0], "delta_HOKHEI_v"] = 1.0
            df.loc[df.index[0], "delta_WELLTIME_v"] = 1.0

        # 显示差值
        print(f"\n计算v值 - 差值预览（前5行）:")
        for i in range(min(5, len(df))):
            print(
                f"  行{i}: delta_HOKHEI_v={df.iloc[i]['delta_HOKHEI_v']:.6f}, delta_WELLTIME_v={df.iloc[i]['delta_WELLTIME_v']:.6f}")

        # 处理负的时间差
        df["delta_WELLTIME_v"] = df["delta_WELLTIME_v"].abs()

        # 如果时间差为0，设置为一个很小的值避免除以0
        df["delta_WELLTIME_v"] = df["delta_WELLTIME_v"].apply(lambda x: max(x, 1e-10))

        # 计算v值
        df["v"] = df["delta_HOKHEI_v"] / df["delta_WELLTIME_v"]

        # 显示v值
        print(f"\n计算v值 - v值预览（前5行）:")
        for i in range(min(5, len(df))):
            print(f"  行{i}: v={df.iloc[i]['v']:.12f}")

        return df

    def calculate_equivalent_density(self, pressure_MPa: float, depth_m: float) -> float:
        """计算压力当量密度"""
        g = 9.81
        if depth_m > 0:
            return (pressure_MPa * 1e6) / (g * depth_m) / 1000.0
        return 0.0

    def fill_collapse_fracture_data(self, df: pd.DataFrame,
                                    collapse_fracture_map: Dict[
                                        int, Tuple[float, float, float, float]]) -> pd.DataFrame:
        """填充坍塌和破裂压力数据"""
        df = df.copy()

        # 初始化列
        df["collapse_pressure"] = 0.0
        df["fracture_pressure"] = 0.0
        df["original_collapse_eq_density"] = 0.0
        df["original_fracture_eq_density"] = 0.0

        if not collapse_fracture_map:
            print("警告: 坍塌破裂压力数据为空，将使用默认值0.0")
            return df

        # 为每一行查找匹配的数据
        for idx, row in df.iterrows():
            depth_int = int(row["Depth"])

            if depth_int in collapse_fracture_map:
                cp, fp, cden, fden = collapse_fracture_map[depth_int]
                df.at[idx, "collapse_pressure"] = cp
                df.at[idx, "fracture_pressure"] = fp
                df.at[idx, "original_collapse_eq_density"] = cden
                df.at[idx, "original_fracture_eq_density"] = fden
            else:
                # 查找最接近的深度
                closest_depth = None
                min_diff = float('inf')

                for map_depth in collapse_fracture_map.keys():
                    diff = abs(map_depth - depth_int)
                    if diff < min_diff:
                        min_diff = diff
                        closest_depth = map_depth

                if closest_depth is not None and min_diff <= 10:
                    cp, fp, cden, fden = collapse_fracture_map[closest_depth]
                    df.at[idx, "collapse_pressure"] = cp
                    df.at[idx, "fracture_pressure"] = fp
                    df.at[idx, "original_collapse_eq_density"] = cden
                    df.at[idx, "original_fracture_eq_density"] = fden

        return df

    def calculate_tripping_out_pressure(self, depth: float, a: float, v_value: float,
                                        dw: float, denL: float) -> float:
        """起钻压力计算 - 根据C++代码逻辑"""
        # 单位转换（C++中除以1000.0）
        dw_m = dw / 1000.0
        dh_m = self.Dh / 1000.0
        dhi_m = self.Dhi / 1000.0
        denL_m= denL * 1000.0

        # 流变参数计算（与C++一致）
        self.n = 3.322 * math.log10(self.fai600 / self.fai300)
        self.K = 0.511 * self.fai300 / (511.0 ** self.n)
        self.PV = self.fai600 - self.fai300
        self.tao = 0.511 * (self.fai300 - self.PV)

        v_used = v_value

        # 调试信息
        if depth < 10:  # 只显示前几行的调试信息
            print(f"\n调试信息 - 输入参数:")
            print(f"  depth={depth}, a={a}, v={v_used}, dw={dw}, denL={denL}")
            print(f"  dw_m={dw_m}, dh_m={dh_m}, dhi_m={dhi_m}")
            print(f"  流变参数: n={self.n:.4f}, K={self.K:.4f}, PV={self.PV:.2f}, tao={self.tao:.4f}")



        # 计算雷诺数（与C++公式一致）
        try:
            # 分别计算各个部分
            diameter_diff = dw_m - dh_m
            if diameter_diff <= 0:
                return 0.0

            # 计算 (dw_m - dh_m) ** n
            diameter_power = diameter_diff ** self.n

            # 计算 v_used ** (2 - n)
            if v_used <= 0:
                velocity_power = 0
            else:
                velocity_power = v_used ** (2 - self.n)

            # 计算分母
            denominator_part1 = pow(12, (self.n - 1)) * self.K
            denominator_part2 = pow(((2 * self.n + 1) / (3 * self.n)), self.n)

            denominator = denominator_part1 * denominator_part2

            if denominator == 0:
                Re = 1.0  # 避免除以零
            else:
                Re = diameter_power * velocity_power * denL_m / denominator

                # 确保Re是实数
                if isinstance(Re, complex):
                    Re = abs(Re)  # 取绝对值

            if depth < 10:
                print(f"  雷诺数计算: diameter_power={diameter_power:.6f}, velocity_power={velocity_power:.6f}")
                print(f"  denominator={denominator:.6f}, Re={Re:.6f}")

        except (ValueError, ZeroDivisionError) as e:
            if depth < 10:
                print(f"  雷诺数计算错误: {e}")
            Re = 1.0

        Re1 = 3470 - 1370 * self.n

        # 计算摩擦系数（与C++逻辑一致）
        f = 0
        if isinstance(Re, (int, float)) and Re > 0 and Re1 > 0:  # 确保Re是实数
            if Re > Re1:
                a_val = (math.log10(self.n) + 3.93) / 50
                b = (1.75 - math.log10(self.n)) / 7
                f = a_val / (Re ** b) if Re > 0 else 0.0
                f *= 0.8  # C++中乘以0.8
                if depth < 10:
                    print(f"  层流摩擦系数: a_val={a_val:.6f}, b={b:.6f}, f={f:.6f}")
            else:
                f = 24 / Re * 0.8 if Re > 0 else 0.0  # C++中乘以0.8
                if depth < 10:
                    print(f"  湍流摩擦系数: f={f:.6f}")
        else:
            f = 0.0
            if depth < 10:
                print(f"  Re无效: Re={Re}, Re1={Re1}")

        # 检查(dw_m - dh_m)是否为0
        if abs(dw_m - dh_m) < 1e-10:
            if depth < 10:
                print(f"  警告: (dw_m - dh_m)接近0")
            return 0.0  # 避免除以零

        # 计算各压力分量（与C++公式一致）
        P1 = 4 * self.tao * depth / (dw_m - dh_m) * 0.9  # C++中乘以0.9
        if depth < 10:
            print(f"  P1计算: 4*{self.tao:.4f}*{depth}/{dw_m - dh_m:.6f}*0.9 = {P1:.6f}")

        P2 = 0
        if Re > Re1:
            P2 = 2 * f * denL_m * v_used * v_used / (dw_m - dh_m)
            if depth < 10:
                print(f"  P2计算(湍流): 2*{f:.6f}*{denL}*{v_used}^2/{dw_m - dh_m:.6f} = {P2:.6f}")
        else:
            # 检查分母是否为0
            inner_term = 4 * (2 * self.n + 1) * v_used / self.n / (dw_m - dh_m)
            if inner_term <= 0:  # 避免负数
                P2 = 0
                if depth < 10:
                    print(f"  inner_term <= 0: {inner_term}")
            else:
                P2 = 4 * self.K * depth / (dw_m - dh_m) * (inner_term ** self.n) * 0.9
                if depth < 10:
                    print(f"  P2计算(层流): inner_term={inner_term:.6f}, P2={P2:.6f}")

        # 检查分母是否为0
        denominator_p3 = (dw_m ** 2 - dh_m ** 2 + dhi_m ** 2)
        if abs(denominator_p3) < 1e-10:
            P3 = 0
            if depth < 10:
                print(f"  分母接近0: denominator_p3={denominator_p3:.6f}")
        else:
            P3 = denL_m * depth * a * (dh_m ** 2 - dhi_m ** 2) / denominator_p3 * 0.85


        total_pressure = (P1 + P2 + P3) / (10 ** 6)  # 转换为MPa


        # 抽吸效应（C++中的额外调整）
        swab_effect = 0.1 * (depth / 1000.0)
        total_pressure *= (1.0 - swab_effect)



        return max(total_pressure, 0.0)

    def calculate_tripping_in_pressure(self, depth: float, a: float, v_value: float,
                                       dw: float, denL: float) -> float:
        """下钻压力计算 - 根据C++代码逻辑"""
        # 单位转换
        dw_m = dw / 1000.0
        dh_m = self.Dh / 1000.0
        dhi_m = self.Dhi / 1000.0
        denL_m= denL * 1000.0

        # 流变参数
        self.n = 3.322 * math.log10(self.fai600 / self.fai300)
        self.K = 0.511 * self.fai300 / (511.0 ** self.n)
        self.PV = self.fai600 - self.fai300
        self.tao = 0.511 * (self.fai300 - self.PV)

        v_used = v_value

        # 检查输入参数的有效性
        if dw_m <= dh_m or v_used <= 0 or denL <= 0 or self.K <= 0:
            return 0.0

        # 计算雷诺数
        # 修改计算方式，避免复数
        try:
            # 分别计算各个部分
            diameter_diff = dw_m - dh_m
            if diameter_diff <= 0:
                return 0.0

            # 计算 (dw_m - dh_m) ** n
            diameter_power = diameter_diff ** self.n

            # 计算 v_used ** (2 - n)
            if v_used <= 0:
                velocity_power = 0
            else:
                velocity_power = v_used ** (2 - self.n)

            # 计算分母
            denominator_part1 = pow(12, (self.n - 1)) * self.K
            denominator_part2 = pow(((2 * self.n + 1) / (3 * self.n)), self.n)

            denominator = denominator_part1 * denominator_part2

            if denominator == 0:
                Re = 1.0  # 避免除以零
            else:
                Re = diameter_power * velocity_power * denL_m / denominator

                # 确保Re是实数
                if isinstance(Re, complex):
                    Re = abs(Re)  # 取绝对值

        except (ValueError, ZeroDivisionError):
            Re = 1.0

        Re1 = 3470 - 1370 * self.n

        # 计算摩擦系数（C++中乘以1.2）
        f = 0
        if isinstance(Re, (int, float)) and Re > 0 and Re1 > 0:  # 确保Re是实数
            if Re > Re1:
                a_val = (math.log10(self.n) + 3.93) / 50
                b = (1.75 - math.log10(self.n)) / 7
                f = a_val / (Re ** b) if Re > 0 else 0.0
                f *= 1.2  # C++中乘以1.2
            else:
                f = 24 / Re * 1.2 if Re > 0 else 0.0  # C++中乘以1.2
        else:
            f = 0.0

        # 检查(dw_m - dh_m)是否为0
        if abs(dw_m - dh_m) < 1e-10:
            return 0.0  # 避免除以零

        # 计算各压力分量
        P1 = 4 * self.tao * depth / (dw_m - dh_m) * 1.1  # C++中乘以1.1

        P2 = 0
        if Re > Re1:
            P2 = 2 * f * denL_m * v_used * v_used / (dw_m - dh_m)
        else:
            # 检查分母是否为0
            inner_term = 4 * (2 * self.n + 1) * v_used / self.n / (dw_m - dh_m)
            if inner_term <= 0:  # 避免负数
                P2 = 0
            else:
                P2 = 4 * self.K * depth / (dw_m - dh_m) * (inner_term ** self.n) * 1.1

        # 检查分母是否为0
        denominator_p3 = (dw_m ** 2 - dh_m ** 2 + dhi_m ** 2)
        if abs(denominator_p3) < 1e-10:
            P3 = 0
        else:
            P3 = denL_m * depth * a * (dh_m ** 2 - dhi_m ** 2) / denominator_p3 * 1.15

        total_pressure = (P1 + P2 + P3) / (10 ** 6)

        # 激动效应
        surge_effect = 0.15 * (depth / 1000.0)
        total_pressure *= (1.0 + surge_effect)

        return total_pressure

    def adjust_pressures_and_calculate_densities(self, df: pd.DataFrame, operation: OperationType) -> pd.DataFrame:
        """调整压力并计算当量密度"""
        df = df.copy()

        if operation == OperationType.TRIPPING_OUT:
            # 起钻：坍塌/破裂压力 + 计算的压力
            df["adjusted_collapse_pressure"] = df["collapse_pressure"] + df["pressure"]
            df["adjusted_fracture_pressure"] = df["fracture_pressure"] + df["pressure"]
        else:
            # 下钻：坍塌/破裂压力 - 计算的压力
            df["adjusted_collapse_pressure"] = df["collapse_pressure"] - df["pressure"]
            df["adjusted_fracture_pressure"] = df["fracture_pressure"] - df["pressure"]

        # 确保非负
        df["adjusted_collapse_pressure"] = df["adjusted_collapse_pressure"].clip(lower=0.0)
        df["adjusted_fracture_pressure"] = df["adjusted_fracture_pressure"].clip(lower=0.0)

        # 计算调整后的当量密度
        df["adjusted_collapse_eq_density"] = df.apply(
            lambda row: self.calculate_equivalent_density(row["adjusted_collapse_pressure"], row["Depth"]),
            axis=1
        )

        df["adjusted_fracture_eq_density"] = df.apply(
            lambda row: self.calculate_equivalent_density(row["adjusted_fracture_pressure"], row["Depth"]),
            axis=1
        )

        return df

    def calculate_all_pressures(self, df: pd.DataFrame, operation: OperationType) -> pd.DataFrame:
        """主计算函数"""
        if df.empty:
            print("错误: 没有数据可计算")
            return df

        # 确保数据类型正确
        df["Depth"] = pd.to_numeric(df["Depth"], errors='coerce')
        df["Dw"] = pd.to_numeric(df["Dw"], errors='coerce')
        df["denL"] = pd.to_numeric(df["denL"], errors='coerce')
        df["a"] = pd.to_numeric(df["a"], errors='coerce')
        df["v"] = pd.to_numeric(df["v"], errors='coerce')

        # 删除NaN值
        df = df.dropna(subset=["Depth", "Dw", "denL", "a", "v"])

        if len(df) == 0:
            print("错误: 清理NaN值后没有数据")
            return df

        # 计算流变参数
        self.n = 3.322 * math.log10(self.fai600 / self.fai300)
        self.K = 0.511 * self.fai300 / (511.0 ** self.n)
        self.PV = self.fai600 - self.fai300
        self.tao = 0.511 * (self.fai300 - self.PV)

        print(f"流变参数: n={self.n:.4f}, K={self.K:.4f}, PV={self.PV:.2f}, tao={self.tao:.4f}")

        # 计算钻井压力
        if operation == OperationType.TRIPPING_OUT:
            print("计算起钻压力...")
            df["pressure"] = df.apply(
                lambda row: self.calculate_tripping_out_pressure(
                    row["Depth"], row["a"], row["v"], row["Dw"], row["denL"]
                ), axis=1
            )
        else:
            print("计算下钻压力...")
            df["pressure"] = df.apply(
                lambda row: self.calculate_tripping_in_pressure(
                    row["Depth"], row["a"], row["v"], row["Dw"], row["denL"]
                ), axis=1
            )

        # 调整压力并计算当量密度
        df = self.adjust_pressures_and_calculate_densities(df, operation)

        return df

    def save_results(self, df: pd.DataFrame, operation: OperationType,
                     main_filename: str='', collapse_filename: str=''):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存详细结果到CSV
        csv_filename = f"drilling_results_{timestamp}.csv"

        # 计算钻井当量密度
        df["drilling_eq_density"] = df.apply(
            lambda row: self.calculate_equivalent_density(row["pressure"], row["Depth"]),
            axis=1
        )

        # 创建结果DataFrame
        result_df = df[[
            "Depth", "Dw", "denL", "a", "v", "pressure",
            "collapse_pressure", "fracture_pressure",
            "original_collapse_eq_density", "original_fracture_eq_density",
            "adjusted_collapse_pressure", "adjusted_fracture_pressure",
            "adjusted_collapse_eq_density", "adjusted_fracture_eq_density",
            "drilling_eq_density"
        ]].copy()

        # 重命名列
        result_df.columns = [
            "Depth(m)", "Dw(mm)", "denL(kg/m3)", "a", "v", "Drilling_Pressure(MPa)",
            "Input_Pc(MPa)", "Input_Pf(MPa)",
            "Input_DENmc(g/cm3)", "Input_DENmf(g/cm3)",
            "Adjusted_Pc(MPa)", "Adjusted_Pf(MPa)",
            "Adjusted_DENmc(g/cm3)", "Adjusted_DENmf(g/cm3)",
            "Drilling_EQ_Density(g/cm3)"
        ]

        # 添加操作类型列
        result_df["Operation"] = "Tripping_Out" if operation == OperationType.TRIPPING_OUT else "Tripping_In"

        # 写入CSV文件
        try:
            result_df.to_csv(csv_filename, index=False, float_format="%.6f")
            print(f"详细结果已保存到文件: {csv_filename}")
        except Exception as e:
            print(f"保存CSV文件失败: {str(e)}")

        # 保存简化的对比数据文件
        txt_filename = f"comparison_results_{timestamp}.txt"
        try:
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write("# 钻井压力计算结果对比\n")
                f.write(f"# 生成时间: {timestamp}\n")
                f.write(f"# 操作类型: {'起钻' if operation == OperationType.TRIPPING_OUT else '下钻'}\n")
                f.write(f"# 流变参数: fai300 = {self.fai300}°, fai600 = {self.fai600}°\n")
                f.write(f"# 井筒参数: Dh = {self.Dh} mm, Dhi = {self.Dhi} mm\n")
                f.write(
                    f"# 调整规则: {'起钻: 调整后压力 = 原始压力 + 钻井压力' if operation == OperationType.TRIPPING_OUT else '下钻: 调整后压力 = 原始压力 - 钻井压力'}\n")
                f.write("# 列说明:\n")
                f.write("#   Depth - 深度(m)\n")
                f.write("#   Input_Pc - 输入坍塌压力(MPa)\n")
                f.write("#   Input_Pf - 输入破裂压力(MPa)\n")
                f.write("#   Drilling_Pressure - 计算钻井压力(MPa)\n")
                f.write("#   Adjusted_Pc - 调整后坍塌压力(MPa)\n")
                f.write("#   Adjusted_Pf - 调整后破裂压力(MPa)\n")
                f.write("#   Input_DENmc - 输入坍塌当量密度(g/cm³)\n")
                f.write("#   Input_DENmf - 输入破裂当量密度(g/cm³)\n")
                f.write("#   Adjusted_DENmc - 调整后坍塌当量密度(g/cm³)\n")
                f.write("#   Adjusted_DENmf - 调整后破裂当量密度(g/cm³)\n")
                f.write("#\n")

                # 写入表头
                f.write("Depth\tInput_Pc\tInput_Pf\tDrilling_Pressure\tAdjusted_Pc\tAdjusted_Pf\t"
                        "Input_DENmc\tInput_DENmf\tAdjusted_DENmc\tAdjusted_DENmf\n")

                # 写入数据
                for _, row in df.iterrows():
                    f.write(f"{row['Depth']:.0f}\t"
                            f"{row['collapse_pressure']:.6f}\t"
                            f"{row['fracture_pressure']:.6f}\t"
                            f"{row['pressure']:.6f}\t"
                            f"{row['adjusted_collapse_pressure']:.6f}\t"
                            f"{row['adjusted_fracture_pressure']:.6f}\t"
                            f"{row['original_collapse_eq_density']:.6f}\t"
                            f"{row['original_fracture_eq_density']:.6f}\t"
                            f"{row['adjusted_collapse_eq_density']:.6f}\t"
                            f"{row['adjusted_fracture_eq_density']:.6f}\n")

            print(f"对比数据已保存到文件: {txt_filename}")
        except Exception as e:
            print(f"保存TXT文件失败: {str(e)}")

    def run(self):
        """主程序"""
        print("=" * 40)
        print("      钻井压力计算程序（起钻/下钻）")
        print("      包含坍塌破裂压力调整和当量密度计算")
        print("=" * 40)

        # 1. 选择操作类型
        operation = self.select_operation_type()

        # 2. 输入主数据文件名
        main_filename = self.input_filename("请输入主数据文件名", "202002.csv")

        # 3. 输入坍塌破裂压力文件名
        collapse_filename = self.input_filename("请输入坍塌破裂压力文件名", "PcPf.csv")

        # 4. 手动输入Dh和Dhi
        self.input_dh_dhi_values()

        # 5. 手动输入fai300和fai600
        self.input_fai_values()

        # 6. 读取坍塌破裂压力数据
        print("\n正在读取数据...")
        collapse_fracture_map = self.read_collapse_fracture_data(collapse_filename)

        # 7. 读取主数据
        df = self.read_main_data(main_filename)

        if df.empty or len(df) == 0:
            print("\n错误: 未能读取主数据，程序退出")
            print(f"读取的数据行数: {len(df) if not df.empty else 0}")
            input("按Enter键退出...")
            return

        # 8. 计算a值和v值
        print("\n计算a值和v值...")
        df = self.calculate_a_values(df)
        df = self.calculate_v_values(df)

        # 9. 填充坍塌破裂压力数据
        print("填充坍塌破裂压力数据...")
        df = self.fill_collapse_fracture_data(df, collapse_fracture_map)

        # 显示计算出的a和v值
        print("\n计算出的a和v值预览（前5行）:")
        preview_columns = ["Depth", "HOKHEI", "WELLTIME", "delta_HOKHEI", "delta_WELLTIME", "a", "v"]
        preview_columns = [col for col in preview_columns if col in df.columns]
        preview_df = df[preview_columns].head()
        print(preview_df.to_string())

        # 10. 根据操作类型计算压力
        print("\n计算钻井压力...")
        df = self.calculate_all_pressures(df, operation)

        if df.empty or len(df) == 0:
            print("错误: 计算后没有有效数据")
            input("按Enter键退出...")
            return

        # 11. 保存结果
        print("\n保存计算结果...")
        self.save_results(df, operation, main_filename, collapse_filename)

        print("\n" + "=" * 40)
        print("程序执行完成！")
        print(f"操作类型: {'起钻' if operation == OperationType.TRIPPING_OUT else '下钻'}")
        print(f"计算数据行数: {len(df)}")
        print("结果文件已保存")
        print("=" * 40)

        # 显示前几行结果预览
        print("\n结果预览（前5行）:")
        result_columns = ["Depth", "pressure", "collapse_pressure", "fracture_pressure",
                          "adjusted_collapse_pressure", "adjusted_fracture_pressure"]
        result_columns = [col for col in result_columns if col in df.columns]
        print(df[result_columns].head())

        # 提示用户按任意键退出
        input("\n按Enter键退出程序...")


if __name__ == "__main__":
    try:
        calculator = DrillingPressureCalculator()
        calculator.run()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序发生错误: {str(e)}")
        import traceback

        traceback.print_exc()
        input("按Enter键退出...")