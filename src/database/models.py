import pandas as pd
from dataclasses import dataclass, field
from typing import Dict

# 1. 定义列的元数据配置
# 格式: 字段名: (中文名称, 单位, 数据类型)
COLUMN_SPECS = {
    "DEPTH": ("深度(MD)", "m", float),
    "TVD":   ("垂深", "m", float),
    "CAL":   ("井径", "in", float),
    "BIT":   ("钻头尺寸", "in", float),
    "DEV":   ("井斜角", "°", float),
    "DAZ":   ("井眼方位角", "°", float),
    "GR":    ("伽马", "API", float),
    "AC":    ("声波时差", "us/ft", float), # 或 us/m
    "DEN":   ("密度", "g/cm3", float),
    "E":     ("杨氏模量", "GPa", float),
    "U":     ("泊松比", "", float),          # 无量纲
    "C":     ("黏聚力", "MPa", float),
    "Fi":    ("内摩擦角", "°", float),
    "ST":    ("抗拉强度", "MPa", float),
    "SV":    ("垂向应力", "MPa", float),
    "SHMAX": ("最大水平主应力", "MPa", float),
    "SHMIN": ("最小水平主应力", "MPa", float),
    "Pp":    ("孔隙压力", "MPa", float)
}

@dataclass
class GeomechanicsData:
    """
    用于存储地质力学数据的 Dataclass
    """
    well_name: str
    start_dep: float # 开始深度
    end_dep: float  # 结束深度
    dlt_dep: float  # 深度间隔
    
    # 存储主要数据的 DataFrame
    data: pd.DataFrame
    
    # 存储量纲/单位信息的字典，格式 {col_name: unit}
    units: Dict[str, str] = field(default_factory=dict)
    
    # 存储中文描述信息的字典，格式 {col_name: description}
    descriptions: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后验证数据完整性"""
        if self.data is not None and not self.data.empty:
            # 确保 DataFrame 的列名与元数据匹配
            expected_cols = list(COLUMN_SPECS.keys())
            # 简单的校验，实际应用中可以根据需要放宽
            pass

    def get_unit(self, column_name: str) -> str:
        """获取指定列的单位"""
        return self.units.get(column_name, "")

def load_geomechanics_file(file_path: str) -> GeomechanicsData:
    """
    读取地质力学数据文件
    
    文件格式定义:
    Row 1: WellName (String)
    Row 2: StarDep, EndDen(Dep), DltDep (Float)
    Row 3: Headers (18 strings)
    Row 4+: Data
    """
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # --- 解析头部信息 ---
        
        # 行 1: WellName
        line1 = f.readline().strip()
        if not line1:
            raise ValueError("文件为空或格式错误: 第1行缺失")
        # 假设第一行只有一个变量，直接读取
        well_name = line1.split()[0]
        
        # 行 2: StarDep, EndDep, DltDep
        line2 = f.readline().strip()
        if not line2:
            raise ValueError("文件格式错误: 第2行缺失")
        try:
            parts_l2 = line2.split()
            if len(parts_l2) < 3:
                raise ValueError(f"第2行数据不足，需要3个数值，实际发现: {parts_l2}")
            start_dep = float(parts_l2[0])
            end_dep = float(parts_l2[1])
            dlt_dep = float(parts_l2[2])
        except ValueError as e:
            raise ValueError(f"解析第2行数值失败: {e}")

        # 行 3: Headers
        line3 = f.readline().strip()
        headers = line3.split()
        
        # 校验 Header 数量 (图片要求18个变量)
        if len(headers) != 18:
            print(f"警告: 头部定义了 {len(headers)} 列，但标准定义建议为 18 列。")

    # --- 使用 Pandas 读取数据体 ---
    # skiprows=3 跳过前3行 (WellName, Depths, Headers)
    # header=None 因为我们已经手动读取了 header
    # delim_whitespace=True 处理空格或Tab分隔
    df = pd.read_csv(
        file_path, 
        skiprows=3, 
        header=None, 
        delim_whitespace=True,
        names=headers  # 使用第3行解析出的列名
    )

    # --- 类型转换与元数据处理 ---
    unit_map = {}
    desc_map = {}
    
    for col in df.columns:
        if col in COLUMN_SPECS:
            # 获取配置
            cn_name, unit, dtype = COLUMN_SPECS[col]
            
            # 1. 强制转换数据类型
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                print(f"警告: 列 {col} 无法转换为 {dtype}, 错误: {e}")
            
            # 2. 记录元数据
            unit_map[col] = unit
            desc_map[col] = cn_name
        else:
            # 如果有未定义的列，默认为 float 且无单位
            df[col] = df[col].astype(float)
            unit_map[col] = "Unknown"
            desc_map[col] = col

    return GeomechanicsData(
        well_name=well_name,
        start_dep=start_dep,
        end_dep=end_dep,
        dlt_dep=dlt_dep,
        data=df,
        units=unit_map,
        descriptions=desc_map
    )

# ==========================================
# 测试代码：生成一个模拟文件并读取
# ==========================================
def create_dummy_file(filename):
    """生成符合图片格式的测试文件"""
    content = """Well_001
1000.0 1000.5 0.1
DEPTH TVD CAL BIT DEV DAZ GR AC DEN E U C Fi ST SV SHMAX SHMIN Pp
1000.0 998.5 8.5 8.5 2.5 45.0 60.5 75.0 2.45 25.5 0.25 15.0 30.0 5.0 45.0 35.0 25.0 20.0
1000.1 998.6 8.5 8.5 2.6 45.1 61.0 76.0 2.46 26.0 0.26 15.5 30.5 5.1 45.2 35.2 25.2 20.1
1000.2 998.7 8.5 8.5 2.7 45.2 62.0 77.0 2.47 26.5 0.27 16.0 31.0 5.2 45.4 35.4 25.4 20.2
"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    # 1. 创建模拟文件
    dummy_filename = "geomech_sample.txt"
    create_dummy_file(dummy_filename)
    
    # 2. 调用加载程序
    try:
        geo_data = load_geomechanics_file(dummy_filename)
        
        print(f"=== 成功加载井: {geo_data.well_name} ===")
        print(f"深度范围: {geo_data.start_dep} - {geo_data.end_dep} (步长: {geo_data.dlt_dep})")
        print("\n--- 数据预览 (前3行) ---")
        print(geo_data.data.head(3))
        
        print("\n--- 属性元数据验证 (单位) ---")
        # 打印几个关键列的单位，验证量纲是否保留
        for col in ["DEPTH", "DEN", "E", "Pp"]:
            print(f"{col}: {geo_data.descriptions[col]} (单位: {geo_data.get_unit(col)})")
            
        print("\n--- 数据类型验证 ---")
        print(geo_data.data.dtypes)
        
    except Exception as e:
        print(f"发生错误: {e}")
