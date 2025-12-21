import lasio
from welly import Well
import welly
import pandas as pd
import matplotlib.pyplot as plt


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
    "FI":    ("内摩擦角", "°", float),
    "ST":    ("抗拉强度", "MPa", float),
    "SV":    ("垂向应力", "MPa", float),
    "SHMAX": ("最大水平主应力", "MPa", float),
    "SHMIN": ("最小水平主应力", "MPa", float),
    "PP":    ("孔隙压力", "MPa", float)
}


class wellModels:
    @staticmethod
    def read_las_file(file_path):
        """
        Reads a LAS file and returns the lasio.LASFile object.

        :param file_path: Path to the LAS file.
        :return: lasio.LASFile object.
        """
        try:
            las_file = lasio.read(file_path)
            return las_file
        except Exception as e:
            print(f"Error reading LAS file: {e}")
            return None

    @staticmethod
    def get_curve_data(las_file: lasio.LASFile, curve_name: list[str]):
        """
        Retrieves data for a specific curve from the LAS file.

        :param las_file: lasio.LASFile object.
        :param curve_name: Name of the curve to retrieve data for.
        :return: Data array for the specified curve or None if not found.
        """
        try:
            curves_data = [las_file.curves[name].data for name in curve_name]
            return curves_data
        except Exception as e:
            print(f"Error retrieving curve data: {e}")
            print(f"Curve '{curve_name}' not found in LAS file.")
            return None
    
    @staticmethod
    def get_well_info(las_file):
        """
        Retrieves well information from the LAS file.

        :param las_file: lasio.LASFile object.
        :return: Dictionary containing well information.
        """
        well_info = {}
        try:
            well_info['well_name'] = las_file.well.WELL.value
            well_info['location'] = las_file.well.LOC.value
            well_info['company'] = las_file.well.COMP.value
            well_info['date'] = las_file.well.DATE.value
        except Exception as e:
            print(f"Error retrieving well info: {e}")
        
        return well_info
    
    @staticmethod
    def get_depth_range(las_file: lasio.LASFile):
        """
        Retrieves the depth range from the LAS file.

        :param las_file: lasio.LASFile object.
        :return: Tuple containing (start_depth, end_depth).
        """
        try:
            start_depth = las_file.curves[0].data[0]
            end_depth = las_file.curves[0].data[-1]
            return (start_depth, end_depth)
        except Exception as e:
            print(f"Error retrieving depth range: {e}")
            return (None, None)
    
    @staticmethod
    def from_file_to_las(file_path,meta_data, las_file):
        """
        Converts a CSV file to LAS format and saves it.

        :param csv_file: Path to the input CSV file.
        :param las_file: Path to the output LAS file.
        """
        try:
            las = lasio.LASFile()
            # df = pd.read_csv(csv_file)
            df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip()
            df.columns = df.columns.str.upper()
            print("df columns:", df.columns)
            # set metadata
            for key, value in meta_data.items():
                las.well[key] = lasio.HeaderItem(key, value=value)  
            # add curves
            # print(las.header)
            #去除columns中的空格
            for column in df.columns:
                print("column in df: ",column,'1')
                unit = COLUMN_SPECS.get(column, ("", "", ""))[1] if column in COLUMN_SPECS else ""
                descr = COLUMN_SPECS.get(column, ("", "", ""))[0] if column in COLUMN_SPECS else ""
                las.append_curve(column, df[column], unit=unit, descr=descr)

            las.write(las_file)
            print(f"Successfully converted {file_path} to {las_file}.")
            return las
        except Exception as e:
            print(f"Error converting CSV to LAS: {e}")
            return None

if __name__ == "__main__":
    # Example usage
    file_path = "data/X3.xlsx"
    meta_data = {"WELL": "测试井", "LOC": "Location", "COMP": "Company", "DATE": "2024-06-01", "UWI": "1234567890",'NULL': -9999}
    las_file = wellModels.from_file_to_las(file_path, meta_data ,"output/output.las")
    
    if las_file:
        well_info = wellModels.get_well_info(las_file)
        print("Well Information:", well_info)
        
        depth_range = wellModels.get_depth_range(las_file)
        print("Depth Range:", depth_range)
        
        curve_name = ["AC", "TVD"]
        curve_data = wellModels.get_curve_data(las_file, curve_name)
        if curve_data is not None:
            print(f"Data for curve '{curve_name}':", curve_data)
    
        # well = Well.from_lasio(las_file)
    
    well = Well.from_las("output/output.las")
    # well = Well.from_las('https://geocomp.s3.amazonaws.com/data/P-129.LAS')
#     remap = {
#     'UWI': 'LIC',  # Commonly used unique name; not a true UWI.
#     'KB': 'EKB',
#     'TD': 'TDD',  # Driller's TD.
#     'LATI': 'LOC',
#     'LONG': 'UWI',
#     'SECT': None,
#     'TOWN': None,
#     'LOC': None
# }

    # well = Well.from_las('https://geocomp.s3.amazonaws.com/data/P-129.LAS', remap=remap)
    print(well.header)
    print(well.data)
    df = well.df()
    print(df.info())
    print(well.header)
    # print(well.get_alias())
    # 设置字体为支持中文
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    trck_name = ['TVD','AC','DEN','E','C','FI',['SIGMMA_V1','SIGMMA_H1','SIGMMA_HH1'],'TVD']
    fig = well.plot(tracks=trck_name)
    fig.show()
    fig.savefig("output/well_plot.png")
    # print(df.head(4))
    # AC = well.data['AC']
    # print(AC.describe())
    # print(AC)
    # AC.plot()
    # tracks = ['AC','SH',['TOC', 'TOC2']]
    # well.plot(tracks=tracks)
    # plt.savefig("output/well_plot.png")
    # plt.show()

    # well.plot_2d(logs=curve_name)
