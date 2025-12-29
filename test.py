from src.models.wellModels import Well


if __name__ == "__main__":
    # Example usage
    dataPath = 'data/202002.csv'
    well = Well()
    table_name='dynamic_data'
    # df = well.read_static_data_tableFile_duckdb(dataPath,table_name='static_data')
    df  = well.read_dynamic_data_tableFile_duckdb(dataPath,table_name=table_name)
    print(f'{well.header}')
    print(df.info())
    print('#' * 50)
    # df = well.get_dataframe_by_depth(3770, 3780,table_name=table_name)
    # print(df.info())