#!/usr/bin/env python

import calculatefoDrrillingVelocityMpkg
import PYCalculate_for_homorockPkg
import PYCalculate_for_added_homorockPkg

print("MATLAB Package for Drilling Velocity Calculation Test")
print("===============================================")
# Import the matlab module only after you have imported
# MATLAB Compiler SDK generated Python modules.
# import matlab
import datetime as dt
import pandas as pd

try:
    # Initialize Python module created with MATLAB Compiler SDK.
    my_PYcalculatefordrillingVelocityPkg = calculatefoDrrillingVelocityMpkg.initialize()
    my_PYCalculate_for_homorockPkg = PYCalculate_for_homorockPkg.initialize()
    my_PYCalculate_for_added_homorock = PYCalculate_for_added_homorockPkg.initialize()
except Exception as e:
    print("Error initializing PYcalculatefordrillingVelocity package\\n:{}".format(e))
    exit(1)

try:
    # Create a DataFrame from the data for the past 24 hours.
    # Convert data into a DataFrame.
    xIn = pd.read_excel(r"data\SAMPLE.xlsx")
    # xin_for_for_homorock = pd.read_excel("data/资226龙潭组.xlsx")
    xin_for_added_homorock = pd.read_excel("data/计算for_added_homorock.xlsx")
    print(xin_for_added_homorock.info())
    # 将所有列转换为float数值类型
    xin_for_added_homorock = xin_for_added_homorock.astype(float)
    # Call the packaged function.
    # print(xIn.info())
    # # Convert data into a DataFrame.
    # xIn = pd.DataFrame(data)

    # yOut:pd.DataFrame = my_PYcalculatefordrillingVelocityPkg.PYcalculatefordrillingVelocity(xIn)
    # print(type(yOut))
    # print(f'近钻头平均碰撞次数: {yOut["N_Collision_Count"].mean():.1f}')
    # print(f'远钻头平均碰撞次数: : {yOut["F_Collision_Count"].mean():.1f}')
    # yOut.to_csv(r"output/calculatefordrillingVelocity.csv", index=False)

    # yOut_homorock:pd.DataFrame = my_PYCalculate_for_homorockPkg.PYCalculate_for_homorock(xin_for_for_added_homorock)
    # print(type(yOut_homorock))

    # yOut_homorock.to_csv(r"output/calculateforhomorock.csv", index=False)

    yOut_added_homorock = my_PYCalculate_for_added_homorock.PYCalculate_for_added_homorock(xin_for_added_homorock)
    print(yOut_added_homorock)
    # yOut_added_homorock.to_csv(r"output/calculateforaddedhomorock.csv", index=False)


except Exception as e:
    print("Error occurred during program execution\n:{}".format(e))

my_PYcalculatefordrillingVelocityPkg.terminate()