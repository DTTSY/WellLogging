#!/usr/bin/env python

import calculatefoDrrillingVelocityMpkg
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
except Exception as e:
    print("Error initializing PYcalculatefordrillingVelocity package\\n:{}".format(e))
    exit(1)

try:
    # Create a DataFrame from the data for the past 24 hours.
    # Convert data into a DataFrame.
    xIn = pd.read_excel(r"data\SAMPLE.xlsx", sheet_name='Sheet1')
    # Call the packaged function.
    print(xIn.info())
    # data = {
    #     "Time": [
    #         dt.datetime.strptime(i, "%Y-%m-%d %H:%M")
    #         for i in [
    #             "2024-10-30 18:00",
    #             "2024-10-30 21:00",
    #             "2024-10-31 00:00",
    #             "2024-10-31 03:00",
    #             "2024-10-31 06:00",
    #             "2024-10-31 09:00",
    #             "2024-10-31 12:00",
    #             "2024-10-31 15:00",
    #             "2024-10-31 18:00"
    #         ]
    #     ],
    #     "Temperature (°F)": [67, 64, 62, 60, 58, 61, 70, 75, 79],
    #     "Weather": [
    #         "Clear",
    #         "Overcast",
    #         "Passing clouds",
    #         "Passing clouds",
    #         "Passing clouds",
    #         "Partly sunny",
    #         "Partly sunny",
    #         "Passing clouds",
    #         "Partly sunny"
    #     ],
    #     "Wind Speed (mph)": [8, 10, 7, 6, 5, 7, 9, 10, 12],
    #     "Humidity (%)": [51, 73, 78, 80, 82, 74, 65, 58, 50],
    # }

    # # Convert data into a DataFrame.
    # xIn = pd.DataFrame(data)
    yOut:pd.DataFrame = my_PYcalculatefordrillingVelocityPkg.PYcalculatefordrillingVelocity(xIn)
    print(type(yOut))
    print(f'近钻头平均碰撞次数: {yOut["N_Collision_Count"].mean():.1f}')
    print(f'远钻头平均碰撞次数: : {yOut["F_Collision_Count"].mean():.1f}')
    yOut.to_csv(r"output/calculatefordrillingVelocity.csv", index=False)

except Exception as e:
    print("Error occurred during program execution\n:{}".format(e))

my_PYcalculatefordrillingVelocityPkg.terminate()