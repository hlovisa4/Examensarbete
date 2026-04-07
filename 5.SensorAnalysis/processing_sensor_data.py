# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:42:08 2026

@author: Lovisa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = "C:/Users/Lovisa/Downloads/FW_ Request for tree coordinates_IDs for box-setup datasets (11324 & 47355)/TOA5_47356_Sept22_2025.txt"

df = pd.read_csv(FILE_PATH)
df = df.dropna(axis=1, how='all')
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

columns_to_keep = ['TIMESTAMP', 'RECORD', 'Pan_T', 'Batt_V', 'Teros12_temp(1)',
       'Teros12_temp(2)', 'Teros12_temp(3)', 'Teros12_temp(4)',
       'Teros12_temp(5)', 'Teros12_temp(6)', 'Teros12_VWC(1)',
       'Teros12_VWC(2)', 'Teros12_VWC(3)', 'Teros12_VWC(4)', 'Teros12_VWC(5)',
       'Teros12_VWC(6)']
df = df[columns_to_keep]

df = df.rename(columns={'Teros12_VWC(1)': 'VWC_1_14m', 'Teros12_VWC(2)': 'VWC_1_8m', 
                        'Teros12_VWC(3)': 'VWC_1_1.3m', 'Teros12_VWC(4)': 'VWC_2_14m', 
                        'Teros12_VWC(5)': 'VWC_2_8m', 'Teros12_VWC(6)': 'VWC_2_1.3m'})




ids = [1,2]
heights = [1.3, 8, 14]
for i in ids:
    for h in heights:
        df[f'calculated_sap_flow_{i}_{h}m'] = 0.0006822 * df[f"VWC_{i}_{h}m"] - 1.3869976
  
plt.plot(df["TIMESTAMP"], df['calculated_sap_flow_1_8m'])