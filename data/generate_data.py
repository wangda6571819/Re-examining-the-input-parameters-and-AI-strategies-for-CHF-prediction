import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
from iapws import IAPWS97

# Load the CSV file, skipping the first two header lines
file_path = './chf_public.csv'
data = pd.read_csv(file_path, header=None)
data.drop(data.columns[-1], axis=1, inplace=True)

# Set the column names using the first row and remove the first two rows
column_names = data.iloc[0].tolist()
data.columns = column_names
data = data.drop([0, 1])


# Enthalpy value of water
# Enthalpy value of air
def calculateEnthalpy(pressure):
    """
    根据压力计算水焓或气焓（kind='L'或'V'）
    :param kind: 字符串，'L'表示水焓，'V'表示气焓
    :param pressure: 压力值（kPa）
    :return: 水焓或气焓（kJ/kg）
    """
    pressure = float(pressure)
    vapor = IAPWS97(P= pressure/1000, x= 1.0)
    HVapor = vapor.h

    vaporH = IAPWS97(P= pressure/1000, x= 0)
    hLiquid = vaporH.h
    return [hLiquid,HVapor]
        
def calculateEnthalpyTemperature(pressure) :
    pressure = float(pressure)
    vapor = IAPWS97(P= pressure/1000, x= 1.0)
    return vapor.T

def calculateHD(row):
    hd = float(row['Heated Length']) / float(row['Tube Diameter'])
    return hd

data['Hw'] = data.apply(lambda x : calculateEnthalpy(x['Pressure'])[0],axis=1)
data['Hg'] = data.apply(lambda x : calculateEnthalpy(x['Pressure'])[1],axis=1)

data['Tsat'] = data.apply(lambda x : calculateEnthalpyTemperature(x['Pressure']),axis=1)
data['H/D'] = data.apply(lambda x : calculateHD(x),axis=1)

# Convert pressure from kPa to MPa (since IAPWS97 uses MPa)
# data['Pressure_MPa'] = data['Pressure'] / 1000.0

# Convert temperature from Celsius to Kelvin (since IAPWS97 uses Kelvin)
data['TinK'] = pd.to_numeric(data['Inlet Temperature'], errors='coerce') + 273.15

# Calculate the densities at the given pressure (P)
data['RQufP'] = data['Pressure'].apply(lambda P: IAPWS97(P=float(P)/1000, x=0).rho)  # Saturated water density (kg/m^3)
data['Roufg'] = data['Pressure'].apply(lambda P: IAPWS97(P=float(P)/1000, x=1).rho)  # Saturated steam density (kg/m^3)

# Calculate roufin (density of water at the inlet temperature Tin)
data['Rouin'] = data['TinK'].apply(lambda T: IAPWS97(T=T, x=0).rho)  # Density in kg/m^3

# Calculate the surface tension of water at the inlet temperature (Tin)
data['Sigmaf'] = data['TinK'].apply(lambda T: IAPWS97(T=T, x=0).sigma)  # Surface tension in N/m

# Save the updated data to a new CSV file
data.to_csv('all_properties_data.csv', index=False)





