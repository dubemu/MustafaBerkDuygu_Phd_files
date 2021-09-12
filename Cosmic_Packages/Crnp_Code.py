# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:21:08 2020

@author: musta
"""



import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import os




#Define initial Parameters:
Porosity=0.51 #Average porosity obtained during sampling
Beta = 0.0077 #Attenuation coefficient (h/Pa)
k= 216.68 #Gas constant for water vapor (g k/J)
P0= 1035.25 #Air Pressure (hPa)
gw = 0.1479 #Gravimetric water content (kg/kg)
bd= 1.304 #Soil bulk density (g/cm³)
Nc = 978.7 #Corrected neutron counting rate
Ref_Int=56.44 #Average solar intensity at Athens station for the year 2017 
soc = 0 #Soil organic carbon (kg/kg)
lw =0 #Clay lattice water (kg/kg) 
Neutron_Station="ATHN" #Name of the neutron station used in nmdb database

Modules=r'C:\Users\musta\Desktop\Cosmic_Packages\Modules'
Data_Files=r'C:\Users\musta\Desktop\Cosmic_Packages\Data'
Generated_Files=r'C:\Users\musta\Desktop\Cosmic_Packages\Generated_Files'


#Read Row Data:
cakit_data=pd.read_csv(os.path.join(Data_Files,"Cakit_Data.csv"), sep=";", parse_dates=['DateTime(UTC)'],date_parser=lambda x: pd.to_datetime(x))
cakit_data = cakit_data.set_index(pd.DatetimeIndex(cakit_data['DateTime(UTC)']))
cakit_data.drop(["DateTime(UTC)"], axis=1,inplace=True)

cakit_data.head()



import sys
sys.path.insert(1,Modules) #We need this line to import crnpfunctions
import crnpfunctions

crnpfunctions.get_intensity_data(station=Neutron_Station,save_file=Generated_Files)
crnpfunctions.get_crnp_data(Porosity,Beta,k,P0,gw,bd,Nc,Ref_Int,soc,lw,cakit_data,Data_Files,Generated_Files)



