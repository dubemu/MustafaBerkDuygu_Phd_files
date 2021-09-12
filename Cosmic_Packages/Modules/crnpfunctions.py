# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:33:42 2020

@author: Mustafa Berk Duygu

All required functions for CRNP analyses
"""

def get_intensity_data(station="ATHN",save_file=r"C:\Users\musta\Desktop\Cosmic_Packages\Generated_Files"):
    """
    Gathers neutron intensity data for nearby neutron monitoring station
    The data obtained will be used in the intensity correction.
    """    
    import datetime
    import pandas as pd
    import urllib
    import os
    now = datetime.datetime.now()    
    url=f'http://www.nmdb.eu/nest/draw_graph.php?formchk=1&stations[]={station}&tabchoice=revori&dtype=corr_for_pressure&tresolution=60&force=1&yunits=0&date_choice=bydate&start_day=11&start_month=11&start_year=2016&start_hour=12&start_min=46&end_day='+str(now.day)+'&end_month='+str(now.month)+'&end_year='+str(now.year)+'&end_hour=00&end_min=46&output=ascii'
    source_code=urllib.request.urlopen(url).read().decode()
    data=[]
    split_source=source_code.split('\n')
    for line in split_source:
        split_line=line.split(';')
        data.append(split_line)
    starting_line=data.index(['  start_date_time   RCORR_P'])
    newdata=data[starting_line+1:-4]    
    dataframe=pd.DataFrame(newdata)   
    dataframe.columns=["Timestamp","Intensity"]
    dataframe = dataframe.set_index(dataframe['Timestamp'])
    dataframe.drop(["Timestamp"], axis=1,inplace=True)
    dataframe.to_csv(os.path.join(save_file,"Intensity_Data.csv"))
    
def get_crnp_data(Porosity,Beta,k,P0,gw,bd,Nc,Ref_Int,soc,lw,data,Data_Files,Generated_Files):
    '''
    function to convert crnp_neutron counts to volumetric soil moisture
    '''
    
    import pandas as pd
    import math
    import numpy as np
    import os
    """
    constants
    """       
    a0= 0.0808 #calibration constant
    a1= 0.372 #calibration constant
    a2= 0.115 #calibration constant
   
    """
    functions
    """
    def f_bar(p):
        return math.exp(float(Beta)*(float(p)-float(P0)))
    def ew(t): #SATURATED WATER PRESSURE
        return 6.112*(math.exp((17.62*t)/(243.12+t)))
    def abs_hum(rh,ew,t): #ABSOLUTE HUMIDITY
        return (rh/100)*(ew*k/(t+273.15))
    def f_hum(h): # VAPOR FACTOR
        return 1+(0.0054*h)
    def f_sol(cur_int): # COSMIC RAY INTENSITY (SOLAR) FACTOR
        return Ref_Int/cur_int
    def soil_water_content(N): # N0 CALIBRATION METHOD
        return bd*(((a0/(N/N0-a1)-a2))-lw-soc)
    data.insert(len(data.columns),"fbar",data.P4_mb.apply(f_bar))   
    data.insert(len(data.columns),"Ew",data.T1_C.apply(ew))
    data.insert(len(data.columns),"H",
                np.vectorize(abs_hum)(data.RH1, data.Ew, data.T1_C))
    data.insert(len(data.columns),"fhum",data.H.apply(f_hum))
    NeutronData=pd.read_csv(os.path.join(Generated_Files,"Intensity_Data.csv"), parse_dates=['Timestamp'],date_parser=lambda x: pd.to_datetime(x))
    #Handling missing data of NeutronData Station by Assigning Ref_Int to them 
    #NeutronData.loc[NeutronData['Intensity'] < 40, 'Intensity'] = Ref_Int 
    NeutronData = NeutronData.set_index(pd.DatetimeIndex(NeutronData['Timestamp']))
    NeutronData.drop(["Timestamp"], axis=1,inplace=True)
    intensity_data=NeutronData.reindex(NeutronData.index.union(data.index))# Combining NMDB data with CRNP
    intensity_data.Intensity.fillna(method="ffill", inplace=True)
    intensity_data=intensity_data.loc[intensity_data.index.dropna()]
    intensity_data=intensity_data.reindex(data.index)
    data.insert(len(data.columns),"Intensity",intensity_data.Intensity)
    data.insert(len(data.columns),"fsol",data.Intensity.apply(f_sol))
    data.insert(len(data.columns),"F",data.fsol * data.fhum * data.fbar)  
    N0 = Nc/(a0/(gw+lw+soc+a2)+a1) #Theoretical dry counting rate (counts / h)
    data.insert(len(data.columns),"Ncorr",data.F*data.N1Cts)
    data.insert(len(data.columns),"Water_content",
                data.Ncorr.apply(soil_water_content))
    data.insert(len(data.columns),"Timestamp_12hrMA",data.index)
    #Shifting timestamp to match the moving average
    data.Timestamp_12hrMA=data.Timestamp_12hrMA.shift(5) 
    data.insert(len(data.columns),"Ncorr_12hrMA",data.Ncorr.rolling(window=12,center=False).mean())
    data.insert(len(data.columns),"Water_Content_12hrMA",
                data.Ncorr_12hrMA.apply(soil_water_content)*100)
    # CREATING A NEW DATASET FOR DAILY VOLUMETRIC CRNP SOIL MOISTURE===============
    CRNP_hourly = pd.DataFrame(data=data.Water_Content_12hrMA)
    CRNP_hourly.columns=["CRNP"]
    CRNP_hourly.to_csv(os.path.join(Generated_Files,"Cakit_Hourly_sm.csv"))
#    Neutrons_hourly = pd.DataFrame(data=data.N1Cts)
#    Neutrons_hourly.columns=["Neutrons"]
#    Neutrons_daily_avg = Neutrons_hourly.Neutrons.resample('D', how = 'mean')    
    CRNP = pd.DataFrame(data=data.Water_Content_12hrMA)
    CRNP.Water_Content_12hrMA=CRNP.Water_Content_12hrMA.shift(-6)
    CRNP.dropna(axis=0, inplace=True)
    CRNP_daily_avg = CRNP.Water_Content_12hrMA.resample('D', how = 'mean')
    CRNP_daily_avg.index.rename("Date", inplace=True)  
    CRNP_daily_avg=pd.DataFrame({'Timestamp':CRNP_daily_avg.index, 'SM':CRNP_daily_avg.values})
    CRNP_daily_avg = CRNP_daily_avg.set_index(pd.DatetimeIndex(CRNP_daily_avg['Timestamp']))
    CRNP_daily_avg.drop(["Timestamp"], axis=1,inplace=True)
    CRNP_daily_avg.index.names = ['Timestamp']
    data.to_csv(os.path.join(Generated_Files,"Cakit_Crnp_Calculation_Table.csv"))
    CRNP_daily_avg.to_csv(os.path.join(Generated_Files,"Cakit_Daily_sm.csv"))
#    CRNP_daily_avg_ascending=CRNP.Water_Content_12hrMA[CRNP.Water_Content_12hrMA.index.hour<=9].resample('D').mean()
#    CRNP_daily_avg_ascending.index.rename("Date", inplace=True)  
#    CRNP_daily_avg_ascending=pd.DataFrame({'Timestamp':CRNP_daily_avg_ascending.index, 'SM':CRNP_daily_avg_ascending.values})
#    CRNP_daily_avg_ascending = CRNP_daily_avg_ascending.set_index(pd.DatetimeIndex(CRNP_daily_avg_ascending['Timestamp']))
#    CRNP_daily_avg_ascending.drop(["Timestamp"], axis=1,inplace=True)
#    CRNP_daily_avg_ascending.index.names = ['Timestamp']
#    CRNP_daily_avg_ascending.to_csv("GeneratedCSVs\DailyAverageSM_CRNP_Ascending.csv")
    
    

def graph_sm_data(data,filename):
    import matplotlib.pyplot as plt
    import os
    fig = plt.figure()
    fig.set_size_inches(9,6)
    ax = fig.add_subplot(111)
    ax.set_ylabel('Volumetric Soil Moisture')
    ax.set_xlabel('Date')
    ax.plot(data.Water_Content_12hrMA, "blue", label="Volumetric Soil Moisture(%)")
    ax0 = ax.twinx()
    ax0.plot(data.Ncorr, "red", label="Corrected Neutron Counts")
    ax0.set_ylabel('Neutron Counts')
    ax0.legend(loc="lower right")
    ax.legend(loc="lower left")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(os.path.join(filename,"graph.png"), dpi=300)
    plt.close()


