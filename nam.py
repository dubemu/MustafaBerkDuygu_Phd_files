# region modules


# =============================================================================
# Sorulacak Sorular:
# Update fonksiyonu tam olarak neye yarıyor?
# 
# =============================================================================
#%%
import os

os.chdir(r"C:\Users\musta\Desktop\CAK")
cwd = os.getcwd()

import matplotlib.pyplot as plt
import nam_fun as nam_f
import numpy as np
import objectivefunctions as obj
import pandas as pd
import seaborn
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
from scipy import stats
from scipy.optimize import minimize

# endregion

# pd.plotting.register_matplotlib_converters(explicit=True)
seaborn.set()
np.seterr(all='ignore')


class Nam(object):
#    _dir = r'D:DRIVETUBITAKHydro_ModelDataDarbogaz'
#    _data = "Darbogaz.csv"
    def __init__(self, area, input_parameters, calibration="Q", a=5/7):
        self.a=a
        self.history2 = []
        self.history = []
        self._working_directory = None
        self.Cal_Data_file = None
        self.df = None
        self.P = None
        self.T = None
        self.E = None
        self.SMobs = None
        self.SMobs_val = None
        self.Qobs = None
        self.Qobs_val = None
        self.area = area / (3.6 * 24)
        self.Area = area
        self.Spinoff = 0
        self.parameters = None
        self.Qfit = None
        self.dfh = None
        # self.initial = np.array([10, 100, 0.5, 500, 10, 0.5, 0.5, 0, 2000, 2.15,2])
        # self.initial = np.array([5.59441567e+00,6.85168038e+02,1.30412167e-01,8.47239393e+02,4.00934557e+01,4.21557738e-01,4.88201564e-01,4.09627612e-02,1.67517734e+03,4.09537018e-01,3.71693424e+00])
        self.initial = np.array(input_parameters)
        self.Qsim = None
        self.n = None
        self.Date = None
        self.bounds = (
            (0.01, 50), (0.01, 1000), (0.01, 0.9), (200, 1000), (10,
                                                               50), (0.01, 0.99), (0.01, 0.99), (0.01, 0.99),
            (500, 5000), (0, 4), (-2, 4), (0.6,0.6))
        self.NSE = [None]*4
        self.lNSE = [None]*4
        self.RMSE = [None]*4
        self.PBIAS = [None]*4
        self.R2 = [None]*4
        self.VE = [None]*4
#        FOR KG EFFICIENCY:
        self.r_m = [None]*4
        self.beta_m = [None]*4
        self.alpha_m = [None]*4
        self.KGE = [None]*4
        self.Cal = calibration
        self.statistics = None
        self.export = 'Result.csv'
        self.flowduration = None

    @property
    def process_path(self):
        return self._working_directory

    @process_path.setter
    def process_path(self, value):
        self._working_directory = value
        pass

    def DataRead(self):
        self.df = pd.read_csv(self.Cal_Data_file, sep=',',
                              parse_dates=[0], header=0)
        self.df = self.df.set_index('Date')


    def InitData(self):
        self.P = self.df.P
        self.T = self.df.Temp
        self.E = self.df.E
        self.SMobs = self.df.SWI_cal
        self.SMobs_val = self.df.SWI_val        
        self.Qobs = self.df.Q_cal
        self.Qobs_val = self.df.Q_val
        self.n = self.df.__len__()
        self.Qsim = np.zeros(self.n)
        self.SM = np.zeros(self.n)
        self.Date = self.df.index.to_pydatetime()

    def nash(self, qobserved, qsimulated):
        s, e = np.array(qobserved), np.array(qsimulated)
        # s,e=simulation,evaluation
        mean_observed = np.nanmean(e)
        # compute numerator and denominator
        numerator = np.nansum((e - s) ** 2)
        denominator = np.nansum((e - mean_observed) ** 2)
        # compute coefficient
        return 1 - (numerator / denominator)

    def Objective_Q(self, x):
        self.Qsim = nam_f.nam_method(
            x, self.P, self.T, self.E, self.area, self.Spinoff)[0]
        n = obj.nashsutcliffe(self.Qobs, self.Qsim)
        return 1 - n

    def Objective_SM(self, x):
        self.SMsim = nam_f.nam_method(
            x, self.P, self.T, self.E, self.area, self.Spinoff)[1]
        n = obj.nashsutcliffe(self.SMobs, self.SMsim)
        return 1 - n
    
    def Objective_Q_SM(self, x):
        self.Qsim,self.SMsim = nam_f.nam_method(
            x, self.P, self.T, self.E, self.area, self.Spinoff)
        
        Qsim_cal=self.Qsim[0:len(self.Qobs.dropna())]
        Qobs_cal=self.Qobs.dropna()        
        SMsim_cal=self.SMsim[0:len(self.SMobs.dropna())]
        SMobs_cal=self.SMobs.dropna()
        v = self.Qobs[self.Qobs>0.00000000001].min()
        
        mean_obs = np.mean(Qobs_cal)
        mean_sim = np.mean(Qsim_cal)
        
        A = sum((np.log(Qsim_cal + v) - np.log(Qobs_cal + v)) ** 2)
        B = sum((np.log(Qsim_cal + v) - np.log(mean_obs + v)) ** 2)
        FlogNS = A / B
        # transformation parameter
        y = 0.3
        Qsimprime = np.divide(np.power(Qsim_cal + 1, y) - 1, y)
        Qobsprime = np.divide(np.power(Qobs_cal + 1, y) - 1, y)
        mean_obs_prime = np.mean(Qobsprime)
        FboxNS = sum((Qsimprime - Qobsprime) ** 2) / (sum((Qsimprime - mean_obs_prime) ** 2))
        r = np.corrcoef(Qsim_cal, Qobs_cal)[0, 1]
        Qsim_std = np.std(Qsim_cal)
        Qobs_std = np.std(Qobs_cal)
        X = (1 - r) ** 2
        Y = (1 - Qsim_std / Qobs_std) ** 2
        Z = (1 - mean_sim / mean_obs) ** 2
        FKGE = (X + Y + Z) ** 0.5
        Fbias = (max(Qsim_cal.mean() / Qobs_cal.mean(), Qobs_cal.mean() / Qsim_cal.mean()) - 1) ** 2
        Fsm = sum((SMsim_cal - SMobs_cal) ** 2) / (sum((SMsim_cal - SMobs_cal.mean()) ** 2))
        Fq = FlogNS + FboxNS + FKGE + Fbias
        Fj = self.a * Fq + (1 - self.a) * Fsm
        return Fj  


    def run(self):
        self.DataRead()
        self.InitData()
        
        
        def callback_Q(x):
            fobj = self.Objective_Q(x)
            self.history.append(fobj)
            self.history2.append(self.parameters)

        def callback_SM(x):
            fobj = self.Objective_SM(x)
            self.history.append(fobj)
            self.history2.append(self.parameters)

        def callback_Q_SM(x):
            fobj = self.Objective_Q_SM(x)
            self.history.append(fobj)
            self.history2.append(self.parameters)



        
#        if self.Cal=="Q":
#            self.parameters = minimize(self.Objective_Q, self.initial, method='SLSQP', bounds=self.bounds, callback=callback,
#                                       options={'maxiter': 1e6, "ftol":1e-06, 'disp': True})
#            self.Qsim,self.SMsim  = nam_f.nam_method(
#                self.parameters.x, self.P, self.T, self.E, self.area, self.Spinoff)
#            self.parameters = self.parameters.x
#
#        elif self.Cal=="SM":
#            self.parameters = minimize(self.Objective_SM, self.initial, method='SLSQP', bounds=self.bounds,
#                                       options={'maxiter': 1e6, "ftol":1e-06, 'disp': True})
#            self.Qsim,self.SMsim = nam_f.nam_method(
#                self.parameters.x, self.P, self.T, self.E, self.area, self.Spinoff)
#            self.parameters = self.parameters.x
#        elif self.Cal=="Q_SM":
#            self.parameters = minimize(self.Objective_Q_SM, self.initial, method='SLSQP', bounds=self.bounds,
#                                       options={'maxiter': 1e6, "ftol":1e-06, 'disp': True})
#            self.Qsim,self.SMsim = nam_f.nam_method(
#                self.parameters.x, self.P, self.T, self.E, self.area, self.Spinoff)
#            self.parameters = self.parameters.x
#        else:
#            self.Qsim,self.SMsim = nam_f.nam_method(
#                self.initial, self.P, self.T, self.E, self.area, self.Spinoff)
#            self.parameters = self.initial
        if self.Cal=="Q":
            self.parameters = minimize(self.Objective_Q, self.initial, method='SLSQP', bounds=self.bounds, callback=callback_Q,
                                       tol = 1e-9,options={'maxiter': 1e9, 'disp': True})
            self.Qsim,self.SMsim  = nam_f.nam_method(
                self.parameters.x, self.P, self.T, self.E, self.area, self.Spinoff)
            self.parameters = self.parameters.x

        elif self.Cal=="SM":
            self.parameters = minimize(self.Objective_SM, self.initial, method='SLSQP', bounds=self.bounds, callback=callback_SM,
                                       tol = 1e-9,options={'maxiter': 1e9, 'disp': True})
            self.Qsim,self.SMsim = nam_f.nam_method(
                self.parameters.x, self.P, self.T, self.E, self.area, self.Spinoff)
            self.parameters = self.parameters.x
        elif self.Cal=="Q_SM":
            self.parameters = minimize(self.Objective_Q_SM, self.initial, method='SLSQP', bounds=self.bounds, callback=callback_Q_SM,
                                       tol = 1e-9,options={'maxiter': 1e9, 'disp': True})
            self.Qsim,self.SMsim = nam_f.nam_method(
                self.parameters.x, self.P, self.T, self.E, self.area, self.Spinoff)
            self.parameters = self.parameters.x
        else:
            self.Qsim,self.SMsim = nam_f.nam_method(
                self.initial, self.P, self.T, self.E, self.area, self.Spinoff)
            self.parameters = self.initial
            
            
    def update(self):
        fit = self.interpolation()
        self.Qfit = fit(self.Qobs)
        self.df['Qsim'] = self.Qsim
        self.df['Qfit'] = self.Qfit
        self.flowduration = pd.DataFrame()
        self.flowduration['Qsim_x'] = self.flowdur(self.Qsim)[0]
        self.flowduration['Qsim_y'] = self.flowdur(self.Qsim)[1]
        self.flowduration['Qobs_x'] = self.flowdur(self.Qobs)[0]
        self.flowduration['Qobs_y'] = self.flowdur(self.Qobs)[1]
        # self.df.to_csv(os.path.join(self.process_path, self.export), index=True, header=True)

    def stats(self):
        mean_Q_cal = np.mean(self.Qobs)
        mean_Q_val = np.mean(self.Qobs_val)
        mean_SM_cal = np.mean(self.SMobs)
        mean_SM_val = np.mean(self.SMobs_val)
        Qsim_cal=self.Qsim[0:len(self.Qobs.dropna())]
        Qsim_val=self.Qsim[len(self.Qobs.dropna())-1:-1]
        Qobs_cal=self.Qobs.dropna()
        Qobs_val=self.Qobs_val.dropna()
        SMsim_cal=self.SMsim[0:len(self.SMobs.dropna())]
        SMsim_val=self.SMsim[len(self.SMobs.dropna())-1:-1]
        SMobs_cal=self.SMobs.dropna()
        SMobs_val=self.SMobs_val.dropna()      
        
        Q_df_cal=pd.DataFrame(self.Qobs)
        Q_df_cal.insert(len(Q_df_cal.columns),"Q_sim",self.Qsim)
        Q_df_cal = Q_df_cal[Q_df_cal['Q_cal'].notna()]
        Q_df_cal=Q_df_cal.resample("M").mean()
        SM_df_cal=pd.DataFrame(self.SMobs)
        SM_df_cal.insert(len(SM_df_cal.columns),"SM_sim",self.SMsim)
        SM_df_cal = SM_df_cal[SM_df_cal['SWI_cal'].notna()]
        SM_df_cal=SM_df_cal.resample("M").mean()        
        Q_df_val=pd.DataFrame(self.Qobs_val)
        Q_df_val.insert(len(Q_df_val.columns),"Q_sim",self.Qsim)
        Q_df_val = Q_df_val[Q_df_val['Q_val'].notna()]
        Q_df_val=Q_df_val.resample("M").mean()
        SM_df_val=pd.DataFrame(self.SMobs_val)
        SM_df_val.insert(len(SM_df_val.columns),"SM_sim",self.SMsim)
        SM_df_val = SM_df_val[SM_df_val['SWI_val'].notna()]
        SM_df_val=SM_df_val.resample("M").mean()              

        self.r_m[0]=stats.linregress(Q_df_cal)[2]
        self.r_m[1]=stats.linregress(SM_df_cal)[2]
        self.r_m[2]=stats.linregress(Q_df_val)[2]
        self.r_m[3]=stats.linregress(SM_df_val)[2]

        self.beta_m[0]=Q_df_cal[Q_df_cal.columns[1]].mean()/Q_df_cal[Q_df_cal.columns[0]].mean()
        self.beta_m[1]=SM_df_cal[SM_df_cal.columns[1]].mean()/SM_df_cal[SM_df_cal.columns[0]].mean()
        self.beta_m[2]=Q_df_val[Q_df_val.columns[1]].mean()/Q_df_val[Q_df_val.columns[0]].mean()
        self.beta_m[3]=SM_df_val[SM_df_val.columns[1]].mean()/SM_df_val[SM_df_val.columns[0]].mean()

        self.alpha_m[0]=Q_df_cal[Q_df_cal.columns[1]].std()/Q_df_cal[Q_df_cal.columns[0]].std()
        self.alpha_m[1]=SM_df_cal[SM_df_cal.columns[1]].std()/SM_df_cal[SM_df_cal.columns[0]].std()
        self.alpha_m[2]=Q_df_val[Q_df_val.columns[1]].std()/Q_df_val[Q_df_val.columns[0]].std()
        self.alpha_m[3]=SM_df_val[SM_df_val.columns[1]].std()/SM_df_val[SM_df_val.columns[0]].std()

        self.KGE[0]=1-(((self.r_m[0]-1)**2+(self.beta_m[0]-1)**2+(self.alpha_m[0]-1)**2)**(0.5))
        self.KGE[1]=1-(((self.r_m[1]-1)**2+(self.beta_m[1]-1)**2+(self.alpha_m[1]-1)**2)**(0.5))
        self.KGE[2]=1-(((self.r_m[2]-1)**2+(self.beta_m[2]-1)**2+(self.alpha_m[2]-1)**2)**(0.5))
        self.KGE[3]=1-(((self.r_m[3]-1)**2+(self.beta_m[3]-1)**2+(self.alpha_m[3]-1)**2)**(0.5))

        self.VE[0] = (Qsim_cal.sum()-Qobs_cal.sum())/(Qobs_cal.sum())
        self.VE[1] = (SMsim_cal.sum()-SMobs_cal.sum())/(SMobs_cal.sum())
        self.VE[2] = (Qsim_val.sum()-Qobs_val.sum())/(Qobs_val.sum())
        self.VE[3] = (SMsim_val.sum()-SMobs_val.sum())/(SMobs_val.sum())

        self.R2[0] = stats.linregress(Qobs_cal, Qsim_cal)[2]
        self.R2[1] = stats.linregress(SMobs_cal, SMsim_cal)[2]
        self.R2[2] = stats.linregress(Qobs_val, Qsim_val)[2]
        self.R2[3] = stats.linregress(SMobs_val, SMsim_val)[2]

        self.NSE[0] = 1 - (np.nansum((Qsim_cal - Qobs_cal) ** 2) /
                        np.nansum((Qobs_cal - mean_Q_cal) ** 2))
        self.NSE[1] = 1 - (np.nansum((SMsim_cal - SMobs_cal) ** 2) /
                        np.nansum((SMobs_cal - mean_SM_cal) ** 2))
        self.NSE[2] = 1 - (np.nansum((Qsim_val - Qobs_val) ** 2) /
                        np.nansum((Qobs_val - mean_Q_val) ** 2))
        self.NSE[3] = 1 - (np.nansum((SMsim_val - SMobs_val) ** 2) /
                        np.nansum((SMobs_val - mean_SM_val) ** 2))

        self.lNSE[0] = 1 - (np.nansum((np.log(Qsim_cal) - np.log(Qobs_cal)) ** 2) /
                        np.nansum((np.log(Qobs_cal) - np.log(mean_Q_cal)) ** 2))
        self.lNSE[1] = 1 - (np.nansum((np.log(SMsim_cal) - np.log(SMobs_cal)) ** 2) /
                        np.nansum((np.log(SMobs_cal) - np.log(mean_SM_cal)) ** 2))
        self.lNSE[2] = 1 - (np.nansum((np.log(Qsim_val) - np.log(Qobs_val)) ** 2) /
                        np.nansum((np.log(Qobs_val) - np.log(mean_Q_val)) ** 2))
        self.lNSE[3] = 1 - (np.nansum((np.log(SMsim_val) - np.log(SMobs_val)) ** 2) /
                        np.nansum((np.log(SMobs_val) - np.log(mean_SM_val)) ** 2))

        self.RMSE[0] = np.sqrt(np.nansum((Qsim_cal - Qobs_cal) ** 2) / len(Qsim_cal))
        self.RMSE[1] = np.sqrt(np.nansum((SMsim_cal - SMobs_cal) ** 2) / len(SMsim_cal))
        self.RMSE[2] = np.sqrt(np.nansum((Qsim_val - Qobs_val) ** 2) / len(Qsim_val))
        self.RMSE[3] = np.sqrt(np.nansum((SMsim_val - SMobs_val) ** 2) / len(SMsim_val))

        self.PBIAS[0] = (np.nansum(Qobs_cal - Qsim_cal) / np.nansum(Qobs_cal)) * 100
        self.PBIAS[1] = (np.nansum(SMobs_cal - SMsim_cal) / np.nansum(SMobs_cal)) * 100
        self.PBIAS[2] = (np.nansum(Qobs_val - Qsim_val) / np.nansum(Qobs_val)) * 100
        self.PBIAS[3] = (np.nansum(SMobs_val - SMsim_val) / np.nansum(SMobs_val)) * 100

#        self.statistics = obj.calculate_all_functions(self.Qobs, self.Qsim)

    def interpolation(self):

        idx = np.isfinite(self.Qobs) & np.isfinite(self.Qsim)
        fit = np.polyfit(self.Qobs[idx], self.Qsim[idx], 1)
        fit_fn = np.poly1d(fit)
        return fit_fn

    def draw(self):
        self.stats()
        fit = self.interpolation()
        self.Qfit = fit(self.Qobs)
        width = 15  # Figure width
        height = 15  # Figure height
        f = plt.figure(figsize=(width, height))
        widths = [2, 2, 2]
        heights = [2, 3, 3, 1]
        gs = GridSpec(ncols=3, nrows=4, figure=f, width_ratios=widths,
                      height_ratios=heights)
        ax1 = f.add_subplot(gs[2, :])
        ax2 = f.add_subplot(gs[0, :], sharex=ax1)
        ax11 = f.add_subplot(gs[1, :])
        ax3 = f.add_subplot(gs[-1, 0])
        ax4 = f.add_subplot(gs[-1, -1])
        ax5 = f.add_subplot(gs[-1, -2])
        color = 'tab:blue'
        ax2.set_ylabel('Precipitation (mm) ', color="black",
                       style='italic', fontweight='bold', labelpad=20, fontsize=13)
        ax2.bar(self.Date, self.df.P, color=color,
                align='center', width=1)
        ax2.tick_params(axis='y', labelcolor="black")
        # ax2.set_ylim(0, max(self.df.P) * 1.1, )
        ax2.set_ylim(max(self.df.P) * 1.1, 0)
        ax2.legend(['Precipitation'])
        color = 'tab:red'
#        ax2.set_title('NAM Simulation', style='italic',
#                      fontweight='bold', fontsize=16)
        ax1.set_ylabel(r'Discharge (m$^3$/s)', color="black",
                       style='italic', fontweight='bold', labelpad=20, fontsize=13)
        ax1.plot(self.Date, self.Qobs, 'b-', self.Date,
                 self.Qobs_val, 'g-', self.Date,
                 self.Qsim, 'r--', linewidth=2.0)
        ax1.tick_params(axis='y', labelcolor="black")
        ax1.tick_params(axis='x', labelrotation=0)
        ax1.set_xlabel('Date', style='italic',
                       fontweight='bold', labelpad=20, fontsize=13)
        ax1.legend(('Observed Run-off', 'Observed Run-off (Validation) ','Simulated Run-off'), loc=2)
        plt.setp(ax2.get_xticklabels(), visible=False)
#        anchored_text = AnchoredText("NSE_Calibration = %.2f\nNSE_Validation = %.2f\nRMSE_Calibration = %.2f\nRMSE_Validation = %.2f\nPBIAS_Calibration = %0.2f\nPBIAS_Validation = %0.2f\nr²_Calibration = %0.2f\nr²_Validation = %0.2f" % (self.NSE[0],self.NSE[2], self.RMSE[0],self.RMSE[2], self.PBIAS[0], self.PBIAS[2], self.R2[0], self.R2[2]),
#                                     loc=1, prop=dict(size=12))
#        ax1.add_artist(anchored_text)


        ax11.set_ylabel(r'Soil Moisture (SWI)', color="black",
                       style='italic', fontweight='bold', labelpad=20, fontsize=13)
        ax11.plot(self.Date, self.SMobs, 'b-', self.Date,
                 self.SMobs_val, 'g-', self.Date,
                 self.SMsim, 'r--', linewidth=2.0)
        ax11.tick_params(axis='y', labelcolor="black")
        ax11.tick_params(axis='x', labelrotation=0)
        ax11.set_xlabel('Date', style='italic',
                       fontweight='bold', labelpad=20, fontsize=13)
        ax11.legend(('Observed Soil Moisture', 'Observed Soil Moisture (Validation)','Simulated Soil Moisture'), loc=2)
        plt.setp(ax2.get_xticklabels(), visible=False)
#        anchored_text_sm = AnchoredText("NSE_Calibration = %.2f\nNSE_Validation = %.2f\nRMSE_Calibration = %.2f\nRMSE_Validation = %.2f\nPBIAS_Calibration = %0.2f\nPBIAS_Validation = %0.2f\nr²_Calibration = %0.2f\nr²_Validation = %0.2f" % (self.NSE[1],self.NSE[3], self.RMSE[1],self.RMSE[3], self.PBIAS[1], self.PBIAS[3], self.R2[1], self.R2[3]),
#                                     loc=1, prop=dict(size=12))
#        ax11.add_artist(anchored_text_sm)


        # plt.subplots_adjust(hspace=0.05)
        ax3.set_title('Flow Duration Curve', fontsize=11, style='italic')
        ax3.set_yscale("log")
        ax3.set_ylabel(r'Discharge (m$^3$/s)', style='italic',
                       fontweight='bold', labelpad=20, fontsize=9)
        ax3.set_xlabel('Percentage Exceedence (%)', style='italic',
                       fontweight='bold', labelpad=20, fontsize=9)
        exceedence, sort, low_percentile, high_percentile = self.flowdur(
            self.Qsim)
        ax3.legend(['Precipitation'])
        ax3.plot(self.flowdur(self.Qsim)[0], self.flowdur(self.Qsim)[1], 'b-', self.flowdur(self.Qobs)[0],
                 self.flowdur(self.Qobs)[1], 'r--')
        # ax3.plot(self.flowdur(self.Qobs)[0], self.flowdur(self.Qobs)[1])
        ax3.legend(('Observed', 'Simulated'),
                   loc="upper right", prop=dict(size=7))

        plt.grid(True, which="minor", ls="-")

        st = stats.linregress(self.Qobs, self.Qsim)
        # ax4.set_yscale("log")
        # ax4.set_xscale("log")
        ax4.set_title('Regression Analysis', fontsize=11, style='italic')
        ax4.set_ylabel(r'Simulated', style='italic',
                       fontweight='bold', labelpad=20, fontsize=9)
        ax4.set_xlabel('Observed', style='italic',
                       fontweight='bold', labelpad=20, fontsize=9)

        # ax4.plot(self.Qobs, fit(self.Qsim), '--k')
        # ax4.scatter(self.Qsim, self.Qobs)
        ax4.plot(self.Qobs, self.Qsim, 'bo', self.Qobs, self.Qfit, '--k')
#        ax4.add_artist(anchored_text)

        self.update()
        self.dfh = self.df.resample('M').mean()
        Date = self.dfh.index.to_pydatetime()
        ax5.set_title('Monthly Mean', fontsize=11, style='italic')
        ax5.set_ylabel(r'Discharge (m$^3$/s)', color="black",
                       style='italic', fontweight='bold', labelpad=20, fontsize=9)
        # ax5.set_xlabel('Date', style='italic', fontweight='bold', labelpad=20, fontsize=9)
        ax5.tick_params(axis='y', labelcolor="black")
        ax5.tick_params(axis='x', labelrotation=45)
        # ax5.set_xlabel('Date', style='italic', fontweight='bold', labelpad=20, fontsize=9)
        ax5.legend(('Observed', 'Simulated'), loc="upper right")
        exceedence, sort, low_percentile, high_percentile = self.flowdur(
            self.Qsim)
        ax5.tick_params(axis='x', labelsize=9)
        ax5.plot(Date, self.dfh.Q, 'b-', Date,
                 self.dfh.Qsim, 'r--', linewidth=2.0)
        ax5.legend(('Observed', 'Simulated'), prop={'size': 7}, loc=1)
        # ax5.plot(dfh.Q)
        # ax5.plot(dfh.Qsim)
        # ax5.legend()
        plt.grid(True, which="minor", ls="-")
        plt.subplots_adjust(hspace=0.03)
        f.tight_layout()
        figure_name=f'./Figures/Calibration_Validation/{name}.png'
        plt.savefig(figure_name, dpi=300)
        plt.close()



    def flowdur(self, x):
        exceedence = np.arange(1., len(np.array(x)) + 1) / len(np.array(x))
        exceedence *= 100
        sort = np.sort(x, axis=0)[::-1]
        low_percentile = np.percentile(sort, 5, axis=0)
        high_percentile = np.percentile(sort, 95, axis=0)
        return exceedence, sort, low_percentile, high_percentile

    def drawflow(self):
        f = plt.figure(figsize=(15, 10))
        ax = f.add_subplot(111)
        # fig, ax = plt.subplots(1, 1)
        ax.set_yscale("log")
        ax.set_ylabel(r'Discharge (m$^3$/s)', style='italic',
                      fontweight='bold', labelpad=20, fontsize=16)
        ax.set_xlabel('Percentage Exceedence (%)', style='italic',
                      fontweight='bold', labelpad=20, fontsize=16)
        exceedence, sort, low_percentile, high_percentile = self.flowdur(
            self.Qsim)
        ax.plot(self.flowdur(self.Qsim)[0], self.flowdur(self.Qsim)[1])
        ax.plot(self.flowdur(self.Qobs)[0], self.flowdur(self.Qobs)[1])
        plt.grid(True, which="minor", ls="-")
        # ax.fill_between(exceedence, low_percentile, high_percentile)
        # plt.show()
        return ax
#%%

params_sm_cakit=[6.96774603e+00, 4.86097436e+02, 6.66252688e-01, 5.42599800e+02,
       2.43817666e+01, 8.21292175e-01, 1.00000425e-02, 1.00001673e-02,
       7.74848669e+02, 9.64177345e-01, 2.06296623e+00,0.6]
params_q_cakit=[7.13331518e+00, 4.93158779e+02, 6.60415057e-01, 5.53811180e+02,
       2.41496948e+01, 8.37687389e-01, 9.70309055e-03, 1.02260119e-02,
       7.60075399e+02, 9.87646434e-01, 1.91849383e+00,0.6]
params_q_sm_cakit=[6.83819065e+00, 4.91689295e+02, 6.72805073e-01, 5.43443247e+02,
       2.45605944e+01, 8.31873344e-01, 1.01688312e-02, 9.95780696e-03,
       7.38433537e+02, 9.70434862e-01, 2.05284677e+00,0.6]

params_q_darbogaz_new=[7.19868240e+00, 4.69945128e+02, 6.66753783e-01, 5.59941397e+02,
       2.44018577e+01, 8.45490934e-01, 9.79186929e-03, 1.03230708e-02,
       7.96510288e+02, 1.00031549e+00, 1.93798110e+00,0.6]
params_sm_darbogaz_new=[6.87469023e+00, 4.83299840e+02, 6.51606270e-01, 5.51381463e+02,
       2.31446965e+01, 8.51957617e-01, 1.00947942e-02, 9.31205632e-03,
       8.87230091e+02, 9.58967652e-01, 2.19686785e+00,0.6]
params_q_sm_darbogaz_new=[6.91756481e+00, 4.97444592e+02, 6.70290127e-01, 5.38071736e+02,
       2.48460137e+01, 8.26984678e-01, 1.04239699e-02, 1.00746565e-02,
       7.36917851e+02, 9.53081794e-01, 2.17979394e+00,0.6]

params_q_sm_darbogaz_smap=[6.87393256e+00, 4.82670155e+02, 6.78720753e-01, 5.45742509e+02,
       2.46989921e+01, 8.38894552e-01, 1.04883389e-02, 1.00239816e-02,
       7.42390306e+02, 9.81604323e-01, 2.10284599e+00,0.6]
params_sm_darbogaz_smap=[6.78593313e+00, 4.90864988e+02, 6.48817941e-01, 5.53981746e+02,
       2.29020968e+01, 8.52911040e-01, 1.02532116e-02, 9.21283049e-03,
       8.96524337e+02, 9.73725497e-01, 2.22109619e+00,0.6]

params_sm_cakit_smap=[7.04059415e+00, 4.62932063e+02, 6.74079576e-01, 5.48977328e+02,
       2.46536173e+01, 8.31101795e-01, 1.00683176e-02, 1.01089104e-02,
       7.84492049e+02, 9.52252546e-01, 2.08541158e+00,0.6]
params_q_sm_cakit_smap=[6.83599411e+00, 4.92092425e+02, 6.72350643e-01, 5.43564861e+02,
       2.45629479e+01, 8.31942149e-01, 1.01717406e-02, 1.04506209e-02,
       7.38162434e+02, 9.71214927e-01, 2.05245138e+00,0.6]

#%%
#SENSITIVITY ANALYSES FOR a
area=121.2 #421 for cakıt 121.2 for darbogaz
params=params_q_sm_darbogaz

calibration_nse_Q=[]
calibration_lnse_Q=[]
calibration_rmse_Q=[]
calibration_pbias_Q=[]
calibration_r2_Q=[]
calibration_ve_Q=[]
calibration_rm_Q=[]
calibration_betam_Q=[]
calibration_alpham_Q=[]
calibration_kge_Q=[]
validation_nse_Q=[]
validation_lnse_Q=[]
validation_rmse_Q=[]
validation_pbias_Q=[]
validation_r2_Q=[]
validation_ve_Q=[]
validation_rm_Q=[]
validation_betam_Q=[]
validation_alpham_Q=[]
validation_kge_Q=[]


calibration_nse_SM=[]
calibration_lnse_SM=[]
calibration_rmse_SM=[]
calibration_pbias_SM=[]
calibration_r2_SM=[]
calibration_ve_SM=[]
calibration_rm_SM=[]
calibration_betam_SM=[]
calibration_alpham_SM=[]
calibration_kge_SM=[]
validation_nse_SM=[]
validation_lnse_SM=[]
validation_rmse_SM=[]
validation_pbias_SM=[]
validation_r2_SM=[]
validation_ve_SM=[]
validation_rm_SM=[]
validation_betam_SM=[]
validation_alpham_SM=[]
validation_kge_SM=[]


trials=np.linspace(0,1,50)
i=0
for item in trials:
    Q_SM = Nam(area, params, calibration="Q_SM", a=item)
    Q_SM.process_path = cwd
    Q_SM.Cal_Data_file = os.path.join(Q_SM.process_path,'Data', "Darbogaz_with_sm.csv")
    Q_SM.run()
    Q_SM.stats()
    i=i+1
    calibration_nse_Q.append(Q_SM.NSE[0])
    calibration_lnse_Q.append(Q_SM.lNSE[0])
    calibration_rmse_Q.append(Q_SM.RMSE[0])
    calibration_pbias_Q.append(Q_SM.PBIAS[0])
    calibration_r2_Q.append(Q_SM.R2[0])
    calibration_ve_Q.append(Q_SM.VE[0])
    calibration_rm_Q.append(Q_SM.r_m[0])
    calibration_betam_Q.append(Q_SM.alpha_m[0])
    calibration_alpham_Q.append(Q_SM.beta_m[0])
    calibration_kge_Q.append(Q_SM.KGE[0])
    validation_nse_Q.append(Q_SM.NSE[2])
    validation_lnse_Q.append(Q_SM.lNSE[2])
    validation_rmse_Q.append(Q_SM.RMSE[2])
    validation_pbias_Q.append(Q_SM.PBIAS[2])
    validation_r2_Q.append(Q_SM.R2[2])
    validation_ve_Q.append(Q_SM.VE[2])
    validation_rm_Q.append(Q_SM.r_m[2])
    validation_betam_Q.append(Q_SM.alpha_m[2])
    validation_alpham_Q.append(Q_SM.beta_m[2])
    validation_kge_Q.append(Q_SM.KGE[2])
    calibration_nse_SM.append(Q_SM.NSE[1])
    calibration_lnse_SM.append(Q_SM.lNSE[1])
    calibration_rmse_SM.append(Q_SM.RMSE[1])
    calibration_pbias_SM.append(Q_SM.PBIAS[1])
    calibration_r2_SM.append(Q_SM.R2[1])
    calibration_ve_SM.append(Q_SM.VE[1])
    calibration_rm_SM.append(Q_SM.r_m[1])
    calibration_betam_SM.append(Q_SM.alpha_m[1])
    calibration_alpham_SM.append(Q_SM.beta_m[1])
    calibration_kge_SM.append(Q_SM.KGE[1])
    validation_nse_SM.append(Q_SM.NSE[3])
    validation_lnse_SM.append(Q_SM.lNSE[3])
    validation_rmse_SM.append(Q_SM.RMSE[3])
    validation_pbias_SM.append(Q_SM.PBIAS[3])
    validation_r2_SM.append(Q_SM.R2[3])
    validation_ve_SM.append(Q_SM.VE[3])
    validation_rm_SM.append(Q_SM.r_m[3])
    validation_betam_SM.append(Q_SM.alpha_m[3])
    validation_alpham_SM.append(Q_SM.beta_m[3])
    validation_kge_SM.append(Q_SM.KGE[3])    
    print (i)



stats_sensitivity=pd.DataFrame()
stats_sensitivity.insert(len(stats_sensitivity.columns),"a",trials)
stats_sensitivity.insert(len(stats_sensitivity.columns),"NSE_calibration_Q",calibration_nse_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"lNSE_calibration_Q",calibration_lnse_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"RMSE_calibration_Q",calibration_rmse_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"PBIAS_calibration_Q",calibration_pbias_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"R2_calibration_Q",calibration_r2_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"VE_calibration_Q",calibration_ve_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"RM_calibration_Q",calibration_rm_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"BETAM_calibration_Q",calibration_betam_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"ALPHAM_calibration_Q",calibration_alpham_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"KGE_calibration_Q",calibration_kge_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"NSE_validation_Q",validation_nse_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"lNSE_validation_Q",validation_lnse_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"RMSE_validation_Q",validation_rmse_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"PBIAS_validation_Q",validation_pbias_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"R2_validation_Q",validation_r2_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"VE_validation_Q",validation_ve_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"RM_validation_Q",validation_rm_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"BETAM_validation_Q",validation_betam_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"ALPHAM_validation_Q",validation_alpham_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"KGE_validation_Q",validation_kge_Q)
stats_sensitivity.insert(len(stats_sensitivity.columns),"NSE_calibration_SM",calibration_nse_SM)
stats_sensitivity.insert(len(stats_sensitivity.columns),"lNSE_calibration_SM",calibration_lnse_SM)
stats_sensitivity.insert(len(stats_sensitivity.columns),"RMSE_calibration_SM",calibration_rmse_SM)
stats_sensitivity.insert(len(stats_sensitivity.columns),"PBIAS_calibration_SM",calibration_pbias_SM)
stats_sensitivity.insert(len(stats_sensitivity.columns),"R2_calibration_SM",calibration_r2_SM)
stats_sensitivity.insert(len(stats_sensitivity.columns),"VE_calibration_SM",calibration_ve_SM)
stats_sensitivity.insert(len(stats_sensitivity.columns),"RM_calibration_SM",calibration_rm_SM)
stats_sensitivity.insert(len(stats_sensitivity.columns),"BETAM_calibration_SM",calibration_betam_SM)
stats_sensitivity.insert(len(stats_sensitivity.columns),"ALPHAM_calibration_SM",calibration_alpham_SM)
stats_sensitivity.insert(len(stats_sensitivity.columns),"KGE_calibration_SM",calibration_kge_SM)
stats_sensitivity.insert(len(stats_sensitivity.columns),"NSE_validation_SM",validation_nse_SM)
stats_sensitivity.insert(len(stats_sensitivity.columns),"lNSE_validation_SM",validation_lnse_SM)
stats_sensitivity.insert(len(stats_sensitivity.columns),"RMSE_validation_SM",validation_rmse_SM)
stats_sensitivity.insert(len(stats_sensitivity.columns),"PBIAS_validation_SM",validation_pbias_SM)
stats_sensitivity.insert(len(stats_sensitivity.columns),"R2_validation_SM",validation_r2_SM)
stats_sensitivity.insert(len(stats_sensitivity.columns),"VE_validation_SM",validation_ve_SM)
stats_sensitivity.insert(len(stats_sensitivity.columns),"RM_validation_SM",validation_rm_SM)
stats_sensitivity.insert(len(stats_sensitivity.columns),"BETAM_validation_SM",validation_betam_SM)
stats_sensitivity.insert(len(stats_sensitivity.columns),"ALPHAM_validation_SM",validation_alpham_SM)
stats_sensitivity.insert(len(stats_sensitivity.columns),"KGE_validation_SM",validation_kge_SM)
stats_sensitivity = stats_sensitivity.set_index(stats_sensitivity['a'])
stats_sensitivity.drop(["a"], axis=1,inplace=True)

stats_sensitivity.to_csv(os.path.join(cwd,"Figures","sensitivity","stats_darbogaz.csv"))

#%%
area=121.2
params=params_sm_darbogaz_new
SM_darbogaz = Nam(area, params, calibration="SM")
SM_darbogaz.process_path = cwd
SM_darbogaz.Cal_Data_file = os.path.join(SM_darbogaz.process_path,'Data', "Darbogaz90_with_sm.csv")
SM_darbogaz.run()
name="Calibration_SM_darbogaz"
SM_darbogaz.draw()

params=params_q_darbogaz_new
Q_darbogaz = Nam(area, params, calibration="Q")
Q_darbogaz.process_path = cwd
Q_darbogaz.Cal_Data_file = os.path.join(Q_darbogaz.process_path,'Data', "Darbogaz90_with_sm.csv")
Q_darbogaz.run()
name="Calibration_Q_darbogaz"
Q_darbogaz.draw()

params=params_q_sm_darbogaz_90
Q_SM_darbogaz = Nam(area, params, calibration="Q_SM")
Q_SM_darbogaz.process_path = cwd
Q_SM_darbogaz.Cal_Data_file = os.path.join(Q_SM_darbogaz.process_path,'Data', "Darbogaz90_with_sm.csv")
Q_SM_darbogaz.run()
name="Calibration_Q_SM_darbogaz"
Q_SM_darbogaz.draw()

params=params_q_sm_darbogaz_smap_90
Q_SM_darbogaz_smap = Nam(area, params, calibration="Q_SM")
Q_SM_darbogaz_smap.process_path = cwd
Q_SM_darbogaz_smap.Cal_Data_file = os.path.join(Q_SM_darbogaz_smap.process_path,'Data', "Darbogaz_90_with_sm_smap.csv")
Q_SM_darbogaz_smap.run()
name="Calibration_Q_SM_darbogaz_smap"
Q_SM_darbogaz_smap.draw()

params=params_sm_darbogaz_smap
SM_darbogaz_smap = Nam(area, params, calibration="SM")
SM_darbogaz_smap.process_path = cwd
SM_darbogaz_smap.Cal_Data_file = os.path.join(SM_darbogaz_smap.process_path,'Data', "Darbogaz_90_with_sm_smap.csv")
SM_darbogaz_smap.run()
name="Calibration_SM_darbogaz_smap"
SM_darbogaz_smap.draw()

area=421
params=params_sm_cakit
SM_cakit = Nam(area, params, calibration="SM")
SM_cakit.process_path = cwd
SM_cakit.Cal_Data_file = os.path.join(SM_cakit.process_path,'Data', "Cakit_with_sm.csv")
SM_cakit.run()
name="Calibration_SM_cakit"
SM_cakit.draw()

params=params_q_cakit
Q_cakit = Nam(area, params, calibration="Q")
Q_cakit.process_path = cwd
Q_cakit.Cal_Data_file = os.path.join(Q_cakit.process_path,'Data', "Cakit_with_sm.csv")
Q_cakit.run()
name="Calibration_Q_cakit"
Q_cakit.draw()

params=params_q_sm_cakit
Q_SM_cakit = Nam(area, params, calibration="Q_SM")
Q_SM_cakit.process_path = cwd
Q_SM_cakit.Cal_Data_file = os.path.join(Q_SM_cakit.process_path,'Data', "Cakit_with_sm.csv")
Q_SM_cakit.run()
name="Calibration_Q_SM_cakit"
Q_SM_cakit.draw()

params=params_q_sm_cakit_smap
Q_SM_cakit_smap = Nam(area, params, calibration="Q_SM")
Q_SM_cakit_smap.process_path = cwd
Q_SM_cakit_smap.Cal_Data_file = os.path.join(Q_SM_cakit_smap.process_path,'Data', "Cakit_with_sm_smap.csv")
Q_SM_cakit_smap.run()
name="Calibration_Q_SM_cakit_smap"
Q_SM_cakit_smap.draw()

params=params_sm_cakit_smap
SM_cakit_smap = Nam(area, params, calibration="SM")
SM_cakit_smap.process_path = cwd
SM_cakit_smap.Cal_Data_file = os.path.join(SM_cakit_smap.process_path,'Data', "Cakit_with_sm_smap.csv")
SM_cakit_smap.run()
name="Calibration_SM_cakit_smap"
SM_cakit_smap.draw()


#%%
#CAKIT

simulations_cakit=pd.DataFrame()
simulations_cakit.insert(len(simulations_cakit.columns),"SM(Q)",Q_cakit.SMsim)
simulations_cakit.insert(len(simulations_cakit.columns),"SM(SM)",SM_cakit.SMsim)
simulations_cakit.insert(len(simulations_cakit.columns),"SM(Q_SM)",Q_SM_cakit.SMsim)
simulations_cakit.insert(len(simulations_cakit.columns),"Q(Q)",Q_cakit.Qsim)
simulations_cakit.insert(len(simulations_cakit.columns),"Q(SM)",SM_cakit.Qsim)
simulations_cakit.insert(len(simulations_cakit.columns),"Q(Q_SM)",Q_SM_cakit.Qsim)
simulations_cakit.insert(len(simulations_cakit.columns),"SM(SM)_smap",SM_cakit_smap.SMsim)
simulations_cakit.insert(len(simulations_cakit.columns),"SM(Q_SM)_smap",Q_SM_cakit_smap.SMsim)
simulations_cakit.insert(len(simulations_cakit.columns),"Q(SM)_smap",SM_cakit_smap.Qsim)
simulations_cakit.insert(len(simulations_cakit.columns),"Q(Q_SM)_smap",Q_SM_cakit_smap.Qsim)

parameters_df_cakit=pd.DataFrame()
parameters_df_cakit.insert(len(parameters_df_cakit.columns),"Q",Q_cakit.parameters)
parameters_df_cakit.insert(len(parameters_df_cakit.columns),"Q_SM",Q_SM_cakit.parameters)
parameters_df_cakit.insert(len(parameters_df_cakit.columns),"SM",SM_cakit.parameters)
parameters_df_cakit.insert(len(parameters_df_cakit.columns),"Q_SM_smap",Q_SM_cakit_smap.parameters)
parameters_df_cakit.insert(len(parameters_df_cakit.columns),"SM_smap",SM_cakit_smap.parameters)



#history_df_cakit=pd.concat([pd.DataFrame({'Q':Q_cakit.history}),pd.DataFrame({'Q_SM':Q_SM_cakit.history}),pd.DataFrame({'SM':SM_cakit.history})], axis=1)

column_names=["Calibration_Q","Calibration_SM","Validation_Q","Validation_SM"]
stats_df_cakit=pd.DataFrame()
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_NSE",Q_cakit.NSE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_SM_NSE",Q_SM_cakit.NSE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"SM_NSE",SM_cakit.NSE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_SM_NSE_smap",Q_SM_cakit_smap.NSE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"SM_NSE_smap",SM_cakit_smap.NSE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_lNSE",Q_cakit.lNSE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_SM_lNSE",Q_SM_cakit.lNSE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"SM_lNSE",SM_cakit.lNSE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_SM_lNSE_smap",Q_SM_cakit_smap.lNSE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"SM_lNSE_smap",SM_cakit_smap.lNSE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_RMSE",Q_cakit.RMSE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_SM_RMSE",Q_SM_cakit.RMSE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"SM_RMSE",SM_cakit.RMSE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_SM_RMSE_smap",Q_SM_cakit_smap.RMSE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"SM_RMSE_smap",SM_cakit_smap.RMSE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_PBIAS",Q_cakit.PBIAS)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_SM_PBIAS",Q_SM_cakit.PBIAS)
stats_df_cakit.insert(len(stats_df_cakit.columns),"SM_PBIAS",SM_cakit.PBIAS)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_SM_PBIAS_smap",Q_SM_cakit_smap.PBIAS)
stats_df_cakit.insert(len(stats_df_cakit.columns),"SM_PBIAS_smap",SM_cakit_smap.PBIAS)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_R2",Q_cakit.R2)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_SM_R2",Q_SM_cakit.R2)
stats_df_cakit.insert(len(stats_df_cakit.columns),"SM_R2",SM_cakit.R2)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_SM_R2_smap",Q_SM_cakit_smap.R2)
stats_df_cakit.insert(len(stats_df_cakit.columns),"SM_R2_smap",SM_cakit_smap.R2)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_VE",Q_cakit.VE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_SM_VE",Q_SM_cakit.VE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"SM_VE",SM_cakit.VE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_SM_VE_smap",Q_SM_cakit_smap.VE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"SM_VE_smap",SM_cakit_smap.VE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_KGE",Q_cakit.KGE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_SM_KGE",Q_SM_cakit.KGE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"SM_KGE",SM_cakit.KGE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Q_SM_KGE_smap",Q_SM_cakit_smap.KGE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"SM_KGE_smap",SM_cakit_smap.KGE)
stats_df_cakit.insert(len(stats_df_cakit.columns),"Parameter_Type",column_names)
stats_df_cakit = stats_df_cakit.set_index(stats_df_cakit['Parameter_Type'])
stats_df_cakit.drop(["Parameter_Type"], axis=1,inplace=True)

simulations_cakit.to_csv(os.path.join(cwd,"Figures","Calibration_Validation","simulations_cakit.csv"))
parameters_df_cakit.to_csv(os.path.join(cwd,"Figures","Calibration_Validation","parameters_cakit.csv"))
#history_df_cakit.to_csv(os.path.join(cwd,"Figures","Calibration_Validation","history_cakit.csv"))
stats_df_cakit.to_csv(os.path.join(cwd,"Figures","Calibration_Validation","stats_cakit.csv"))


simulations_darbogaz=pd.DataFrame()
simulations_darbogaz.insert(len(simulations_darbogaz.columns),"SM(Q)",Q_darbogaz.SMsim)
simulations_darbogaz.insert(len(simulations_darbogaz.columns),"SM(SM)",SM_darbogaz.SMsim)
simulations_darbogaz.insert(len(simulations_darbogaz.columns),"SM(Q_SM)",Q_SM_darbogaz.SMsim)
simulations_darbogaz.insert(len(simulations_darbogaz.columns),"Q(Q)",Q_darbogaz.Qsim)
simulations_darbogaz.insert(len(simulations_darbogaz.columns),"Q(SM)",SM_darbogaz.Qsim)
simulations_darbogaz.insert(len(simulations_darbogaz.columns),"Q(Q_SM)",Q_SM_darbogaz.Qsim)
simulations_darbogaz.insert(len(simulations_darbogaz.columns),"SM(SM)_smap",SM_darbogaz_smap.SMsim)
simulations_darbogaz.insert(len(simulations_darbogaz.columns),"SM(Q_SM)_smap",Q_SM_darbogaz_smap.SMsim)
simulations_darbogaz.insert(len(simulations_darbogaz.columns),"Q(SM)_smap",SM_darbogaz_smap.Qsim)
simulations_darbogaz.insert(len(simulations_darbogaz.columns),"Q(Q_SM)_smap",Q_SM_darbogaz_smap.Qsim)

parameters_df_darbogaz=pd.DataFrame()
parameters_df_darbogaz.insert(len(parameters_df_darbogaz.columns),"Q",Q_darbogaz.parameters)
parameters_df_darbogaz.insert(len(parameters_df_darbogaz.columns),"Q_SM",Q_SM_darbogaz.parameters)
parameters_df_darbogaz.insert(len(parameters_df_darbogaz.columns),"SM",SM_darbogaz.parameters)
parameters_df_darbogaz.insert(len(parameters_df_darbogaz.columns),"Q_SM_smap",Q_SM_darbogaz_smap.parameters)
parameters_df_darbogaz.insert(len(parameters_df_darbogaz.columns),"SM_smap",SM_darbogaz_smap.parameters)



#history_df_darbogaz=pd.concat([pd.DataFrame({'Q':Q_darbogaz.history}),pd.DataFrame({'Q_SM':Q_SM_darbogaz.history}),pd.DataFrame({'SM':SM_darbogaz.history})], axis=1)

column_names=["Calibration_Q","Calibration_SM","Validation_Q","Validation_SM"]
stats_df_darbogaz=pd.DataFrame()
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_NSE",Q_darbogaz.NSE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_SM_NSE",Q_SM_darbogaz.NSE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"SM_NSE",SM_darbogaz.NSE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_SM_NSE_smap",Q_SM_darbogaz_smap.NSE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"SM_NSE_smap",SM_darbogaz_smap.NSE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_lNSE",Q_darbogaz.lNSE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_SM_lNSE",Q_SM_darbogaz.lNSE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"SM_lNSE",SM_darbogaz.lNSE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_SM_lNSE_smap",Q_SM_darbogaz_smap.lNSE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"SM_lNSE_smap",SM_darbogaz_smap.lNSE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_RMSE",Q_darbogaz.RMSE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_SM_RMSE",Q_SM_darbogaz.RMSE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"SM_RMSE",SM_darbogaz.RMSE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_SM_RMSE_smap",Q_SM_darbogaz_smap.RMSE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"SM_RMSE_smap",SM_darbogaz_smap.RMSE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_PBIAS",Q_darbogaz.PBIAS)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_SM_PBIAS",Q_SM_darbogaz.PBIAS)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"SM_PBIAS",SM_darbogaz.PBIAS)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_SM_PBIAS_smap",Q_SM_darbogaz_smap.PBIAS)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"SM_PBIAS_smap",SM_darbogaz_smap.PBIAS)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_R2",Q_darbogaz.R2)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_SM_R2",Q_SM_darbogaz.R2)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"SM_R2",SM_darbogaz.R2)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_SM_R2_smap",Q_SM_darbogaz_smap.R2)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"SM_R2_smap",SM_darbogaz_smap.R2)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_VE",Q_darbogaz.VE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_SM_VE",Q_SM_darbogaz.VE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"SM_VE",SM_darbogaz.VE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_SM_VE_smap",Q_SM_darbogaz_smap.VE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"SM_VE_smap",SM_darbogaz_smap.VE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_KGE",Q_darbogaz.KGE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_SM_KGE",Q_SM_darbogaz.KGE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"SM_KGE",SM_darbogaz.KGE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Q_SM_KGE_smap",Q_SM_darbogaz_smap.KGE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"SM_KGE_smap",SM_darbogaz_smap.KGE)
stats_df_darbogaz.insert(len(stats_df_darbogaz.columns),"Parameter_Type",column_names)
stats_df_darbogaz = stats_df_darbogaz.set_index(stats_df_darbogaz['Parameter_Type'])
stats_df_darbogaz.drop(["Parameter_Type"], axis=1,inplace=True)

simulations_darbogaz.to_csv(os.path.join(cwd,"Figures","Calibration_Validation","simulations_darbogaz.csv"))
parameters_df_darbogaz.to_csv(os.path.join(cwd,"Figures","Calibration_Validation","parameters_darbogaz.csv"))
#history_df_darbogaz.to_csv(os.path.join(cwd,"Figures","Calibration_Validation","history_darbogaz.csv"))
stats_df_darbogaz.to_csv(os.path.join(cwd,"Figures","Calibration_Validation","stats_darbogaz.csv"))










#dict_keys(['_working_directory', 'Cal_Data_file', 'Val_Data_file', 'df', 'P', 'T', 'E', 'SMobs', 'Qobs', 'area', 'Area', 'Spinoff', 'parameters', 'Qfit', 'dfh', 'initial', 'Qsim', 'n', 'Date', 'bounds', 'NSE', 'RMSE', 'PBIAS', 'Cal', 'statistics', 'export', 'flowduration', 'SM', 'SMsim'])

#compare=pd.DataFrame(index=Q_SM.df.index)
#compare.insert(len(compare.columns),"Q",Q.df.P)
#compare.insert(len(compare.columns),"SM",SM.df.P)
#compare.insert(len(compare.columns),"Q_SM",Q_SM.df.P)
#compare.plot()
#
#compare.mean()





#%%
name="darbogaz_Discharge"

width=12
height=12
f = plt.figure(figsize=(width, height))
plt.rcParams.update({'font.size': 13})
widths = [1]
heights = [1, 2, 2, 2, 2, 2]
gs = GridSpec(ncols=1, nrows=6, figure=f, width_ratios=widths,
              height_ratios=heights)
ax1 = f.add_subplot(gs[0, :])
ax2 = f.add_subplot(gs[1, :], sharex=ax1)
ax3 = f.add_subplot(gs[2, :], sharex=ax1)
ax4 = f.add_subplot(gs[3, :], sharex=ax1)
ax5 = f.add_subplot(gs[4, :], sharex=ax1)
ax6 = f.add_subplot(gs[5, :], sharex=ax1)
ax2.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 
ax3.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 
ax4.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 
ax5.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 
ax6.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 
color = 'tab:blue'
ax1.set_title('Precipitation (mm)', fontweight='bold')
ax1.bar(Q_darbogaz.Date, Q_darbogaz.df.P, color=color,
        align='center', width=1)
ax1.tick_params(axis='y', labelcolor="black")
ax1.set_ylim(max(Q_darbogaz.df.P) * 1.1, 0)

ax2.set_title('Model Calibration with Discharge', fontweight='bold')
ax2.plot(Q_darbogaz.Date, Q_darbogaz.Qobs, 'b-', Q_darbogaz.Date,
         Q_darbogaz.Qobs_val, 'g-', Q_darbogaz.Date,
         Q_darbogaz.Qsim, 'r--', linewidth=2.0)
ax2.tick_params(axis='y', labelcolor="black")
ax2.tick_params(axis='x', labelrotation=0)

ax3.set_title('Model Calibration with Discharge and Soil Moisture (CRNP)', fontweight='bold')
ax3.plot(Q_SM_darbogaz.Date, Q_SM_darbogaz.Qobs, 'b-', Q_SM_darbogaz.Date,
         Q_SM_darbogaz.Qobs_val, 'g-', Q_SM_darbogaz.Date,
         Q_SM_darbogaz.Qsim, 'r--', linewidth=2.0)
ax3.tick_params(axis='y', labelcolor="black")
ax3.tick_params(axis='x', labelrotation=0)

ax4.set_title('Model Calibration with Discharge and Soil Moisture (SMAP)', fontweight='bold')
ax4.plot(Q_SM_darbogaz_smap.Date, Q_SM_darbogaz_smap.Qobs, 'b-', Q_SM_darbogaz_smap.Date,
         Q_SM_darbogaz_smap.Qobs_val, 'g-', Q_SM_darbogaz_smap.Date,
         Q_SM_darbogaz_smap.Qsim, 'r--', linewidth=2.0)
ax4.tick_params(axis='y', labelcolor="black")
ax4.tick_params(axis='x', labelrotation=0)

ax5.set_title('Model Calibration with Soil Moisture (CRNP)', fontweight='bold')
ax5.plot(SM_darbogaz.Date, SM_darbogaz.Qobs, 'b-', SM_darbogaz.Date,
         SM_darbogaz.Qobs_val, 'g-', SM_darbogaz.Date,
         SM_darbogaz.Qsim, 'r--', linewidth=2.0)
ax5.tick_params(axis='y', labelcolor="black")
ax5.tick_params(axis='x', labelrotation=0)

ax6.set_title('Model Calibration with Soil Moisture (SMAP)', fontweight='bold')
ax6.plot(SM_darbogaz_smap.Date, SM_darbogaz_smap.Qobs, 'b-', SM_darbogaz_smap.Date,
         SM_darbogaz_smap.Qobs_val, 'g-', SM_darbogaz_smap.Date,
         SM_darbogaz_smap.Qsim, 'r--', linewidth=2.0)
ax6.tick_params(axis='y', labelcolor="black")
ax6.tick_params(axis='x', labelrotation=0)
ax6.set_xlabel('Date', style='italic', fontweight='bold', labelpad=20)
plt.gcf().text(0, 0.5, 'Discharge (m$^3$/s)',va='center', rotation='vertical', fontweight='bold')

plt.subplots_adjust(wspace=0, hspace=0)
handles, labels = ax3.get_legend_handles_labels()
labels=['Observed Run-off', 'Observed Run-off (Validation) ','Simulated Run-off']
lgd = ax6.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.5),ncol=3)

f.tight_layout()
figure_name=f'./Figures/Calibration_Validation/{name}.png'
plt.savefig(figure_name, dpi=300, bbox_extra_artists=(lgd,))
plt.close()

name="cakit_Discharge"

width=12
height=12
f = plt.figure(figsize=(width, height))
plt.rcParams.update({'font.size': 13})
widths = [1]
heights = [1, 2, 2, 2, 2, 2]
gs = GridSpec(ncols=1, nrows=6, figure=f, width_ratios=widths,
              height_ratios=heights)
ax1 = f.add_subplot(gs[0, :])
ax2 = f.add_subplot(gs[1, :], sharex=ax1)
ax3 = f.add_subplot(gs[2, :], sharex=ax1)
ax4 = f.add_subplot(gs[3, :], sharex=ax1)
ax5 = f.add_subplot(gs[4, :], sharex=ax1)
ax6 = f.add_subplot(gs[5, :], sharex=ax1)
ax2.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 
ax3.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 
ax4.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 
ax5.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 
ax6.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 
color = 'tab:blue'
ax1.set_title('Precipitation (mm)', fontweight='bold')
ax1.bar(Q_cakit.Date, Q_cakit.df.P, color=color,
        align='center', width=1)
ax1.tick_params(axis='y', labelcolor="black")
ax1.set_ylim(max(Q_cakit.df.P) * 1.1, 0)

ax2.set_title('Model Calibration with Discharge', fontweight='bold')
ax2.plot(Q_cakit.Date, Q_cakit.Qobs, 'b-', Q_cakit.Date,
         Q_cakit.Qobs_val, 'g-', Q_cakit.Date,
         Q_cakit.Qsim, 'r--', linewidth=2.0)
ax2.tick_params(axis='y', labelcolor="black")
ax2.tick_params(axis='x', labelrotation=0)

ax3.set_title('Model Calibration with Discharge and Soil Moisture (CRNP)', fontweight='bold')
ax3.plot(Q_SM_cakit.Date, Q_SM_cakit.Qobs, 'b-', Q_SM_cakit.Date,
         Q_SM_cakit.Qobs_val, 'g-', Q_SM_cakit.Date,
         Q_SM_cakit.Qsim, 'r--', linewidth=2.0)
ax3.tick_params(axis='y', labelcolor="black")
ax3.tick_params(axis='x', labelrotation=0)

ax4.set_title('Model Calibration with Discharge and Soil Moisture (SMAP)', fontweight='bold')
ax4.plot(Q_SM_cakit_smap.Date, Q_SM_cakit_smap.Qobs, 'b-', Q_SM_cakit_smap.Date,
         Q_SM_cakit_smap.Qobs_val, 'g-', Q_SM_cakit_smap.Date,
         Q_SM_cakit_smap.Qsim, 'r--', linewidth=2.0)
ax4.tick_params(axis='y', labelcolor="black")
ax4.tick_params(axis='x', labelrotation=0)

ax5.set_title('Model Calibration with Soil Moisture (CRNP)', fontweight='bold')
ax5.plot(SM_cakit.Date, SM_cakit.Qobs, 'b-', SM_cakit.Date,
         SM_cakit.Qobs_val, 'g-', SM_cakit.Date,
         SM_cakit.Qsim, 'r--', linewidth=2.0)
ax5.tick_params(axis='y', labelcolor="black")
ax5.tick_params(axis='x', labelrotation=0)

ax6.set_title('Model Calibration with Soil Moisture (SMAP)', fontweight='bold')
ax6.plot(SM_cakit_smap.Date, SM_cakit_smap.Qobs, 'b-', SM_cakit_smap.Date,
         SM_cakit_smap.Qobs_val, 'g-', SM_cakit_smap.Date,
         SM_cakit_smap.Qsim, 'r--', linewidth=2.0)
ax6.tick_params(axis='y', labelcolor="black")
ax6.tick_params(axis='x', labelrotation=0)
ax6.set_xlabel('Date', style='italic', fontweight='bold', labelpad=20)
plt.gcf().text(0, 0.5, 'Discharge (m$^3$/s)',va='center', rotation='vertical', fontweight='bold')

plt.subplots_adjust(wspace=0, hspace=0)
handles, labels = ax3.get_legend_handles_labels()
labels=['Observed Run-off', 'Observed Run-off (Validation) ','Simulated Run-off']
lgd = ax6.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.5),ncol=3)

f.tight_layout()
figure_name=f'./Figures/Calibration_Validation/{name}.png'
plt.savefig(figure_name, dpi=300, bbox_extra_artists=(lgd,))
plt.close()
#%% Logirthmics:
name="darbogaz_Discharge_log"

width=12
height=12
f = plt.figure(figsize=(width, height))
plt.rcParams.update({'font.size': 13})
widths = [1]
heights = [1, 2, 2, 2, 2, 2]
gs = GridSpec(ncols=1, nrows=6, figure=f, width_ratios=widths,
              height_ratios=heights)
ax1 = f.add_subplot(gs[0, :])
ax2 = f.add_subplot(gs[1, :], sharex=ax1)
ax3 = f.add_subplot(gs[2, :], sharex=ax1)
ax4 = f.add_subplot(gs[3, :], sharex=ax1)
ax5 = f.add_subplot(gs[4, :], sharex=ax1)
ax6 = f.add_subplot(gs[5, :], sharex=ax1)
ax2.set_yscale('log')
ax3.set_yscale('log')
ax4.set_yscale('log')
ax5.set_yscale('log')
ax6.set_yscale('log')
ax2.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 
ax3.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 
ax4.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 
ax5.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 
ax6.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 

color = 'tab:blue'
ax1.set_title('Precipitation (mm)', fontweight='bold')
ax1.bar(Q_darbogaz.Date, Q_darbogaz.df.P, color=color,
        align='center', width=1)
ax1.tick_params(axis='y', labelcolor="black")
ax1.set_ylim(max(Q_darbogaz.df.P) * 1.1, 0)

ax2.set_title('Model Calibration with Discharge', fontweight='bold')
ax2.plot(Q_darbogaz.Date, Q_darbogaz.Qobs, 'b-', Q_darbogaz.Date,
         Q_darbogaz.Qobs_val, 'g-', Q_darbogaz.Date,
         Q_darbogaz.Qsim, 'r--', linewidth=2.0)
ax2.tick_params(axis='y', labelcolor="black")
ax2.tick_params(axis='x', labelrotation=0)

ax3.set_title('Model Calibration with Discharge and Soil Moisture (CRNP)', fontweight='bold')
ax3.plot(Q_SM_darbogaz.Date, Q_SM_darbogaz.Qobs, 'b-', Q_SM_darbogaz.Date,
         Q_SM_darbogaz.Qobs_val, 'g-', Q_SM_darbogaz.Date,
         Q_SM_darbogaz.Qsim, 'r--', linewidth=2.0)
ax3.tick_params(axis='y', labelcolor="black")
ax3.tick_params(axis='x', labelrotation=0)

ax4.set_title('Model Calibration with Discharge and Soil Moisture (SMAP)', fontweight='bold')
ax4.plot(Q_SM_darbogaz_smap.Date, Q_SM_darbogaz_smap.Qobs, 'b-', Q_SM_darbogaz_smap.Date,
         Q_SM_darbogaz_smap.Qobs_val, 'g-', Q_SM_darbogaz_smap.Date,
         Q_SM_darbogaz_smap.Qsim, 'r--', linewidth=2.0)
ax4.tick_params(axis='y', labelcolor="black")
ax4.tick_params(axis='x', labelrotation=0)

ax5.set_title('Model Calibration with Soil Moisture (CRNP)', fontweight='bold')
ax5.plot(SM_darbogaz.Date, SM_darbogaz.Qobs, 'b-', SM_darbogaz.Date,
         SM_darbogaz.Qobs_val, 'g-', SM_darbogaz.Date,
         SM_darbogaz.Qsim, 'r--', linewidth=2.0)
ax5.tick_params(axis='y', labelcolor="black")
ax5.tick_params(axis='x', labelrotation=0)

ax6.set_title('Model Calibration with Soil Moisture (SMAP)', fontweight='bold')
ax6.plot(SM_darbogaz_smap.Date, SM_darbogaz_smap.Qobs, 'b-', SM_darbogaz_smap.Date,
         SM_darbogaz_smap.Qobs_val, 'g-', SM_darbogaz_smap.Date,
         SM_darbogaz_smap.Qsim, 'r--', linewidth=2.0)
ax6.tick_params(axis='y', labelcolor="black")
ax6.tick_params(axis='x', labelrotation=0)
ax6.set_xlabel('Date', style='italic', fontweight='bold', labelpad=20)
plt.gcf().text(0, 0.5, 'Discharge (m$^3$/s)',va='center', rotation='vertical', fontweight='bold')

plt.subplots_adjust(wspace=0, hspace=0)
handles, labels = ax3.get_legend_handles_labels()
labels=['Observed Run-off', 'Observed Run-off (Validation) ','Simulated Run-off']
lgd = ax6.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.5),ncol=3)

f.tight_layout()
figure_name=f'./Figures/Calibration_Validation/{name}.png'
plt.savefig(figure_name, dpi=300, bbox_extra_artists=(lgd,))
plt.close()

name="cakit_Discharge_log"

width=12
height=12
f = plt.figure(figsize=(width, height))
plt.rcParams.update({'font.size': 13})
widths = [1]
heights = [1, 2, 2, 2, 2, 2]
gs = GridSpec(ncols=1, nrows=6, figure=f, width_ratios=widths,
              height_ratios=heights)
ax1 = f.add_subplot(gs[0, :])
ax2 = f.add_subplot(gs[1, :], sharex=ax1)
ax3 = f.add_subplot(gs[2, :], sharex=ax1)
ax4 = f.add_subplot(gs[3, :], sharex=ax1)
ax5 = f.add_subplot(gs[4, :], sharex=ax1)
ax6 = f.add_subplot(gs[5, :], sharex=ax1)
ax2.set_yscale('log')
ax3.set_yscale('log')
ax4.set_yscale('log')
ax5.set_yscale('log')
ax6.set_yscale('log')
ax2.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 
ax3.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 
ax4.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 
ax5.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 
ax6.axvline(x=Q_SM_darbogaz.Date[722], color='k', linestyle='--') 
color = 'tab:blue'
ax1.set_title('Precipitation (mm)', fontweight='bold')
ax1.bar(Q_cakit.Date, Q_cakit.df.P, color=color,
        align='center', width=1)
ax1.tick_params(axis='y', labelcolor="black")
ax1.set_ylim(max(Q_cakit.df.P) * 1.1, 0)

ax2.set_title('Model Calibration with Discharge', fontweight='bold')
ax2.plot(Q_cakit.Date, Q_cakit.Qobs, 'b-', Q_cakit.Date,
         Q_cakit.Qobs_val, 'g-', Q_cakit.Date,
         Q_cakit.Qsim, 'r--', linewidth=2.0)
ax2.tick_params(axis='y', labelcolor="black")
ax2.tick_params(axis='x', labelrotation=0)

ax3.set_title('Model Calibration with Discharge and Soil Moisture (CRNP)', fontweight='bold')
ax3.plot(Q_SM_cakit.Date, Q_SM_cakit.Qobs, 'b-', Q_SM_cakit.Date,
         Q_SM_cakit.Qobs_val, 'g-', Q_SM_cakit.Date,
         Q_SM_cakit.Qsim, 'r--', linewidth=2.0)
ax3.tick_params(axis='y', labelcolor="black")
ax3.tick_params(axis='x', labelrotation=0)

ax4.set_title('Model Calibration with Discharge and Soil Moisture (SMAP)', fontweight='bold')
ax4.plot(Q_SM_cakit_smap.Date, Q_SM_cakit_smap.Qobs, 'b-', Q_SM_cakit_smap.Date,
         Q_SM_cakit_smap.Qobs_val, 'g-', Q_SM_cakit_smap.Date,
         Q_SM_cakit_smap.Qsim, 'r--', linewidth=2.0)
ax4.tick_params(axis='y', labelcolor="black")
ax4.tick_params(axis='x', labelrotation=0)

ax5.set_title('Model Calibration with Soil Moisture (CRNP)', fontweight='bold')
ax5.plot(SM_cakit.Date, SM_cakit.Qobs, 'b-', SM_cakit.Date,
         SM_cakit.Qobs_val, 'g-', SM_cakit.Date,
         SM_cakit.Qsim, 'r--', linewidth=2.0)
ax5.tick_params(axis='y', labelcolor="black")
ax5.tick_params(axis='x', labelrotation=0)

ax6.set_title('Model Calibration with Soil Moisture (SMAP)', fontweight='bold')
ax6.plot(SM_cakit_smap.Date, SM_cakit_smap.Qobs, 'b-', SM_cakit_smap.Date,
         SM_cakit_smap.Qobs_val, 'g-', SM_cakit_smap.Date,
         SM_cakit_smap.Qsim, 'r--', linewidth=2.0)
ax6.tick_params(axis='y', labelcolor="black")
ax6.tick_params(axis='x', labelrotation=0)
ax6.set_xlabel('Date', style='italic', fontweight='bold', labelpad=20)
plt.gcf().text(0, 0.5, 'Discharge (m$^3$/s)',va='center', rotation='vertical', fontweight='bold')

plt.subplots_adjust(wspace=0, hspace=0)
handles, labels = ax3.get_legend_handles_labels()
labels=['Observed Run-off', 'Observed Run-off (Validation) ','Simulated Run-off']
lgd = ax6.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.5),ncol=3)

f.tight_layout()
figure_name=f'./Figures/Calibration_Validation/{name}.png'
plt.savefig(figure_name, dpi=300, bbox_extra_artists=(lgd,))
plt.close()






#%%
#SOIL MOISTURE

name="darbogaz_Soil_Moisture"

width=12
height=12
f = plt.figure(figsize=(width, height))
plt.rcParams.update({'font.size': 13})
widths = [1]
heights = [1, 2, 2, 2, 2, 2]
gs = GridSpec(ncols=1, nrows=6, figure=f, width_ratios=widths,
              height_ratios=heights)
ax1 = f.add_subplot(gs[0, :])
ax2 = f.add_subplot(gs[1, :], sharex=ax1)
ax3 = f.add_subplot(gs[2, :], sharex=ax1)
ax4 = f.add_subplot(gs[3, :], sharex=ax1)
ax5 = f.add_subplot(gs[4, :], sharex=ax1)
ax6 = f.add_subplot(gs[5, :], sharex=ax1)

color = 'tab:blue'
ax1.set_title('Precipitation (mm)', fontweight='bold')
ax1.bar(Q_darbogaz.Date, Q_darbogaz.df.P, color=color,
        align='center', width=1)
ax1.tick_params(axis='y', labelcolor="black")
ax1.set_ylim(max(Q_darbogaz.df.P) * 1.1, 0)

ax2.set_title('Model Calibration with Discharge', fontweight='bold')
ax2.plot(Q_darbogaz.Date, Q_darbogaz.SMobs, 'b-', Q_darbogaz.Date,
         Q_darbogaz.SMobs_val, 'g-', Q_darbogaz.Date,
         Q_darbogaz.SMsim, 'r--', linewidth=2.0)
ax2.tick_params(axis='y', labelcolor="black")
ax2.tick_params(axis='x', labelrotation=0)

ax3.set_title('Model Calibration with Discharge and Soil Moisture (CRNP)', fontweight='bold')
ax3.plot(Q_SM_darbogaz.Date, Q_SM_darbogaz.SMobs, 'b-', Q_SM_darbogaz.Date,
         Q_SM_darbogaz.SMobs_val, 'g-', Q_SM_darbogaz.Date,
         Q_SM_darbogaz.SMsim, 'r--', linewidth=2.0)
ax3.tick_params(axis='y', labelcolor="black")
ax3.tick_params(axis='x', labelrotation=0)

ax4.set_title('Model Calibration with Discharge and Soil Moisture (SMAP)', fontweight='bold')
ax4.plot(Q_SM_darbogaz_smap.Date, Q_SM_darbogaz_smap.SMobs, 'b-', Q_SM_darbogaz_smap.Date,
         Q_SM_darbogaz_smap.SMobs_val, 'g-', Q_SM_darbogaz_smap.Date,
         Q_SM_darbogaz_smap.SMsim, 'r--', linewidth=2.0)
ax4.tick_params(axis='y', labelcolor="black")
ax4.tick_params(axis='x', labelrotation=0)

ax5.set_title('Model Calibration with Soil Moisture (CRNP)', fontweight='bold')
ax5.plot(SM_darbogaz.Date, SM_darbogaz.SMobs, 'b-', SM_darbogaz.Date,
         SM_darbogaz.SMobs_val, 'g-', SM_darbogaz.Date,
         SM_darbogaz.SMsim, 'r--', linewidth=2.0)
ax5.tick_params(axis='y', labelcolor="black")
ax5.tick_params(axis='x', labelrotation=0)

ax6.set_title('Model Calibration with Soil Moisture (SMAP)', fontweight='bold')
ax6.plot(SM_darbogaz_smap.Date, SM_darbogaz_smap.SMobs, 'b-', SM_darbogaz_smap.Date,
         SM_darbogaz_smap.SMobs_val, 'g-', SM_darbogaz_smap.Date,
         SM_darbogaz_smap.SMsim, 'r--', linewidth=2.0)
ax6.tick_params(axis='y', labelcolor="black")
ax6.tick_params(axis='x', labelrotation=0)
ax6.set_xlabel('Date', style='italic', fontweight='bold', labelpad=20)
plt.gcf().text(0, 0.5, 'Soil Moisture (SWI)',va='center', rotation='vertical', fontweight='bold')

plt.subplots_adjust(wspace=0, hspace=0)
handles, labels = ax3.get_legend_handles_labels()
labels=['Observed Soil Moisture', 'Observed Soil Moisture (Validation) ','Simulated Soil Moisture']
lgd = ax6.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.5),ncol=3)

f.tight_layout()
figure_name=f'./Figures/Calibration_Validation/{name}.png'
plt.savefig(figure_name, dpi=300, bbox_extra_artists=(lgd,))
plt.close()



name="cakit_Soil_Moisture"

width=12
height=12
f = plt.figure(figsize=(width, height))
plt.rcParams.update({'font.size': 13})
widths = [1]
heights = [1, 2, 2, 2, 2, 2]
gs = GridSpec(ncols=1, nrows=6, figure=f, width_ratios=widths,
              height_ratios=heights)
ax1 = f.add_subplot(gs[0, :])
ax2 = f.add_subplot(gs[1, :], sharex=ax1)
ax3 = f.add_subplot(gs[2, :], sharex=ax1)
ax4 = f.add_subplot(gs[3, :], sharex=ax1)
ax5 = f.add_subplot(gs[4, :], sharex=ax1)
ax6 = f.add_subplot(gs[5, :], sharex=ax1)

color = 'tab:blue'
ax1.set_title('Precipitation (mm)', fontweight='bold')
ax1.bar(Q_cakit.Date, Q_cakit.df.P, color=color,
        align='center', width=1)
ax1.tick_params(axis='y', labelcolor="black")
ax1.set_ylim(max(Q_cakit.df.P) * 1.1, 0)

ax2.set_title('Model Calibration with Discharge', fontweight='bold')
ax2.plot(Q_cakit.Date, Q_cakit.SMobs, 'b-', Q_cakit.Date,
         Q_cakit.SMobs_val, 'g-', Q_cakit.Date,
         Q_cakit.SMsim, 'r--', linewidth=2.0)
ax2.tick_params(axis='y', labelcolor="black")
ax2.tick_params(axis='x', labelrotation=0)

ax3.set_title('Model Calibration with Discharge and Soil Moisture (CRNP)', fontweight='bold')
ax3.plot(Q_SM_cakit.Date, Q_SM_cakit.SMobs, 'b-', Q_SM_cakit.Date,
         Q_SM_cakit.SMobs_val, 'g-', Q_SM_cakit.Date,
         Q_SM_cakit.SMsim, 'r--', linewidth=2.0)
ax3.tick_params(axis='y', labelcolor="black")
ax3.tick_params(axis='x', labelrotation=0)

ax4.set_title('Model Calibration with Discharge and Soil Moisture (SMAP)', fontweight='bold')
ax4.plot(Q_SM_cakit_smap.Date, Q_SM_cakit_smap.SMobs, 'b-', Q_SM_cakit_smap.Date,
         Q_SM_cakit_smap.SMobs_val, 'g-', Q_SM_cakit_smap.Date,
         Q_SM_cakit_smap.SMsim, 'r--', linewidth=2.0)
ax4.tick_params(axis='y', labelcolor="black")
ax4.tick_params(axis='x', labelrotation=0)

ax5.set_title('Model Calibration with Soil Moisture (CRNP)', fontweight='bold')
ax5.plot(SM_cakit.Date, SM_cakit.SMobs, 'b-', SM_cakit.Date,
         SM_cakit.SMobs_val, 'g-', SM_cakit.Date,
         SM_cakit.SMsim, 'r--', linewidth=2.0)
ax5.tick_params(axis='y', labelcolor="black")
ax5.tick_params(axis='x', labelrotation=0)

ax6.set_title('Model Calibration with Soil Moisture (SMAP)', fontweight='bold')
ax6.plot(SM_cakit_smap.Date, SM_cakit_smap.SMobs, 'b-', SM_cakit_smap.Date,
         SM_cakit_smap.SMobs_val, 'g-', SM_cakit_smap.Date,
         SM_cakit_smap.SMsim, 'r--', linewidth=2.0)
ax6.tick_params(axis='y', labelcolor="black")
ax6.tick_params(axis='x', labelrotation=0)
ax6.set_xlabel('Date', style='italic', fontweight='bold', labelpad=20)
plt.gcf().text(0, 0.5, 'Soil Moisture (SWI)',va='center', rotation='vertical', fontweight='bold')

plt.subplots_adjust(wspace=0, hspace=0)
handles, labels = ax3.get_legend_handles_labels()
labels=['Observed Soil Moisture', 'Observed Soil Moisture (Validation) ','Simulated Soil Moisture']
lgd = ax6.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.5),ncol=3)

f.tight_layout()
figure_name=f'./Figures/Calibration_Validation/{name}.png'
plt.savefig(figure_name, dpi=300, bbox_extra_artists=(lgd,))
plt.close()

#%%
name="darbogaz_Discharge_tr"

width=9
height=9
f = plt.figure(figsize=(width, height))
plt.rcParams.update({'font.size': 13})
widths = [1]
heights = [1, 2, 2, 2]
gs = GridSpec(ncols=1, nrows=4, figure=f, width_ratios=widths,
              height_ratios=heights)
ax1 = f.add_subplot(gs[0, :])
ax2 = f.add_subplot(gs[1, :], sharex=ax1)
ax3 = f.add_subplot(gs[2, :], sharex=ax1)

ax5 = f.add_subplot(gs[3, :], sharex=ax1)

color = 'tab:blue'
ax1.set_title('Yağış (mm)', fontweight='bold')
ax1.bar(Q_darbogaz.Date, Q_darbogaz.df.P, color=color,
        align='center', width=1)
ax1.tick_params(axis='y', labelcolor="black")
ax1.set_ylim(max(Q_darbogaz.df.P) * 1.1, 0)

ax2.set_title('Akış ile Kalibrasyon', fontweight='bold')
ax2.plot(Q_darbogaz.Date, Q_darbogaz.Qobs, 'b-', Q_darbogaz.Date,
         Q_darbogaz.Qobs_val, 'g-', Q_darbogaz.Date,
         Q_darbogaz.Qsim, 'r--', linewidth=2.0)
ax2.tick_params(axis='y', labelcolor="black")
ax2.tick_params(axis='x', labelrotation=0)

ax3.set_title('Akış ve Toprak Nemi ile Kalibrasyon', fontweight='bold')
ax3.plot(Q_SM_darbogaz.Date, Q_SM_darbogaz.Qobs, 'b-', Q_SM_darbogaz.Date,
         Q_SM_darbogaz.Qobs_val, 'g-', Q_SM_darbogaz.Date,
         Q_SM_darbogaz.Qsim, 'r--', linewidth=2.0)
ax3.tick_params(axis='y', labelcolor="black")
ax3.tick_params(axis='x', labelrotation=0)

ax5.set_title('Toprak Nemi ile Kalibrasyon', fontweight='bold')
ax5.plot(SM_darbogaz.Date, SM_darbogaz.Qobs, 'b-', SM_darbogaz.Date,
         SM_darbogaz.Qobs_val, 'g-', SM_darbogaz.Date,
         SM_darbogaz.Qsim, 'r--', linewidth=2.0)
ax5.tick_params(axis='y', labelcolor="black")
ax5.tick_params(axis='x', labelrotation=0)

plt.subplots_adjust(wspace=0, hspace=0)
handles, labels = ax3.get_legend_handles_labels()
labels=['Ölçülen Akış (Kalibrasyon)', 'Ölçülen Akış (Validasyon) ','Simüle Edilen Akış']
lgd = ax5.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.5),ncol=3)

f.tight_layout()
figure_name=f'./Figures/Calibration_Validation/{name}.png'
plt.savefig(figure_name, dpi=300, bbox_extra_artists=(lgd,))
plt.close()

name="cakit_Discharge_tr"

width=9
height=9
f = plt.figure(figsize=(width, height))
plt.rcParams.update({'font.size': 13})
widths = [1]
heights = [1, 2, 2, 2]
gs = GridSpec(ncols=1, nrows=4, figure=f, width_ratios=widths,
              height_ratios=heights)
ax1 = f.add_subplot(gs[0, :])
ax2 = f.add_subplot(gs[1, :], sharex=ax1)
ax3 = f.add_subplot(gs[2, :], sharex=ax1)

ax5 = f.add_subplot(gs[3, :], sharex=ax1)

color = 'tab:blue'
ax1.set_title('Yağış (mm)', fontweight='bold')
ax1.bar(Q_cakit.Date, Q_cakit.df.P, color=color,
        align='center', width=1)
ax1.tick_params(axis='y', labelcolor="black")
ax1.set_ylim(max(Q_cakit.df.P) * 1.1, 0)

ax2.set_title('Akış ile Kalibrasyon', fontweight='bold')
ax2.plot(Q_cakit.Date, Q_cakit.Qobs, 'b-', Q_cakit.Date,
         Q_cakit.Qobs_val, 'g-', Q_cakit.Date,
         Q_cakit.Qsim, 'r--', linewidth=2.0)
ax2.tick_params(axis='y', labelcolor="black")
ax2.tick_params(axis='x', labelrotation=0)

ax3.set_title('Akış ve Toprak Nemi ile Kalibrasyon', fontweight='bold')
ax3.plot(Q_SM_cakit.Date, Q_SM_cakit.Qobs, 'b-', Q_SM_cakit.Date,
         Q_SM_cakit.Qobs_val, 'g-', Q_SM_cakit.Date,
         Q_SM_cakit.Qsim, 'r--', linewidth=2.0)
ax3.tick_params(axis='y', labelcolor="black")
ax3.tick_params(axis='x', labelrotation=0)

ax5.set_title('Toprak Nemi ile Kalibrasyon', fontweight='bold')
ax5.plot(SM_cakit.Date, SM_cakit.Qobs, 'b-', SM_cakit.Date,
         SM_cakit.Qobs_val, 'g-', SM_cakit.Date,
         SM_cakit.Qsim, 'r--', linewidth=2.0)
ax5.tick_params(axis='y', labelcolor="black")
ax5.tick_params(axis='x', labelrotation=0)

plt.subplots_adjust(wspace=0, hspace=0)
handles, labels = ax3.get_legend_handles_labels()
labels=['Ölçülen Akış (Kalibrasyon)', 'Ölçülen Akış (Validasyon) ','Simüle Edilen Akış']
lgd = ax5.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.5),ncol=3)

f.tight_layout()
figure_name=f'./Figures/Calibration_Validation/{name}.png'
plt.savefig(figure_name, dpi=300, bbox_extra_artists=(lgd,))
plt.close()

