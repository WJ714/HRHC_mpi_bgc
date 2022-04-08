import numpy as np
import xgb4caglar
import xarray as xr

def latent_heat_vaporization(TA):
    """latent_heat_vaporization(TA)

    Latent heat of vaporization as a function of air temperature (deg C).
    Uses the formula: lmbd = (2.501 - 0.00237*Tair)10^6

    Parameters
    ----------
    TA : list or list like
        Air temperature (deg C)

    Returns
    -------
    lambda : list or list like
        Latent heat of vaporization (J kg-1)

    References
    ----------
    - Stull, B., 1988: An Introduction to Boundary Layer Meteorology (p.641)
      Kluwer Academic Publishers, Dordrecht, Netherlands

    - Foken, T, 2008: Micrometeorology. Springer, Berlin, Germany.
    """
    k1   = 2.501
    k2   = 0.00237
    lmbd = ( k1 - k2 * TA ) * 1e+06
    return(lmbd)

def LE_to_ET(LE, TA, nStepsPerDay):
    """LE_to_ET(LE, TA)

    Convert LE (W m-2) to ET (kg m-2 s-1, aka mm s-1).

    Parameters
    ----------
    LE : list or list like
        Latent Energy (W m-2)
    TA : list or list like
        Air temperature (deg C)

    Returns
    -------
    ET : list or list like
        Evapotranspiration (kg m-2 s-1, aka mm s-1)"""
    lmbd = latent_heat_vaporization(TA)
    ET   = (LE/lmbd) * 60*60*(24/nStepsPerDay)
    ET   = ET.assign_attrs(long_name = 'evapotranspiration',
                            units     = 'mm per timestep')
    return(ET)

class HRHC_Correcter():

    def __init__(self, ec, mono=[-1]):
        """[summary]

        Parameters
        ----------
        ec : dataset
            half-houly or hourly dataset
        mono : list
            used to define the monotonic constraints in XGBoost, by default [-1,1]
        """
        self.ec   = ec
        self.mono = mono

    def basemask(self):

        """
        mask4LER(self)

        Mask all non-observed data in dataset.

        Function runs emerged dataset to get observed EC data.
        
        Parameters
        -----------
        
        Return
        -------
        ds : xarray.Dataset
            dataset with partitioning results
        """
        
        na_QC = 15
        if self.ec.agg_code == 'HH':
            na_QC = 3
        if np.all(self.ec.LE_QC == na_QC):
            self.ec["LE_QC"] = self.ec["NEE_QC"]
            
        if np.all(self.ec.H_QC == na_QC):
            self.ec["H_QC"] = self.ec["NEE_QC"]
            
        if np.isnan(self.ec["NETRAD_QC"].values).all() or np.all(self.ec.NETRAD_QC == na_QC):
            self.ec["NETRAD_QC"] = self.ec["NEE_QC"]

        maskQC = (self.ec["LE_QC"]==0) & (self.ec["NETRAD_QC"]==0) & (self.ec["H_QC"]==0)        
#       maskQC = maskQC & (self.ec["G_QC"]==0) ## no need to add G_QC

        mask_base = maskQC & (self.ec.RH >= 0) & (self.ec.ET >= 0) #

        self.mask_base = mask_base


    def xgb_pred(self):

        """
        xgb4LER(self)

        Parameters
        -----------

        Returns:
            [type]: [description]
        """
        le = self.ec["LE"].copy()
        rn = self.ec['NETRAD'].copy()
        H = self.ec['H'].copy()
        G = self.ec['G'].copy()
        
        if G.isnull().all():
            residual = (rn - H) # meaning ignore G if there is no G at all in the dataset
        else:
            residual = (rn - H -G)
        residual[residual<0.1] = np.nan
        le[le<0] = np.nan
        self.ec["EBC"] = le / residual
        
        xvars = ["RH"]
        self.xvars = xvars
        X = self.ec[xvars].to_array().T.values
        self.X = X
        
        y = (le/residual).T.values
        
        mask = (y>0) & (self.mask_base.T.values)
        y[y<=0] = np.nan
        y = np.log(y)
	     
        pred_mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
        mask_na = pred_mask & mask
        allID = np.arange(mask_na.size)
        goodID = allID[mask_na]
        np.random.shuffle(goodID)
        split = goodID.size // 10
        idxTest = np.zeros(mask_na.shape, dtype=bool)
        idxTest[goodID[:split]] = True
        idxTrain = np.zeros(mask_na.shape, dtype=bool)
        idxTrain[goodID[split:]] = True

        LER_obj = xgb4caglar.xgbTrain(X, y, idxTest=idxTest, idxTrain=idxTrain, 
                                        x_mono=np.array(self.mono),
                                        idxPred=mask_na, ntrees=1000, early_stopping_rounds=20, 
                                        trainModel=True, calcFI=False, retrainWithTest=True)

        LER_pred = np.zeros(y.shape)*np.nan
        LER_pred[mask_na] = np.exp(LER_obj.pred())
      
        self.ec['LER_pred'] = (("time"), LER_pred)
        self.ec['mask'] = (("time"), mask_na)
        self.ec["MSE"] = LER_obj.mse
        self.ec["MEF"] = LER_obj.mef

        self.mask_na = mask_na
        self.residual = residual
        self.LER_obj = LER_obj

    def LE_corr(self):

        """
        LE_corr()

        Correct LE via xgb

        Parameters
        ----------
        ec : xarray.Dataset
            Merged dataset with all EC varibles and/or TEA partitioning results (masked to get only observed data)
        mask_na : bool
            output of xgb_pred function
        LER_pred : xarray.Dataset
            XGB predicted LER, dataset with EC coords

        Returns
        -------
        LE :
        imb_model: 
        imb_ref:
        Fcor:
        LEcor:
        """
    
        LE = self.ec["LE"]       

        LER_pred = self.ec['LER_pred'].copy()
        imb_ref= np.nanpercentile(LER_pred, 60)
            
        if imb_ref > LER_pred[self.mask_na][(np.abs(self.ec["RH"][self.mask_na]-0.5)).argmin()]:
            imb_ref = LER_pred[self.mask_na][(np.abs(self.ec["RH"][self.mask_na]-0.5)).argmin()]  
        if imb_ref < LER_pred[self.mask_na][(np.abs(self.ec["RH"][self.mask_na]-0.95)).argmin()]:
            imb_ref = LER_pred[self.mask_na][(np.abs(self.ec["RH"][self.mask_na]-0.95)).argmin()]
        LER_pred[LER_pred>imb_ref]= imb_ref

        self.ec['LER_pred_plot'] = LER_pred.copy()

        imb_model=LER_pred.copy()
        imb_cor = imb_ref/imb_model
        lower_bound = 1
        upper_bound = 999
        Fcor= np.fmin(np.fmax(imb_cor, lower_bound), upper_bound)
        LE_pot = (0.9 * self.ec.NETRAD[self.mask_na]).reindex_like(self.ec.LE)
        LEcor = np.fmin((LE*Fcor), np.fmax(LE,LE_pot)).fillna(LE)
        
        if self.ec.agg_code == 'HH':
            hourlyMask      = xr.DataArray(np.ones(self.ec.time.shape).astype(bool),coords=[self.ec.time],dims=['time'])
            self.nStepsPerDay    = 48
        else:
            hourlyMask      = np.ones(self.ec.time.shape).astype(bool)
            hourlyMask[::2] = False
            hourlyMask      = xr.DataArray(hourlyMask, coords=[self.ec.time], dims=['time'])
            self.nStepsPerDay = 24
            
        ET_cor = LE_to_ET(LEcor, self.ec.TA, self.nStepsPerDay)
        

        self.ec['imb_ref'] = imb_ref
        self.ec['Fcor'] = Fcor
        self.ec['LEcor'] = LEcor
        self.ec['ETcor'] = ET_cor

