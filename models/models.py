import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import sts
import matplotlib.dates as mdates

class STS_model():
    def __init__(self, obs, external_obs = None):
        #super().__init__()
        self.obs = obs
        self.external_obs = external_obs
        self.day_of_week = None
        self.month_of_yr = None
        self.external = None
        self.residue = None
        self.model = None
                
    def build_model(self, day = True, month = True, res = True, ext = False):
        if day:
            self.day_of_week = self.day_of_week_effect()
        if month:
            self.month_of_yr = self.month_of_yr_effect()
        if ext:
            self.external = self.external_effect()
        if res:
            self.residue = self.residue_effect()
        
        # get rid of None in the list
        list_of_effects = [self.day_of_week, self.month_of_yr, 
                            self.external, self.residue]
        
        list_of_effects = list(filter(None.__ne__, list_of_effects))

        self.model = sts.Sum(components=list_of_effects,
                            observed_time_series=self.obs)
    
    #@tf.function(experimental_compile=True)
    #@tf.function
    def train(self, num_steps = 100, lr = 0.1):
        self.lr = lr
        self.num_steps = num_steps
        #self.optimizer = optimizer
        self.surrogate_posterior = self.variational_posterior()
        self.elbo_loss_curve = \
        tfp.vi.fit_surrogate_posterior(
                    target_log_prob_fn = \
                    self.model.joint_log_prob(observed_time_series = self.obs),
                    surrogate_posterior = self.surrogate_posterior,
                    optimizer = tf.optimizers.Adam(learning_rate = self.lr),
                    num_steps = self.num_steps)
    
        return self.elbo_loss_curve
    
    def variational_posterior(self):
        return tfp.sts.build_factored_surrogate_posterior(model = self.model)
            
    def day_of_week_effect(self):
        effect = sts.Seasonal(
                        num_seasons=7, num_steps_per_season=1,
                        observed_time_series=self.obs,
                        name='day_of_week_effect')
        
        return effect
           
    def month_of_yr_effect(self):
        effect = tfp.sts.Seasonal(
        num_seasons=12,
        num_steps_per_season=[31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
        #drift_scale_prior=tfd.LogNormal(loc=-1., scale=0.1),
        #initial_effect_prior=tfd.Normal(loc=0., scale=5.),
        name='month_of_year')

        return effect
       
    def external_effect(self):
        effect = sts.LinearRegression(
                design_matrix=tf.reshape(self.external_obs 
                                         - np.mean(self.external_obs),
                (-1, 1)), name='external_effect')
        
        return effect
    
    def residue_effect(self, order = 1):
        effect = sts.Autoregressive(
                order=order, observed_time_series=self.obs,
                name='autoregressive_residue')
        
        return effect
    
from statsmodels.tsa.api import ARIMA, SimpleExpSmoothing
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Data_preprocessing():    
    def __init__(self, data, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        # sales columns
        self.sales_cols = ['vol_A', 'vol_B', 'vol_C']
        # rest of the columns
        self.cat_cols = list(set(self.data.columns.values).difference(set(self.sales_col)))
        self.data = self.add_diff(data)
        self.window=  self.kwargs['window']
        self.train, self.test = self.data.iloc[:-window], self.data.iloc[-window:]
        self.scale = self.kwargs['scale']
        self.scale_type = self.kwargs['scale_type']
        if self.scale:
            self.train, self.test = self.data_scaler(self.train, type = self.scale_type), \
                                    self.data_scaler(self.test, type = self.scale_type)
        self.smooth = self.kwargs['smooth']
        if self.smooth:
            self.train = self.exp_avg(self.train)

    def data_scaler(self, data, type = 'max_min'):
        if type == 'max_min':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        scaled_data = []
        for col in self.sales_cols:
            tmp = pd.Series(scaler.fit_transform(data[col].to_numpy().reshape(-1, 1)).ravel(), 
                            name = str(col))
            scaled_data = pd.concat([tmp])
        scaled_data.index = data.index
        return pd.concat([scaled_data, data[self.cat_cols]], axis = 1)
    
    def add_diff(self, data):
        for col in self.sales_cols:
            log_diff = pd.Series(np.log(data[col]).diff(), 
                             name = 'log_diff'+str(col)).fillna(value=0)
            sq_diff = pd.Series(np.sqrt(data[col]).diff(),
                             name = 'sq_diff'+str(col)).fillna(value=0)
            data = pd.concat([data, log_diff, sq_diff], axis = 1)
        return data
            
            
    def exp_avg(self, month = 6, smooth_level = 0.2):
        """Exponential averaging the price data

        Arguments:
        data {pd.DataFrame} -- raw sales data for smoothing

        Keyword Arguments:
            month {int} -- Time span for each exponential average (default: {6})
            smooth_level {float} -- Hyperparameter for average decay (default : {0.2})
            smaller value means slower decay (more weight on older data)
        Returns:
            pd.DataFrame -- Exponential averaged sales data
        """        
        span = month * 30
        i = 0        
        for col in self.sales_cols:
            smooth_data = []
            while i < len(self.train):
                split_data = self.train[col].iloc[i:i+span]
                es = SimpleExpSmoothing(split_data)
                es_fit = es.fit(smoothing_level=smooth_level, optimized=False)
                smooth_data.append(es_fit.fittedvalues)
                i += span
            smooth_data = pd.concat(smooth_data)
            log_diff = pd.Series(np.log(smooth_data).diff()).fillna(value=0)
            sq_diff = pd.Series(np.sqrt(smooth_data).diff()).fillna(value=0)
            dic = {str(col): smooth_data, 'log_diff_'+str(col): log_diff, 
                   'sq_diff_'+str(col): sq_diff}
            data_smooth = pd.concat([pd.DataFrame(data = dic)], axis = 1)
        
        return data_smooth


                
class Stats_model():
    """Generic time series prediction model template using statsmdodels api
    """    
    def __init__(self, data, col = 'vol_A', window = 30, scale = True, smooth = False):
        """This function initialize with train test split according 
        to the designated datatype and future window

        Arguments:
            [pd.DataFrame] sales data
             
        Keyword Arguments:
            col {str} -- [select the desired target data for the time series] (default: {'vol_A'})
            window {int} -- [the future forecast window in days] (default: {30})
            scale {bool} -- [min-max scaler to transform the data] (default: True)
            smooth {bool} -- [Apply simple exponential averaging] (default: False)
        Raises:
            TypeError: [only three datatypes are allowed]
        """
        self.data = data        
        self.train, self.test = self.data.iloc[:-window], self.data.iloc[-window:]
        self.smooth = smooth
        train = self.exp_avg()
        if self.smooth:
            self.train = self.exp_avg()
        self.col = col
        assert self.col in ['vol_A', 'vol_B', 'vol_C'], \
        "Only support three datatypes: 'vol_A', 'vol_B' and 'vol_C'"
        self.train, self.test = self.train[self.col], self.test[self.col]
        self.model = None
                 
    def fit_model(self):
        self.fit = self.model.fit()
        
    def stationary_test(self):
        """Perform two statistical test to 
        determine if the time series is stationary or not
        """
        adftest = adfuller(self.train, autolag='AIC')
        self.adfoutput = pd.Series(adftest[0:4], index=[
            'Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in adftest[4].items():
            self.adfoutput['Critical Value (%s)' % key] = value

        kpsstest = kpss(self.train, regression='c', nlags=None)
        self.kpss_output = pd.Series(kpsstest[0:3], index=[
            'Test Statistic', 'p-value', 'Lags Used'])
        for key, value in kpsstest[3].items():
                self.kpss_output['Critical Value (%s)' % key] = value
        if self.adfoutput['p-value'] <= 0.01 and self.kpss_output['p-value'] > 0.05:
            self.stationarity = 'Stationary'
            print('the time series is stationary!')
        elif self.adfoutput['p-value'] > 0.01 and self.kpss_output['p-value'] <= 0.05:
            print('the time series is non-stationary!')
            self.stationarity = 'Non-stationary'
        elif self.adfoutput['p-value'] > 0.01 and self.kpss_output['p-value'] > 0.05:
            self.stationarity = 'Trend-stationary'
            print("the time series is trend-stationary")
        else:
            self.stationarity = 'Difference stationary'
            print("the time series is difference stationary.")