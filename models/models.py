import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import sts
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

class STS():
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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Data_Pipe():    
    def __init__(self, data, **kwargs):
        super().__init__()
        self.para = kwargs
        self.data = data
        # sales columns
        self.sales_cols = ['vol_A', 'vol_B', 'vol_C']
        # categorical columns
        self.cat_cols = list(set(self.data.columns.values).difference(set(self.sales_cols)))
        self.window = self.para['window']
        self.target = self.para['target']
        self.dtype = self.para['dtype']
        # keep test data untransformed throughout
        self.train, self.test = self.data[self.target].iloc[:-self.window], \
                                self.data[self.target].iloc[-self.window:]
        # keep the initial values for later inverse transform to original data
        self.initial_values = self.data[self.sales_cols].iloc[0]
        # log or sq_root transform first, then scaling to avoid numerical errors
        if self.dtype != None:
            self.train = self.diff_transform(self.train)
        self.scale = self.para['scale']
        self.scale_type = self.para['scale_type']
        # make a copy, for inverse scaling later
        self.train_unscaled = self.train
        if self.scale:
            self.train = self.data_scaler(self.train)           
        # updata numerical columns after adding log and sq diff data
        self.numeric_cols = list(set(self.data.columns.values).difference(set(self.cat_cols)))

    # def data_scaler(self, data, type = 'max_min'):
    #     if type == 'max_min':
    #         scaler = MinMaxScaler()
    #     elif type == 'standard':
    #         scaler = StandardScaler()
    #     else:
    #         raise Exception("Only two scaler available:\
    #                         'max_min' and 'standard' ")
    #     data = data.copy()
    #     for col in self.sales_cols:
    #         scaled = scaler.fit_transform(data[col].to_numpy().reshape(-1, 1)).ravel()
    #         data.loc[:, col] = scaled
    #     return data
    
    def data_scaler(self, data, inverse = False):
        if self.scale_type == 'max_min':
            self.scaler = MinMaxScaler()
        elif self.scale_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise Exception("Only two scaler available:\
                            'max_min' and 'standard' ")
        data_ = data.to_numpy().reshape(-1, 1)
        self.scaler.fit(data_)
        scaled = self.scaler.transform(data_).ravel()
        scaled = pd.Series(scaled, name = data.name, index = data.index)
        return scaled
      
    def diff_transform(self, data):
        """Add 1st order differenced data after log and sq_root transform

        Arguments:
            data {[pd.Series]} -- [incoming raw sales data]

        Returns:
            [pd.Series] -- [raw data plus transformed data]
        """        
        # for col in self.sales_cols:
        #     log_diff = pd.Series(np.log(data[col]).diff(), 
        #                      name = 'log_diff_'+str(col)).fillna(value=0)
        #     sq_diff = pd.Series(np.sqrt(data[col]).diff(),
        #                      name = 'sq_diff_'+str(col)).fillna(value=0)
        #     data = pd.concat([data, log_diff, sq_diff], axis = 1)
            
        if self.dtype == 'log_diff':
            data = pd.Series(np.log(data).diff(), 
                             name = 'log_diff_'+str(data.name),
                             index = data.index).fillna(value=0)
        elif self.dtype == 'sq_diff':
            data = pd.Series(np.sqrt(data).diff(),
                             name = 'sq_diff_'+str(data.name),
                             index = data.index).fillna(value=0)
        else:
            pass
        
        return data
    
    def inverse_diff_transform(self, data, initial_value):
        """
        helper method for recover log and sq diff transformed 
        data back to original values 
        Arguments:
            data {pd.DataFrame} -- dataframe to be transformed

        Returns:
            [pd.DataFrame] -- Transformed data
        """        
        # if self.dtype == 'log_diff':
        #     for col in data.columns:
        #             data[col] = np.exp(data[col].cumsum()+np.log(initial_value))
        # elif self.dtype == 'sq_diff':
        #     for col in data.columns:
        #             data[col] = (data[col].cumsum()+np.sqrt(initial_value))**2
        # else:
        #     pass
        if self.dtype == 'log_diff':
            data = np.exp(data.cumsum()+np.log(initial_value))
        
        elif self.dtype == 'sq_diff':
            data = (data.cumsum()+np.sqrt(initial_value))**2
        
        else:
            pass
        
        return data 
            
                
class Stats_model():
    """Generic time series prediction model template using statsmdodels api
    """    
    def __init__(self, data, **kwargs):
        """This function initialize with train test split according 
        to the designated datatype and future window

        Arguments:
        data [pd.DataFrame] pre-processed data containing all relevant information
        """
        self.para = kwargs
        self.pipe = Data_Pipe(data, **self.para)
        self.dtype = self.pipe.dtype
        self.external = self.para['external']
        try:
            self.target = self.dtype + str('_') + self.para['target']
        except:
            # using original data if dtype is None
            self.target = self.para['target']
        # extract initial values for later inverse transform from log_diff and sq_diff
        self.initial_value = self.pipe.initial_values[self.para['target']]       
        self.train, self.test = self.pipe.train, self.pipe.test
        if self.para['smooth']:
            self.train = self.exp_smooth()
        self.model = None
                 
    def fit_model(self):
        try:
            self.fit = self.model.fit(maxiter=1500)
        except:
            print("Unable to fit with current parameters")
        
    def exp_smooth(self):
        exp_model = ExponentialSmoothing(self.train, seasonal_periods=7, seasonal='mul')
        result = exp_model.fit()
        return result.fittedvalues
                        
    def simple_exp_avg(self, month = 6, smooth_level = 0.2):
        """Exponential averaging the numerical data

        Keyword Arguments:
            month {int} -- Time span for each exponential average (default: {6})
            smooth_level {float} -- Hyperparameter for average decay (default : {0.2})
            smaller value means slower decay (more weight on older data)
        Returns:
            pd.Series -- Exponential averaged sales data
        """        
        span = month * 30
        i = 0
        fitted_data = []
        while i < len(self.train):
            split_data = self.train.iloc[i:i+span]
            es = SimpleExpSmoothing(split_data)
            es_fit = es.fit(smoothing_level=smooth_level, optimized=False)
            fitted_data.append(es_fit.fittedvalues)
            i += span
        smooth_data = pd.concat(fitted_data)
        return smooth_data
        
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
            print('the time series {} is stationary!'.format(self.target))
        elif self.adfoutput['p-value'] > 0.01 and self.kpss_output['p-value'] <= 0.05:
            print('the time series {} is non-stationary!'.format(self.target))
            self.stationarity = 'Non-stationary'
        elif self.adfoutput['p-value'] > 0.01 and self.kpss_output['p-value'] > 0.05:
            self.stationarity = 'Trend-stationary'
            print("the time series {} is trend-stationary".format(self.target))
        else:
            self.stationarity = 'Difference stationary'
            print("the time series {} is difference stationary.".format(self.target))
        
class ARIMA_model(Stats_model):
    
    def __init__(self, data, **kwargs):
        """Build ARIMA model which optionally includes exogenous variables, 
         they can either be categorical variables or sales data of different customers
        Arguments:
            data {pd.DataFrame} -- pre-processed data containing all relevant information
        """        
        super().__init__(data, **kwargs)
        
        try:  
            self.exog_all = self.pipe.data[self.external]
            self.exog_train = self.exog_all[:len(self.train)]
            # if external data belongs to sales data, and the training set gets scaled
            # and diff transformed, then do the same for external set as well
            # Note two transformations don't commute
            if self.external in self.pipe.sales_cols:
                if self.para['scale']:               
                    self.exog_all = self.pipe.data_scaler(self.pipe.diff_transform(self.exog_all))
                    self.exog_train = self.exog_all[:len(self.train)]
                else: # diff_transform does nothing when dtype == None
                    self.exog_all = self.pipe.diff_transform(self.exog_all)
                    self.exog_train = self.exog_all[:len(self.train)]
        except:
            self.exog_train = None
            self.exog_all = None
        
    def build_model(self):
        """Build ARIMA model with 3 hyperparameters
        See https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
        for detailed information
        """        
        self.p = self.para['p']
        self.d = self.para['d']
        self.q = self.para['q']
        self.model = ARIMA(self.train, order=(self.p, self.d, self.q), 
                           exog=self.exog_train)
       
    def get_prediction(self, convert = True):
        """Gather all data, including training, test, prediction, 
        error rate, etc. into one pandas DataFrame  
        
        Note, when d is non-zero, the forecast has to discard the first d-values         

        Keyword Arguments:
            convert {bool} -- [If convert is true, the fitted data will 
            be converted to the original values] (default: {True})

        Returns:
            None
        """        
        # typ='levels'return data with original value instead of differenced data
        fit_data = self.fit.predict(start = self.train.index[self.para['d']],
                                    end = self.test.index[-1],
                                    exog = self.exog_all, typ='levels')     

        fit_data = pd.Series(fit_data, name = 'fit_data')
        # first convert the scaled data back 
        if self.para['scale']:
            self.pipe.scaler.fit(self.pipe.train_unscaled.to_numpy().reshape(-1, 1))
            fit_data = self.pipe.scaler.inverse_transform(fit_data)
            #fit_data[:len(self.train)] = self.pipe.scaler.inverse_transform(fit_data[:len(self.train)])
        # then undo the diff_transform
        # these two operations don't commute
        if convert:
            fit_data = self.pipe.inverse_diff_transform(fit_data, self.initial_value)
        
        fit_data = pd.Series(fit_data, 
                            name = 'fit_data', index = self.pipe.data.index)
        real_data = pd.Series(self.pipe.data[self.para['target']], name='real_data')
        combine_data = pd.concat([fit_data, real_data], axis = 1)
        diff = combine_data['real_data'] - combine_data['fit_data']
        self.error_rate = pd.Series(100 * np.abs(diff/combine_data['real_data']),
                                    name = 'error_rate')
        self.combine_data = pd.concat([combine_data, self.error_rate], axis=1)
        # mse only includes test errors
        self.mse = np.sum(diff.iloc[-len(self.test):]**2)/len(self.test)
        if self.dtype == None:
            # call forecast method for obtaining uncertainty estimation
            self.forecast()
            self.combine_data = pd.concat([self.combine_data, self.forecast_bound], axis=1)
                                
    def forecast(self):
        """Make forecast using fitted model.
        The forecasting window matches the test window.
        Since the forecast values can be obtained from get_prediction() method,
        this method is only for obtaining uncertainty estimation. Note the estimation
        only makes sense if the incoming data is original type, i.e., dtype=None
        Returns:
            None
        """
        window = len(self.test)
        try:
            exog = self.exog_all[-len(self.test):]
        except:
            exog = None
        forecast_data = self.fit.forecast(window, exog = exog)
        self.upper_bound = forecast_data[2][:, 1]
        self.lower_bound = forecast_data[2][:, 0]
        self.std_err = forecast_data[1]
        self.forecast_bound = pd.DataFrame(np.array([self.lower_bound, self.upper_bound, self.std_err]).T, 
                     columns = ['lower_bound', 'upper_bound', 'std_err'],
                     index = self.test.index)
            
    def plot_fitted(self):
        """Use built-in method to plot
        """        
        self.fit.plot_predict()
    
    def plot_data(self, plot_all = True, plot_error = False, convert = True):
        """Plot both the fitted and train data

        Keyword Arguments:
            plot_all {bool} -- [plot includes the training set;
            otherwise only test set and forecast will be plotted] (default: {True})
            convert {bool} -- [convert the data to original scale]
            plot_error {bool} -- [plot includes the error rate] (default: {False})
        """
        try:
            self.combine_data
        except:
            self.get_prediction()
        if plot_all: # plot data of all range
            date_range = self.pipe.data.index
            real_data = self.pipe.data[self.para['target']]
            fit_data = self.combine_data['fit_data']           
        else:
            # plot forecast part only
            date_range = self.test.index
            real_data = self.pipe.data[self.para['target']].iloc[-len(self.test):]
            fit_data = self.combine_data['fit_data'].iloc[-len(self.test):]       
        
        _, ax_1 = plt.subplots(figsize=(14, 7))
        ax_1.plot(date_range, fit_data, 'r',
                  label="Fitted and Forecast Values")
        ax_1.plot(date_range, real_data, label="Real Values")
        if self.dtype == None:
            ax_1.fill_between(self.test.index, self.combine_data['lower_bound'].dropna(), 
                              self.combine_data['upper_bound'].dropna(),                             
                             color='#ADCCFF', label='Prediction bound')
        ax_1.set_title('Sales data of {}'.format(self.para['target']), fontsize=18)
        ax_1.set_xlabel('Date', fontsize=18)
        ax_1.set_ylabel('{}'.format(self.para['target']), fontsize=18)
        ax_1.legend(prop={'size': 15})
        
        if plot_error:
            ax_2 = ax_1.twinx()  # plot error rate using the same x-axis
            ax_2.set_ylabel('Error Rate %', color='orange', fontsize=18)
            ax_2.plot(self.test.index, self.error_rate[-len(self.test):],
                        color='orange', label="Error Rate %")
            ax_2.set_ylim(bottom = 0, top = 30)
            ax_2.legend(prop={'size': 15})
        # plt.savefig('figs/model_{}_type_{}'.format(self.dtype, plot_all), 
        #             dpi=400)
        plt.show()
        
    
        
class STL_model(Stats_model):
    
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
  