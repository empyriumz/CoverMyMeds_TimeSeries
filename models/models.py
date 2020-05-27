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
        self.data = self.add_diff(self.data)
        # updata numerical columns after adding log and sq diff data
        self.numeric_cols = list(set(self.data.columns.values).difference(set(self.cat_cols)))
        self.window = self.para['window']
        # for test data, keep numerical variables only
        self.train, self.test = self.data.iloc[:-self.window], \
                                self.data.drop(columns = self.cat_cols).iloc[-self.window:]
        self.scale = self.para['scale']
        self.scale_type = self.para['scale_type']
        if self.scale:
            self.train, self.test = self.data_scaler(self.train, type = self.scale_type), \
                                    self.data_scaler(self.test, type = self.scale_type)

    def data_scaler(self, data, type = 'max_min'):
        if type == 'max_min':
            scaler = MinMaxScaler()
        elif type == 'standard':
            scaler = StandardScaler()
        else:
            raise Exception("Only two scaler available:\
                            'max_min' and 'standard' ")
        for col in self.numeric_cols:
            tmp = pd.DataFrame(scaler.fit_transform(data[col].to_numpy().reshape(-1, 1)).ravel(), 
                              columns = [str(col)+'_scaled'], index = data.index)
            data = data.join(tmp)
        data_scaled = data.drop(columns = self.numeric_cols)
        return data_scaled
    
    def add_diff(self, data):
        for col in self.sales_cols:
            log_diff = pd.Series(np.log(data[col]).diff(), 
                             name = 'log_diff_'+str(col)).fillna(value=0)
            sq_diff = pd.Series(np.sqrt(data[col]).diff(),
                             name = 'sq_diff_'+str(col)).fillna(value=0)
            data = pd.concat([data, log_diff, sq_diff], axis = 1)
        return data
            
                
class Stats_model():
    """Generic time series prediction model template using statsmdodels api
    """    
    def __init__(self, data, **kwargs):
        #super().__init__(*data, **kwargs)
        """This function initialize with train test split according 
        to the designated datatype and future window

        Arguments:
        data [Data_Pipe object] sales data
             
        Keyword Arguments:
            col {str} -- [select the desired target data for the time series] (default: {'vol_A'})
            window {int} -- [the future forecast window in days] (default: {30})
        Raises:
            TypeError: [only three datatypes are allowed]
        """
        self.para = kwargs           
        self.cat_cols = data.cat_cols
        self.target = self.para['target']
        self.dtype = self.para['dtype']
        self.train, self.test = data.train[self.target], data.test[self.target]
        #self.numeric_cols = list(set(self.train.columns.values).difference(set(self.cat_cols)))
        if self.para['smooth']:
            self.train = self.exp_avg()
        self.model = None
                 
    def fit_model(self):
        try:
            self.fit = self.model.fit()
        except:
            print("Unable to fit with current parameters")
    
    def exp_avg(self, month = 6, smooth_level = 0.2):
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
        super().__init__(data, **kwargs)        
        self.all_data = None
        # exogenous variables which may affect the prediction of target
        # these can either be categorical variabls or sales data of different customers
        try:
            self.exog = data.train[self.para['external']]
        except:
            self.exog = None
        
    def build_model(self):
        """Build ARIMA model with 3 hyperparameters
        See https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
        for detailed information
        Keyword Arguments:
            p {int} -- [number of time lags] (default: {4})
            d {int} -- [degree of differencing] (default: {0})
            q {int} -- [order of the moving-average model] (default: {3})
        """        
        self.p = self.para['p']
        self.d = self.para['d']
        self.q = self.para['q']
        self.model = ARIMA(self.train, order=(self.p, self.d, self.q), exog = self.exog)
       
    # def predict(self):
    #     self.predict = self.model.predict(exog=self.exog, typ='levels')
        
    def forecast(self):
        """Make forecast using fitted model.
        The forecasting window matches the test data.

        Returns:
            [ARIMAResults.forecast] -- [np.array with forecast values and bounds]
        """
        window = len(self.test)
        return self.fit.forecast(window, exog=self.exog)
    
    def fitted_value(self):
        """a workaround for getting the original fitted data from statsmodel api

        Returns:
            [pd.DataFrame] -- [Extracted values from 
            the Figure plus the error rate 
            when fitting the training set]
        """
        # typ='levels' means return data with original value instead of differenced data   
        fit_data = self.fit.predict(exog=self.exog, typ='levels')
        dates = self.train.index
        combine_data = {'train_data': self.train,
                        'fit_data': fit_data}
        return pd.DataFrame(combine_data, index=dates)
        
    def plot_fitted(self):
        """Use built-in method to plot
        """        
        self.fit.plot_predict()
    
    def forecast_test(self):
        """Gather forecast data and test data

        Returns:
            [pd.DataFrame] -- The dataframe contains both test data
            and relevant forecast information: 
                    * forecast values
                    * forecast bounds
                    * error rate
        """        
        pred = self.forecast()
        combine_data = {'train_data': self.test.values,
                'fit_data': pred[0]}
        return pd.DataFrame(combine_data, index=self.test.index)
    
    def gather_all_data(self, convert = True):
        """Gather all data, including training, test, prediction, error rate,
        prediction bounds etc. into one pandas DataFrame               

        Keyword Arguments:
            convert {bool} -- [If convert is true, the fitted data will 
            be converted to the original values] (default: {True})
        """               
        previous_data = self.fitted_value()
        future_data = self.forecast_test()               
        combine_data = pd.concat([previous_data, future_data], axis = 0)
        if convert:
            combine_data = self.convert_data(combine_data)
        
        diff = combine_data['train_data'] - combine_data['fit_data']
        error_rate = pd.DataFrame(100 * np.abs(diff/combine_data['train_data'])
                                ,columns=['error_rate'])
        # mse is only calculated for test data
        self.mse = np.sum(diff.iloc[-len(self.test):]**2)/len(self.test)
        self.all_data = pd.concat([combine_data, error_rate], axis = 1)
            
        return self.all_data
    
    def convert_data(self, data):
        """Convert the transformed data back to original form

        Arguments:
            data {pd.DataFrame} -- dataframe to be transformed

        Returns:
            [pd.DataFrame] -- Transformed data
        """        
        if self.dtype == 'log_diff':
            for col in data.columns:
                    data[col] = np.exp(data[col].cumsum())
        elif self.dtype == 'sq_diff':
            for col in data.columns:
                    data[col] = (data[col].cumsum()+1)**2
            else:
                pass
        return data
        
    
    def plot_data(self, plot_all = True, plot_error = False, convert = True):
        """Plot both the fitted and train data

        Keyword Arguments:
            plot_all {bool} -- [plot includes the training set;
            otherwise only test set and forecast will be plotted] (default: {True})
            convert {bool} -- [convert the data to original scale]
            plot_error {bool} -- [plot includes the error rate] (default: {False})
        """
        self.gather_all_data(convert)
        if plot_all: # plot data of all range
            data = self.all_data
        else:
            # plot forecast part only
            data = self.all_data.iloc[-len(self.test):]
        
        dates = data.index
        _, ax_1 = plt.subplots(figsize=(14, 7))
        ax_1.plot(dates, data['fit_data'],
                  'r', label="Fitted and Forecast Values")
        ax_1.plot(dates, data['train_data'], label="train Values")
        # ax_1.fill_between(dates.values, data['lower_bound'],
        #                   data['upper_bound'], color='#ADCCFF', alpha='0.6')
        ax_1.set_title('S&P 500 Price', fontsize=18)
        ax_1.set_xlabel('Date', fontsize=18, fontfamily='sans-serif')
        ax_1.set_ylabel('Price', fontsize='x-large')
        ax_1.legend(prop={'size': 15})
        
        if plot_error:
            ax_2 = ax_1.twinx()  # plot error rate using the same x-axis
            ax_2.set_ylabel('Error Rate %', color='orange', fontsize=18)
            ax_2.plot(dates, data['error_rate'],
                        color='orange', label="Error Rate %")
            ax_2.set_ylim(bottom = 0, top = 30)
            ax_2.legend(prop={'size': 15})
        # plt.savefig('figs/model_{}_type_{}'.format(self.dtype, plot_all), 
        #             dpi=400)
        plt.show()