import numpy as np
import tensorflow as tf
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

        # self.model = sts.Sum(components=[list_of_effects],
        #                     observed_time_series=self.obs)
        self.model = sts.Sum(components=[self.day_of_week,
                                         self.month_of_yr,
                                         self.residue],
                            observed_time_series=self.obs)
    
    #@tf.function(experimental_compile=True)
    #@tf.function
    def train(self, num_steps = 100, lr = 0.1):
        self.lr = lr
        self.num_steps = num_steps
        #self.optimizer = optimizer
        self.surrogate_posterior = self.variational_posterior()
        self.elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
                            target_log_prob_fn = self.model.joint_log_prob(
                            observed_time_series = self.obs),
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

import pandas as pd  
# needs to install pyarrow
df = pd.read_parquet('data/cmm_erdos_bootcamp_2020_timeseries.pq', engine='auto')
df['date_val'] = pd.to_datetime(df['date_val'], yearfirst=True)
df.set_index(['date_val'], inplace = True)
df_new = df.drop(columns = ['calendar_year', 'calendar_month', 'calendar_day'])
train_A, test_A = df_new['volume_A'].loc[:'2018-12-31'], df_new['volume_A'].loc['2019-1-1':]
train_A, test_A = train_A.to_numpy(dtype='float32'), test_A.to_numpy(dtype='float32')
model_A = STS_model(train_A)
model_A.build_model()
model_A.train()