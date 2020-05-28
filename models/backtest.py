import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.models import ARIMA_model

class Backtest_ARIMA():
    """We can use either sliding or expanding window method for the backtest 
    """    
    def __init__(self, data, **kwargs):
        """initialize back test parameters

        Arguments:
            data[pd.DataFrame] --- pre-processed data containing all relevant information
            **kwargs {dict}  --- testing parameters            
        """        
        self.para = kwargs
        self.span = self.para['span']
        self.slide = self.para['slide']
        self.dtype = self.para['dtype']
        self.method = self.para['method']
        self.target = self.para['target']
        self.window = self.para['window']
        self.data = data
        self.external = self.para['external']
        assert (self.para['p_max'] + self.para['d_max']
                + self.para['q_max'] !=0), "p, d, q cannot all be zero!"
        self.p_range = range(self.para['p_max']+1)
        self.q_range = range(self.para['q_max']+1)
        if self.dtype == None:
            assert self.para['d_max'] > 0, "d must non zero to avoid stability issue!"
            # make the d_min to be 1 to make data stationary
            self.d_range = range(1, self.para['d_max']+1)
        else:
            self.d_range = range(self.para['d_max']+1)    
    
    def back_test(self):
        """     
        Grid search over p, d, q to find the best
        parameters for the ARIMA model

        Raises:
            TypeError: if the method is not within the two choices

        Returns:
        self.mse [list] -- record the average mse score over all test windows for each
                                parameter combo
        self.aic [list] -- the average aic score
        """        
        self.aic_scores = []
        self.mse_scores = []
        for p in self.p_range:
            for d in self.d_range:
                for q in self.q_range:
                    # avoid undefined model
                    if p + q + d != 0: 
                        i = 0
                        test_para = {'p':p, 'd':d, 'q':d}
                        self.para.update(test_para)
                        params = list(test_para.values())
                        mse_scores = []
                        aic_scores = []
                        # we reserve the last span as test set
                        while i < len(self.data) - self.span: 
                            if self.method == 'slide':
                                split_data = self.data.iloc[i:i+self.span]
                            elif self.method == 'expand':
                                split_data = self.data.iloc[:i+self.span]
                            else:
                                raise TypeError("Only 'slide' and 'expand'\
                                                methods allowed!")
                            model = ARIMA_model(split_data, **self.para)
                            model.build_model()
                            try:
                                model.fit_model()
                                model.get_prediction()
                                mse_scores.append(model.mse)
                                aic_scores.append(model.fit.aic)
                            except:
                                print("Unstable model, skip these parameters:\
                                      {}, {}, {},".format(p, d, q))   
                            i += self.slide
                        # recorde the mean score only
                        self.mse_scores.append([params, np.array(mse_scores).mean()])
                        self.aic_scores.append([params, np.array(aic_scores).mean()])
                    else:
                        continue    
                                               
        return self.mse_scores, self.aic_scores
    
    def model_selection(self):
        """select the best model according to MSE and AIC metrics
        """        
        mse, aic = self.back_test()
        mse, aic = np.array(mse), np.array(aic)
        self.best_params_mse = mse[mse[:, 1].argmin()][0]
        self.best_params_aic = aic[aic[:, 1].argmin()][0]
        print("the best mse model is {} !".format(self.best_params_mse))
        print("the best aic model is {} !".format(self.best_params_aic))