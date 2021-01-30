import pandas as pd
import math
import numpy as np
from scipy import stats
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt
from statistics import mean

data = pd.read_csv('/Users/thomasoman/Desktop/Projects/Bayesian_models_Py/cleaned_data.csv')

null_model_output = pd.read_csv('/Users/thomasoman/Desktop/Projects/Bayesian_models_outputs/null_model_output.csv')

seasonal_model_output = pd.read_csv('/Users/thomasoman/Desktop/Projects/Bayesian_models_outputs/seasonal_model_output.csv')

zonal_model_output = pd.read_csv('/Users/thomasoman/Desktop/Projects/Bayesian_models_outputs/zonal_model_output.csv')

seasonal_zonal_model_output = pd.read_csv('/Users/thomasoman/Desktop/Projects/Bayesian_models_outputs/seasonal_zonal_model_output.csv')

CAR1_model_output = pd.read_csv('/Users/thomasoman/Desktop/Projects/Bayesian_models_outputs/CAR1_model_output.csv')

CAR2_model_output = pd.read_csv('/Users/thomasoman/Desktop/Projects/Bayesian_models_outputs/CAR2_model_output.csv')

KDE_model_output = pd.read_csv('/Users/thomasoman/Desktop/Projects/Bayesian_models_outputs/KDE_model_output.csv')

Hier_model_output = pd.read_csv('/Users/thomasoman/Desktop/Projects/Bayesian_models_outputs/Hier_model_output.csv')

KDE_model_output.columns
# create dataframe with columns DIC, Posterior mean, 95% CI limits for some parameters
def DIC(dataframe):

    posteriormean = np.zeros(len(dataframe.columns))
    deviances = np.zeros(len(dataframe))

    for i in range(len(dataframe.columns)):
        posteriormean[i] = mean(dataframe.iloc[:,i])

    if len(dataframe.columns) == 64:
        for i in range(len(dataframe)):
            month_vec = pd.Series(dataframe.iloc[i,0:12].array)[data['Month'] - 1].to_numpy()
            zone_vec = pd.Series(dataframe.iloc[i,12:63].array)[data['Zone'] - 1].to_numpy()
            deviances[i] = -2*(np.sum(norm.logpdf(np.log(data['Rate'].array + 0.001), loc = month_vec+zone_vec, scale = 1/math.exp(dataframe.iloc[i,63]))))

            deviances_mean = mean(deviances)
            post_mean_month_vec = pd.Series(posteriormean[0:12])[data['Month'] - 1].to_numpy()
            post_mean_zone_vec = pd.Series(posteriormean[12:63])[data['Zone'] - 1].to_numpy()
            dpbar = -2*(np.sum(norm.logpdf(np.log(data['Rate'].array + 0.001), loc = post_mean_month_vec+post_mean_zone_vec, scale = 1/math.exp(posteriormean[63]))))

    elif len(dataframe.columns) == 2:
        for i in range(len(dataframe)):
            deviances[i] = -2*(np.sum(norm.logpdf(np.log(data['Rate'].array + 0.001), loc = dataframe.iloc[i,0], scale = 1/math.exp(dataframe.iloc[i,1]))))

            deviances_mean = mean(deviances)
            dpbar = -2*(np.sum(norm.logpdf(np.log(data['Rate'].array + 0.001), loc = posteriormean[0], scale = 1/math.exp(posteriormean[1]))))

    elif len(dataframe.columns) == 13:
        for i in range(len(dataframe)):
            month_vec = pd.Series(dataframe.iloc[i,0:12].array)[data['Month'] - 1].to_numpy()
            deviances[i] = -2*(np.sum(norm.logpdf(np.log(data['Rate'].array + 0.001), loc = month_vec, scale = 1/math.exp(dataframe.iloc[i,12]))))

            deviances_mean = mean(deviances)
            post_mean_month_vec = pd.Series(posteriormean[0:12])[data['Month'] - 1].to_numpy()
            dpbar = -2*(np.sum(norm.logpdf(np.log(data['Rate'].array + 0.001), loc = post_mean_month_vec, scale = 1/math.exp(posteriormean[12]))))


    elif len(dataframe.columns) == 52:
        for i in range(len(dataframe)):
            zone_vec = pd.Series(dataframe.iloc[i,0:51].array)[data['Zone'] - 1].to_numpy()
            deviances[i] = -2*(np.sum(norm.logpdf(np.log(data['Rate'].array + 0.001), loc = zone_vec, scale = 1/math.exp(dataframe.iloc[i,51]))))

            deviances_mean = mean(deviances)
            post_mean_zone_vec = pd.Series(posteriormean[0:51])[data['Zone'] - 1].to_numpy()
            dpbar = -2*(np.sum(norm.logpdf(np.log(data['Rate'].array + 0.001), loc = post_mean_zone_vec, scale = 1/math.exp(posteriormean[51]))))

    return((2*deviances_mean) - dpbar)

def PosteriorMean(dataframe):
        posteriormean = np.zeros(len(dataframe.columns))
        for i in range(len(dataframe.columns)):
            posteriormean[i] = mean(dataframe.iloc[:,i])
        return(posteriormean)

# all output is the log rate of collisions. we must exp() to find the true rate of collisions

models = [null_model_output, seasonal_model_output, zonal_model_output, seasonal_zonal_model_output, CAR1_model_output, CAR2_model_output, KDE_model_output, Hier_model_output]

model_indexes = ['null model', 'seasonal model' ,'zonal model', 'seasonal and zonal model', 'CAR1 model', 'CAR2 model', 'KDE model', 'Heirarchical model']

columns = ['DIC','Mean collisions at zone 3 in March', '95% confidence interval for collisions at zone 3 in March', 'Mean collisions at zone 15 in June','95% confidence interval for collisions at zone 15 in June', 'Mean collisions at zone 44 in September','95% confidence interval for collisions at zone 44 in September', 'Mean collisions at zone 50 in December','95% confidence interval for collisions at zone 50 in December']

dic_scores = [DIC(x) for x in [null_model_output, seasonal_model_output, zonal_model_output, seasonal_zonal_model_output, CAR1_model_output, CAR2_model_output, KDE_model_output, Hier_model_output]]

# find the rate of collisions in specific month and rate by adding together the individual rates of the parameters

mean_3_march = [np.exp(mean(null_model_output.iloc[:,0])),np.exp(mean(seasonal_model_output.iloc[:,2])), np.exp(mean(zonal_model_output.iloc[:,2]))] + [np.exp(mean(x.iloc[:,2]+x.iloc[:,14])) for x in models[3:]]

CI_3_march = [np.exp((null_model_output.iloc[:,0].quantile(0.025),null_model_output.iloc[:,0].quantile(0.975))),np.exp((seasonal_model_output.iloc[:,2].quantile(0.025),seasonal_model_output.iloc[:,2].quantile(0.975))), np.exp((zonal_model_output.iloc[:,2].quantile(0.025), zonal_model_output.iloc[:,2].quantile(0.975)))] + [np.exp(((x.iloc[:,2]+x.iloc[:,14]).quantile(0.025),(x.iloc[:,2]+x.iloc[:,14]).quantile(0.975))) for x in models[3:]]

mean_15_june = [np.exp(mean(null_model_output.iloc[:,0])),np.exp(mean(seasonal_model_output.iloc[:,2])), np.exp(mean(zonal_model_output.iloc[:,2]))] + [np.exp(mean(x.iloc[:,2]+x.iloc[:,14])) for x in models[3:]]

CI_15_june = [np.exp((null_model_output.iloc[:,0].quantile(0.025),null_model_output.iloc[:,0].quantile(0.975))),np.exp((seasonal_model_output.iloc[:,5].quantile(0.025),seasonal_model_output.iloc[:,5].quantile(0.975))), np.exp((zonal_model_output.iloc[:,14].quantile(0.025), zonal_model_output.iloc[:,14].quantile(0.975)))] + [np.exp(((x.iloc[:,5]+x.iloc[:,26]).quantile(0.025),(x.iloc[:,5]+x.iloc[:,26]).quantile(0.975))) for x in models[3:]]

mean_44_september = [np.exp(mean(null_model_output.iloc[:,0])),np.exp(mean(seasonal_model_output.iloc[:,8])), np.exp(mean(zonal_model_output.iloc[:,43]))] + [np.exp(mean(x.iloc[:,8]+x.iloc[:,55])) for x in models[3:]]

CI_44_september =  [np.exp((null_model_output.iloc[:,0].quantile(0.025),null_model_output.iloc[:,0].quantile(0.975))),np.exp((seasonal_model_output.iloc[:,8].quantile(0.025),seasonal_model_output.iloc[:,8].quantile(0.975))), np.exp((zonal_model_output.iloc[:,43].quantile(0.025), zonal_model_output.iloc[:,43].quantile(0.975)))] + [np.exp(((x.iloc[:,8]+x.iloc[:,55]).quantile(0.025),(x.iloc[:,8]+x.iloc[:,55]).quantile(0.975))) for x in models[3:]]

mean_50_december = [np.exp(mean(null_model_output.iloc[:,0])),np.exp(mean(seasonal_model_output.iloc[:,11])), np.exp(mean(zonal_model_output.iloc[:,49]))] + [np.exp(mean(x.iloc[:,11]+x.iloc[:,61])) for x in models[3:]]

CI_50_december =  [np.exp((null_model_output.iloc[:,0].quantile(0.025),null_model_output.iloc[:,0].quantile(0.975))),np.exp((seasonal_model_output.iloc[:,11].quantile(0.025),seasonal_model_output.iloc[:,11].quantile(0.975))), np.exp((zonal_model_output.iloc[:,49].quantile(0.025), zonal_model_output.iloc[:,49].quantile(0.975)))] + [np.exp(((x.iloc[:,8]+x.iloc[:,61]).quantile(0.025),(x.iloc[:,8]+x.iloc[:,61]).quantile(0.975))) for x in models[3:]]

output_df = pd.DataFrame(data = list(zip(dic_scores, mean_3_march, CI_3_march, mean_15_june, CI_15_june, mean_44_september, CI_44_september, mean_50_december, CI_50_december)), index= model_indexes, columns= columns)
output_df.to_csv('/Users/thomasoman/Desktop/Projects/Bayesian_models_outputs/model_output_statistics.csv', index = False)
