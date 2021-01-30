import pandas as pd
import math
import numpy as np
from scipy import stats
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt

# First need to read in the data and clean it to be in the format required for the models
def read_and_clean(datalocation):
    # read in the .txt file
    data = pd.read_csv(datalocation,sep= " " , header = None)
    # create empty arrays for the columns
    month = np.array([])
    site = np.array([])
    rate = np.array([])
    # initalise column position
    column_pos = 1
    # loop over each row
    for rows in range(len(data[1])):
        # loop over each value in the row
        for value in data.iloc[rows, 1:52]:
            # check if the value is missing
            if pd.notnull(value):
                # add the month of this row to month array
                month = np.append(month, data.iloc[rows,0])
                # add column number to the site array
                site = np.append(site, column_pos)
                # add the value to the rate array
                rate = np.append(rate, value)
            # move along column counter
            column_pos += 1
            # if column counter hits 52 reset back to 1
            if column_pos == 52:
                column_pos = 1
    # make dictionary of the cleaned data
    cleaned_data = {'Month' : month, 'Rate':rate, 'Zone': site}
    # create dataframe of the cleaned data
    cleaned_dataframe = pd.DataFrame(data = cleaned_data)
    return(cleaned_dataframe)

data = read_and_clean('/Users/thomasoman/Desktop/Projects/Datasets/florida_acc.txt')
data.to_csv('/Users/thomasoman/Desktop/Projects/Bayesian_models_Py/cleaned_data.csv', index = False)

# Model 1 - The Null Model
 def null_model_logposterior(j, theta, data):
     # initalise parameters
     mu = theta[0]
     tau = math.exp(theta[1])
     # set priors for different j counters
     if j == 0:
         logprior = norm.logpdf(mu, loc = 0, scale = 1000)
     else:
         logprior = norm.logpdf(tau, loc = 0, scale = 1000)
     # calculate loglikelihood
     loglike = np.sum(norm.logpdf(np.log(data['Rate'].array + 0.001), loc = mu, scale = 1/tau))
     # calculate log posterior
     logpost = loglike + logprior
     return(logpost)

# metropolis hasting alg for null model
def null_model_met_hast(iters, inv, data, burn_in, thin):
    # initalise acceptance vector
    acceptance_count = np.zeros(2)
    # initalise output matrix iters rows and 2 columns
    mat = np.zeros((int((iters-burn_in)/thin), 2))
    # initalise start of markov chain
    start = np.zeros(2)
    # set first row of output to start of chain
    mat[0,] = start.copy()
    # keep track of current value of parameters
    current = mat[0,].copy()
    # loop over to get iters number of output rows
    for i in range(2,iters):
        # loop over the number of paramters
        for j in range(2):
            # generate random walk proposal value from a normal dist with mean at previous parameter value and std at inivation parameter
            proposal = norm.rvs(loc = current[j], scale = inv[j], size = 1)[0]
            # calculate logposterior for the current value of paramters
            lpost = null_model_logposterior(j, current, data)
            # set the proposal vector to the current set of paramters
            proposal_vec = current.copy()
            # swap the current parameter with the proposed value
            proposal_vec[j] = proposal
            # calulate the log of the acceptance probability which is the minimum value between 0 and the difference bewetween the log posteriors at the current and proposed values
            logaccprob = null_model_logposterior(j, proposal_vec, data) - lpost
            # generate a random number between 0 and 1 with equal probability
            u = np.random.random()
            # if the log acceptance probability is greater than the log of the random number accept the proposal, set the next row of the output for the parameter to be the proposal value and increase the acceptace count
            if math.log(u) < logaccprob:
                current[j] = proposal
                acceptance_count[j] += 1
                if iters % thin == 0 and i > burn_in:
                    mat[int((i-burn_in)/thin),j] = proposal
            # otherwise make the next row of output for the parameter the same as the current value
            else:
                if iters % thin == 0 and i > burn_in:
                    mat[int((i-burn_in)/thin),j] = current[j].copy()
    # create a dataframe
    null_model_df = pd.DataFrame(data = mat)
    # caclulate the accepance rate by divinding acceptance count by the number of iterations
    null_acceptance_rate = (acceptance_count/iters)*100
    # return the dataframe and acceptance rate
    return(null_model_df, null_acceptance_rate)

null_model_df, acceptance_rate = null_model_met_hast(iters = 50000, inv = np.array([0.025,0.025]), data = data, burn_in = 500, thin = 5)
# look at output dataframe
null_model_df
# check acceptace rate is in acceptable region (20-40)
acceptance_rate
# check trace plots of the seperate parameters to see chain convergence
plt.plot(null_model_df[0])
plt.plot(null_model_df[1])
# save output as a csv file
null_model_df.to_csv('/Users/thomasoman/Desktop/Projects/Bayesian_models_outputs/null_model_output.csv', index = False)

# Model 2 - Seasonal Model
def seasonal_model_logposterior(j, theta, data):
    # initalise paramters in theta
    phi = np.array(theta[0:12])
    tau = math.exp(theta[12])
    phi_vector = phi_vector = pd.Series(phi)[data['Month'] - 1].to_numpy()
    # calculate log priors
    if j <= 11:
        logprior = norm.logpdf(phi[j], loc = 0, scale = 1000)
    else:
        logprior = norm.logpdf(tau, loc = 0, scale = 1000)
    # calculate loglikelihood
    loglike = np.sum(norm.logpdf(np.log(data['Rate'].array + 0.001), loc = phi_vector, scale = 1/tau))
    # calculate log posterior
    logpost = loglike + logprior
    return(logpost)

# metropolis hasting alg for seasonal model
def seasonal_model_met_hast(iters, inv, data, burn_in, thin):
    # initalise acceptance vector
    acceptance_count = np.zeros(13)
    # initalise output matrix iters rows and 13 columns
    mat = np.zeros((int((iters-burn_in)/thin), 13))
    # initalise start of markov chain
    start = np.zeros(13)
    # set first row of output to start of chain
    mat[0,] = start.copy()
    # keep track of current value of parameters
    current = mat[0,].copy()
    # loop over to get iters number of output rows
    for i in range(2,iters):
        # loop over the number of paramters
        for j in range(13):
            # generate random walk proposal value from a normal dist with mean at previous parameter value and std at inivation parameter
            proposal = norm.rvs(loc = current[j], scale = inv[j], size = 1)[0]
            # calculate logposterior for the current value of paramters
            lpost = seasonal_model_logposterior(j, current, data)
            # set the proposal vector to the current set of paramters
            proposal_vec = current.copy()
            # swap the current parameter with the proposed value
            proposal_vec[j] = proposal
            # calulate the log of the acceptance probability which is the minimum value between 0 and the difference bewetween the log posteriors at the current and proposed values
            logaccprob = seasonal_model_logposterior(j, proposal_vec, data) - lpost
            # generate a random number between 0 and 1 with equal probability
            u = np.random.random()
            # if the log acceptance probability is greater than the log of the random number accept the proposal, set the next row of the output for the parameter to be the proposal value and increase the acceptace count
            if math.log(u) < logaccprob:
                current[j] = proposal
                acceptance_count[j] += 1
                if iters % thin == 0 and i > burn_in:
                    mat[int((i-burn_in)/thin),j] = proposal
            # otherwise make the next row of output for the parameter the same as the current value
            else:
                if iters % thin == 0 and i > burn_in:
                    mat[int((i-burn_in)/thin),j] = current[j].copy()

    seasonal_model_df = pd.DataFrame(data = mat)
    # caclulate the accepance rate by divinding acceptance count by the number of iterations
    seasonal_acceptance_rate = (acceptance_count/iters)*100
    # return the dataframe and acceptance rate
    return(seasonal_model_df, seasonal_acceptance_rate)

seasonal_model_df, seasonal_acceptance_rate = seasonal_model_met_hast(iters = 50000, inv = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.05]), data = data, burn_in = 500, thin = 5)
# check output dataframe
seasonal_model_df
# check acceptace rate is in acceptable region (20-40)
seasonal_acceptance_rate
# check trace plots of the seperate parameters to see chain convergence
plt.plot(seasonal_model_df[0])
plt.ylabel('Phi 1')
plt.xlabel('Iterations')
plt.title('Traceplot of Phi 1')
plt.plot(seasonal_model_df[12])
plt.ylabel('Tau')
plt.xlabel('Iterations')
plt.title('Traceplot of Tau')
# save output as a csv file
seasonal_model_df.to_csv('/Users/thomasoman/Desktop/Projects/Bayesian_models_outputs/seasonal_model_output.csv', index = False)

# Model 3 - Zonal Model
def zonal_model_logposterior(j, theta, data):
    # initalise paramters in theta
    sig = np.array(theta[0:51])
    tau = math.exp(theta[51])
    sig_vector = pd.Series(sig)[data['Zone'] - 1].to_numpy()
    # calculate log priors
    if j <= 50:
        logprior = norm.logpdf(sig[j], loc = 0, scale = 1000)
    else:
        logprior = norm.logpdf(tau, loc = 0, scale = 1000)
    # calculate loglikelihood
    loglike = np.sum(norm.logpdf(np.log(data['Rate'].array + 0.001), loc = sig_vector, scale = 1/tau))
    # calculate log posterior
    logpost = loglike + logprior
    return(logpost)

# metropolis hasting alg for zonal model
def zonal_model_met_hast(iters, inv, data, burn_in, thin):
    # initalise acceptance vector
    acceptance_count = np.zeros(52)
    # initalise output matrix iters rows and 13 columns
    mat = np.zeros((int((iters-burn_in)/thin), 52))
    # initalise start of markov chain
    start = np.zeros(52)
    # set first row of output to start of chain
    mat[0,] = start.copy()
    # keep track of current value of parameters
    current = mat[0,].copy()
    # loop over to get iters number of output rows
    for i in range(2,iters):
        # loop over the number of paramters
        for j in range(52):
            # generate random walk proposal value from a normal dist with mean at previous parameter value and std at inivation parameter
            proposal = norm.rvs(loc = current[j], scale = inv[j], size = 1)[0]
            # calculate logposterior for the current value of paramters
            lpost = zonal_model_logposterior(j, current, data)
            # set the proposal vector to the current set of paramters
            proposal_vec = current.copy()
            # swap the current parameter with the proposed value
            proposal_vec[j] = proposal
            # calulate the log of the acceptance probability which is the minimum value between 0 and the difference bewetween the log posteriors at the current and proposed values
            logaccprob = zonal_model_logposterior(j, proposal_vec, data) - lpost
            # generate a random number between 0 and 1 with equal probability
            u = np.random.random()
            # if the log acceptance probability is greater than the log of the random number accept the proposal, set the next row of the output for the parameter to be the proposal value and increase the acceptace count
            if math.log(u) < logaccprob:
                current[j] = proposal
                acceptance_count[j] += 1
                if iters % thin == 0 and i > burn_in:
                    mat[int((i-burn_in)/thin),j] = proposal
            # otherwise make the next row of output for the parameter the same as the current value
            else:
                if iters % thin == 0 and i > burn_in:
                    mat[int((i-burn_in)/thin),j] = current[j].copy()
    # create a dataframe of the output
    zonal_model_df = pd.DataFrame(data = mat)
    # caclulate the accepance rate by divinding acceptance count by the number of iterations
    zonal_acceptance_rate = (acceptance_count/iters)*100
    # return the dataframe and acceptance rate
    return(zonal_model_df, zonal_acceptance_rate)

zonal_model_df, zonal_acceptance_rate = zonal_model_met_hast(iters = 50000, inv = np.append(np.repeat([0.15],51), 0.05), data = data, burn_in = 500, thin = 5)
# check output dataframe
zonal_model_df
# check acceptace rate is in acceptable region (approx 20-40)
zonal_acceptance_rate
# check trace plots of the seperate parameters to see chain convergence
plt.plot(zonal_model_df[0])
plt.ylabel('Sig 1')
plt.xlabel('Iterations')
plt.title('Traceplot of Sig 1')
plt.plot(zonal_model_df[51])
plt.ylabel('Tau')
plt.xlabel('Iterations')
plt.title('Traceplot of Tau')
# save output as a csv file
zonal_model_df.to_csv('/Users/thomasoman/Desktop/Projects/Bayesian_models_outputs/zonal_model_output.csv', index = False)

# Model 4 - Seasonal+Zonal Model
def seasonal_zonal_model_logposterior(j, theta, data):
    # initalise paramters in theta
    phi = np.array(theta[0:12])
    sig = np.array(theta[12:63])
    tau = math.exp(theta[63])
    phi_vector = pd.Series(phi)[data['Month'] - 1].to_numpy()
    sig_vector = pd.Series(sig)[data['Zone'] - 1].to_numpy()
    # calculate log priors
    if j <= 11:
        logprior = norm.logpdf(phi[j], loc = 0, scale = 1000)
    elif j == 63:
        logprior = norm.logpdf(tau, loc = 0, scale = 1000)
    else:
        logprior = norm.logpdf(sig[j-12], loc = 0, scale = 1000)
    # calculate loglikelihood
    loglike = np.sum(norm.logpdf(np.log(data['Rate'].array + 0.001), loc = phi_vector+sig_vector, scale = 1/tau))
    # calculate log posterior
    logpost = loglike + logprior
    return(logpost)

# metropolis hasting alg for seaonal+zonal model
def seasonal_zonal_model_met_hast(iters, inv, data, thin, burn_in):
    # initalise acceptance vector
    acceptance_count = np.zeros(64)
    # initalise output matrix iters rows and 63 columns
    mat = np.zeros((int((iters-burn_in)/thin), 64))
    # initalise start of markov chain
    start = np.zeros(64)
    # set first row of output to start of chain
    mat[0,] = start.copy()
    # keep track of current value of parameters
    current = mat[0,].copy()
    # loop over to get iters number of output rows
    for i in range(2,iters):
        # loop over the number of paramters
        for j in range(1,64):
            # generate random walk proposal value from a normal dist with mean at previous parameter value and std at inivation parameter
            proposal = norm.rvs(loc = current[j], scale = inv[j], size = 1)[0]
            # calculate logposterior for the current value of paramters
            lpost = seasonal_zonal_model_logposterior(j, current, data)
            # set the proposal vector to the current set of paramters
            proposal_vec = current.copy()
            # swap the current parameter with the proposed value
            proposal_vec[j] = proposal
            # calulate the log of the acceptance probability which is the minimum value between 0 and the difference bewetween the log posteriors at the current and proposed values
            logaccprob = seasonal_zonal_model_logposterior(j, proposal_vec, data) - lpost
            # generate a random number between 0 and 1 with equal probability
            u = np.random.random()
            # if the log acceptance probability is greater than the log of the random number accept the proposal, set the next row of the output for the parameter to be the proposal value and increase the acceptace count
            if math.log(u) < logaccprob:
                current[j] = proposal
                acceptance_count[j] += 1
                if iters % thin == 0 and i > burn_in:
                    mat[int((i-burn_in)/thin),j] = proposal
            # otherwise make the next row of output for the parameter the same as the current value
            else:
                if iters % thin == 0 and i > burn_in:
                    mat[int((i-burn_in)/thin),j] = current[j].copy()
    # create a dataframe of the output
    seasonal_zonal_model_df = pd.DataFrame(data = mat)
    # caclulate the accepance rate by divinding acceptance count by the number of iterations
    seasonal_zonal_acceptance_rate = (acceptance_count/iters)*100
    # return the dataframe and acceptance rate
    return(seasonal_zonal_model_df, seasonal_zonal_acceptance_rate)

seasonal_zonal_model_df, seasonal_zonal_acceptance_rate = seasonal_zonal_model_met_hast(iters = 50000, inv = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.25, 0.05, 0.20, 0.20, 0.05, 0.30, 0.05, 0.25, 0.15, 0.25, 0.15, 0.25, 0.20, 0.20, 0.05, 0.18, 0.18, 0.18, 0.18, 0.15, 0.05, 0.17, 0.17, 0.17, 0.17, 0.20, 0.05]), data = data, burn_in = 500, thin = 5)
# check output dataframe
seasonal_zonal_model_df
# check acceptace rate is in acceptable region (approx 20-40)
seasonal_zonal_acceptance_rate
# check autocorrelation
plt.acorr(seasonal_zonal_model_df[1].to_numpy())
# check trace plots of the seperate parameters to see chain convergence
plt.plot(seasonal_zonal_model_df[1])
plt.ylabel('Phi 2')
plt.xlabel('Iterations')
plt.title('Traceplot of Phi 2')
plt.plot(seasonal_zonal_model_df[13])
plt.ylabel('Sig 1')
plt.xlabel('Iterations')
plt.title('Traceplot of Sig 1')
plt.plot(seasonal_zonal_model_df[63])
plt.ylabel('Tau')
plt.xlabel('Iterations')
plt.title('Traceplot of Tau')
# save output as a csv file
seasonal_zonal_model_df.to_csv('/Users/thomasoman/Desktop/Projects/Bayesian_models_outputs/seasonal_zonal_model_output.csv', index = False)

# Model 5 - CAR 1 model
def CAR1_model_logposterior(j, theta, data):
    # initalise paramters in theta
    phi = np.array(theta[0:12])
    sig = np.array(theta[12:63])
    tau = math.exp(theta[63])
    phi_vector = pd.Series(phi)[data['Month'] - 1].to_numpy()
    sig_vector = pd.Series(sig)[data['Zone'] - 1].to_numpy()
    # calculate log priors
    if j <= 11:
        logprior = norm.logpdf(phi[j], loc = 0, scale = 1000)
    elif j == 63:
        logprior = norm.logpdf(tau, loc = 0, scale = 1000)
    else:
        logprior = norm.logpdf(sig[j-12], loc = 0, scale = 1000)
    # calculate loglikelihood
    loglike = np.sum(norm.logpdf(np.log(data['Rate'].array + 0.001), loc = phi_vector+sig_vector, scale = 1/tau))
    # calculate log posterior
    logpost = loglike + logprior
    return(logpost)

# metropolis hasting alg for CAR1 model
def CAR1_model_met_hast(iters, inv, data, thin, burn_in):
    # initalise acceptance vector
    acceptance_count = np.zeros(64)
    # initalise output matrix iters rows and 63 columns
    mat = np.zeros((int((iters-burn_in)/thin), 64))
    # initalise start of markov chain
    start = np.zeros(64)
    # set first row of output to start of chain
    mat[0,] = start.copy()
    # keep track of current value of parameters
    current = mat[0,].copy()
    # loop over to get iters number of output rows
    for i in range(2,iters):
        # loop over the number of paramters
        for j in range(1,64):
            # generate random walk proposal value from a normal dist where the mean is the weighted sum of the parameters either side of the current parameter and std inv
            if j == 0:
                proposal = norm.rvs(loc = 0.5*current[11]+0.5*current[1], scale = inv[j], size = 1)[0]
            elif j == 11:
                proposal = norm.rvs(loc = 0.5*current[10]+0.5*current[0], scale = inv[j], size = 1)[0]
            elif j > 0 and j < 11:
                proposal = norm.rvs(loc = 0.5*current[j-1]+0.5*current[j+1], scale = inv[j], size = 1)[0]
            else:
                proposal = norm.rvs(loc = current[j], scale = inv[j], size = 1)[0]
            # calculate logposterior for the current value of paramters
            lpost = CAR1_model_logposterior(j, current, data)
            # set the proposal vector to the current set of paramters
            proposal_vec = current.copy()
            # swap the current parameter with the proposed value
            proposal_vec[j] = proposal
            # calulate the log of the acceptance probability which is the minimum value between 0 and the difference bewetween the log posteriors at the current and proposed values
            logaccprob = CAR1_model_logposterior(j, proposal_vec, data) - lpost
            # generate a random number between 0 and 1 with equal probability
            u = np.random.random()
            # if the log acceptance probability is greater than the log of the random number accept the proposal, set the next row of the output for the parameter to be the proposal value and increase the acceptace count
            if math.log(u) < logaccprob:
                current[j] = proposal
                acceptance_count[j] += 1
                if iters % thin == 0 and i > burn_in:
                    mat[int((i-burn_in)/thin),j] = proposal
            # otherwise make the next row of output for the parameter the same as the current value
            else:
                if iters % thin == 0 and i > burn_in:
                    mat[int((i-burn_in)/thin),j] = current[j].copy()
    # create a dataframe of the output
    CAR1_model_df = pd.DataFrame(data = mat)
    # caclulate the accepance rate by divinding acceptance count by the number of iterations
    CAR1_acceptance_rate = (acceptance_count/iters)*100
    # return the dataframe and acceptance rate
    return(CAR1_model_df, CAR1_acceptance_rate)

CAR1_model_df, CAR1_acceptance_rate = CAR1_model_met_hast(iters = 50000, inv = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.20, 0.20, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.20, 0.20, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.2, 0.2, 0.2, 0.2, 0.2, 0.20, 0.05]), data = data, burn_in = 500, thin = 5)
# check output dataframe
CAR1_model_df
# check acceptace rate is in acceptable region (approx 20-40)
CAR1_acceptance_rate
# check autocorrelation plots
plt.acorr(CAR1_model_df[1].to_numpy())
# check trace plots of the seperate parameters to see chain convergence
plt.plot(CAR1_model_df[1])
plt.ylabel('Phi 2')
plt.xlabel('Iterations')
plt.title('Traceplot of Phi 2')
plt.plot(CAR1_model_df[13])
plt.ylabel('Sig 1')
plt.xlabel('Iterations')
plt.title('Traceplot of Sig 1')
plt.plot(CAR1_model_df[63])
plt.ylabel('Tau')
plt.xlabel('Iterations')
plt.title('Traceplot of Tau')
# save output as a csv file
CAR1_model_df.to_csv('/Users/thomasoman/Desktop/Projects/Bayesian_models_outputs/CAR1_model_output.csv', index = False)

# Model 6 - CAR 2 model
def CAR2_model_logposterior(j, theta, data):
    # initalise paramters in theta
    phi = np.array(theta[0:12])
    sig = np.array(theta[12:63])
    tau = math.exp(theta[63])
    phi_vector = pd.Series(phi)[data['Month'] - 1].to_numpy()
    sig_vector = pd.Series(sig)[data['Zone'] - 1].to_numpy()
    # calculate log priors
    if j <= 11:
        logprior = norm.logpdf(phi[j], loc = 0, scale = 1000)
    elif j == 63:
        logprior = norm.logpdf(tau, loc = 0, scale = 1000)
    else:
        logprior = norm.logpdf(sig[j-12], loc = 0, scale = 1000)
    # calculate loglikelihood
    loglike = np.sum(norm.logpdf(np.log(data['Rate'].array + 0.001), loc = phi_vector+sig_vector, scale = 1/tau))
    # calculate log posterior
    logpost = loglike + logprior
    return(logpost)

# metropolis hasting alg for CAR2 model
def CAR2_model_met_hast(iters, inv, data, thin, burn_in):
    # initalise acceptance vector
    acceptance_count = np.zeros(64)
    # initalise output matrix iters rows and 63 columns
    mat = np.zeros((int((iters-burn_in)/thin), 64))
    # initalise start of markov chain
    start = np.zeros(64)
    # set first row of output to start of chain
    mat[0,] = start.copy()
    # keep track of current value of parameters
    current = mat[0,].copy()
    # loop over to get iters number of output rows
    for i in range(2,iters):
        # loop over the number of paramters
        for j in range(1,64):
            # generate random walk proposal value from a normal dist where the mean is the weighted sum of the parameters either side of the current parameter and std inv
            if j == 0:
                proposal = norm.rvs(loc = (1/3)*current[11]+(1/3)*current[1]+(1/6)*current[10]+(1/6)*current[2], scale = inv[j], size = 1)[0]
            elif j == 11:
                proposal = norm.rvs(loc = (1/3)*current[10]+(1/3)*current[0]+(1/6)*current[9]+(1/6)*current[1], scale = inv[j], size = 1)[0]
            elif j > 0 and j < 11:
                proposal = norm.rvs(loc = (1/3)*current[j-1]+(1/3)*current[j+1]+(1/6)*current[j-2]+(1/6)*current[j+2], scale = inv[j], size = 1)[0]
            else:
                proposal = norm.rvs(loc = current[j], scale = inv[j], size = 1)[0]
            # calculate logposterior for the current value of paramters
            lpost = CAR2_model_logposterior(j, current, data)
            # set the proposal vector to the current set of paramters
            proposal_vec = current.copy()
            # swap the current parameter with the proposed value
            proposal_vec[j] = proposal
            # calulate the log of the acceptance probability which is the minimum value between 0 and the difference bewetween the log posteriors at the current and proposed values
            logaccprob = CAR2_model_logposterior(j, proposal_vec, data) - lpost
            # generate a random number between 0 and 1 with equal probability
            u = np.random.random()
            # if the log acceptance probability is greater than the log of the random number accept the proposal, set the next row of the output for the parameter to be the proposal value and increase the acceptace count
            if math.log(u) < logaccprob:
                current[j] = proposal
                acceptance_count[j] += 1
                if iters % thin == 0 and i > burn_in:
                    mat[int((i-burn_in)/thin),j] = proposal
            # otherwise make the next row of output for the parameter the same as the current value
            else:
                if iters % thin == 0 and i > burn_in:
                    mat[int((i-burn_in)/thin),j] = current[j].copy()
    # create a dataframe of the output
    CAR2_model_df = pd.DataFrame(data = mat)
    # caclulate the accepance rate by divinding acceptance count by the number of iterations
    CAR2_acceptance_rate = (acceptance_count/iters)*100
    # return the dataframe and acceptance rate
    return(CAR2_model_df, CAR2_acceptance_rate)

CAR2_model_df, CAR2_acceptance_rate = CAR2_model_met_hast(iters = 50000, inv = np.array([0.1000, 0.00001, 0.1000, 0.0001, 0.1000, 0.1000, 0.1000, 0.1000, 0.0001, 0.00001, 0.1000, 0.1000, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.1500, 0.0500]), data = data, burn_in = 500, thin = 5)
# check output dataframe
CAR2_model_df
# check acceptace rate is in acceptable region (approx 20-40)
CAR2_acceptance_rate
# check autocorrelation plots
plt.acorr(CAR2_model_df[1].to_numpy())
# check trace plots of the seperate parameters to see chain convergence
plt.plot(CAR2_model_df[2])
plt.ylabel('Phi 3')
plt.xlabel('Iterations')
plt.title('Traceplot of Phi 3')
plt.plot(CAR2_model_df[13])
plt.ylabel('Sig 1')
plt.xlabel('Iterations')
plt.title('Traceplot of Sig 1')
plt.plot(CAR2_model_df[63])
plt.ylabel('Tau')
plt.xlabel('Iterations')
plt.title('Traceplot of Tau')
# save output as a csv file
CAR2_model_df.to_csv('/Users/thomasoman/Desktop/Projects/Bayesian_models_outputs/CAR2_model_output.csv', index = False)

# Model 7 - Hierarchical model
def Hier_model_logposterior(j, theta, mu_sig, tau_sig, tau_phi, data):
    # initalise paramters in theta
    phi = np.array(theta[0:12])
    sig = np.array(theta[12:63])
    tau = math.exp(theta[63])
    phi_vector = pd.Series(phi)[data['Month'] - 1].to_numpy()
    sig_vector = pd.Series(sig)[data['Zone'] - 1].to_numpy()
    # calculate log priors
    if j <= 11:
        logprior = norm.logpdf(phi[j], loc = 0, scale = math.sqrt(1/tau_phi))
    elif j == 63:
        logprior = norm.logpdf(tau, loc = 0, scale = 1000)
    else:
        logprior = norm.logpdf(sig[j-12], loc = mu_sig, scale = math.sqrt(1/tau_sig))
    # calculate loglikelihood
    loglike = np.sum(norm.logpdf(np.log(data['Rate'].array + 0.001), loc = phi_vector+sig_vector, scale = 1/tau))
    # calculate log posterior
    logpost = loglike + logprior
    return(logpost)

# metropolis hasting alg for Hier model
def Hier_model_met_hast(a, b, e, f, g, h, iters, inv, data, thin, burn_in):
    # initalise acceptance vector
    acceptance_count = np.zeros(64)
    # initalise output matrix iters rows and 63 columns
    mat = np.zeros((int((iters-burn_in)/thin), 64))
    # initalise hyper paramters matrix
    priormat = np.zeros((iters, 3))
    # initalise start of markov chain
    start = np.zeros(64)
    # set first row of output to start of chain
    mat[0,] = start.copy()
    # keep track of current value of parameters
    current = mat[0,].copy()
    # loop over to get iters number of output rows
    for i in range(2,iters):
        # find hyper paramters
        priormat[i,0] = norm.rvs(loc = (1/(51*priormat[i-1,1]+(1/b)))*(priormat[i-1,1]*np.sum(current[12:63])), scale = 1/(51*priormat[i-1,1]+(1/b)), size = 1)[0]
        priormat[i,1] = gamma.rvs(a = e+(51/2), scale = f+0.5*np.sum(np.square(current[12:63] - priormat[i,0])), size = 1)[0]
        priormat[i,2] = gamma.rvs(a = g+6, scale = h+5*np.sum(np.square(current[0:12])), size = 1)[0]
        # loop over the number of paramters
        for j in range(1,64):
            # generate random walk proposal value from a normal dist
            proposal = norm.rvs(loc = current[j], scale = inv[j], size = 1)[0]
            # calculate logposterior for the current value of paramters
            lpost = Hier_model_logposterior(j, theta = current, mu_sig = priormat[i,0], tau_sig = priormat[i,1], tau_phi = priormat[i,2], data = data)
            # set the proposal vector to the current set of paramters
            proposal_vec = current.copy()
            # swap the current parameter with the proposed value
            proposal_vec[j] = proposal
            # calulate the log of the acceptance probability which is the minimum value between 0 and the difference bewetween the log posteriors at the current and proposed values
            logaccprob = Hier_model_logposterior(j, theta = proposal_vec, mu_sig = priormat[i,0], tau_sig = priormat[i,1], tau_phi = priormat[i,2], data = data) - lpost
            # generate a random number between 0 and 1 with equal probability
            u = np.random.random()
            # if the log acceptance probability is greater than the log of the random number accept the proposal, set the next row of the output for the parameter to be the proposal value and increase the acceptace count
            if math.log(u) < logaccprob:
                current[j] = proposal
                acceptance_count[j] += 1
                if iters % thin == 0 and i > burn_in:
                    mat[int((i-burn_in)/thin),j] = proposal
            # otherwise make the next row of output for the parameter the same as the current value
            else:
                if iters % thin == 0 and i > burn_in:
                    mat[int((i-burn_in)/thin),j] = current[j].copy()
    # create a dataframe of the output
    Hier_model_df = pd.DataFrame(data = mat)
    # caclulate the accepance rate by divinding acceptance count by the number of iterations
    Hier_acceptance_rate = (acceptance_count/iters)*100
    # return the dataframe and acceptance rate
    return(Hier_model_df, Hier_acceptance_rate)

Hier_model_df, Hier_acceptance_rate = Hier_model_met_hast(a = 0, b = 1000, e = 0.001, f = 0.001, g = 0.001, h = 0.001, iters = 50000, inv = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.05], data = data, burn_in = 500, thin = 5)
# check output dataframe
Hier_model_df
# check acceptace rate is in acceptable region (approx 20-40)
Hier_acceptance_rate
# check trace plots of the seperate parameters to see chain convergence
plt.plot(Hier_model_df[1])
plt.ylabel('Phi 2')
plt.xlabel('Iterations')
plt.title('Traceplot of Phi 2')
plt.plot(Hier_model_df[12])
plt.ylabel('Sig 1')
plt.xlabel('Iterations')
plt.title('Traceplot of Sig 1')
plt.plot(Hier_model_df[63])
plt.ylabel('Tau')
plt.xlabel('Iterations')
plt.title('Traceplot of Tau')
# save output as a csv file
Hier_model_df.to_csv('/Users/thomasoman/Desktop/Projects/Bayesian_models_outputs/Hier_model_output.csv', index = False)


# Model 8 - KDE model

def KDE_model_logposterior(j, theta, data):
    # initalise paramters in theta
    phi = np.array(theta[0:12])
    sig = np.array(theta[12:63])
    tau = math.exp(theta[63])
    phi_vector = pd.Series(phi)[data['Month'] - 1].to_numpy()
    sig_vector = pd.Series(sig)[data['Zone'] - 1].to_numpy()
    # calculate log priors
    if j <= 11:
        logprior = norm.logpdf(phi[j], loc = 0, scale = 1000)
    elif j == 63:
        logprior = norm.logpdf(tau, loc = 0, scale = 1000)
    else:
        logprior = norm.logpdf(sig[j-12], loc = 0, scale = 1000)
    # calculate loglikelihood
    loglike = np.sum(norm.logpdf(np.log(data['Rate'].array + 0.001), loc = phi_vector+sig_vector, scale = 1/tau))
    # calculate log posterior
    logpost = loglike + logprior
    return(logpost)

# metropolis hasting alg for KDE model
def KDE_model_met_hast(iters, inv, data, thin, burn_in, b):
    # read co-ordinate file
    coord_df = pd.read_csv('/Users/thomasoman/Desktop/Projects/Datasets/florida_coord.txt', sep= " " , header = None)
    # turn into matric
    coord = coord_df.to_numpy()
    # initalise zero matrix
    distancemat = np.zeros((51,51))
    # calculate euclidean distance between each entry
    for i in range(51):
        for j in range(51):
            distancemat[i,j] = math.sqrt(np.square(coord[i,0] - coord[j,0])+np.square(coord[i,1] - coord[j,1]))
    # calculate weights
    weights = np.exp(-(np.square(distancemat))/b)
    # standardise the weights
    standardisedweight = np.zeros((51,51))
    for k in range(51):
        standardisedweight[k,] = weights[k,]/ np.sum(weights[k,])
    # initalise acceptance vector
    acceptance_count = np.zeros(64)
    # initalise output matrix iters rows and 63 columns
    mat = np.zeros((int((iters-burn_in)/thin), 64))
    # initalise start of markov chain
    start = np.zeros(64)
    # set first row of output to start of chain
    mat[0,] = start.copy()
    # keep track of current value of parameters
    current = mat[0,].copy()
    # loop over to get iters number of output rows
    for i in range(2,iters):
        # loop over the number of paramters
        for j in range(64):
            if j != 12:
            # generate random walk proposal value from a normal dist where the mean is the standardised weighted sum zones of the  either side of the current zone and std inv
                if j > 11 and j != 63:
                    proposal = norm.rvs(loc = np.sum(standardisedweight[j-12]*current[12:63]), scale = inv[j], size = 1)[0]
                else:
                    proposal = norm.rvs(loc = current[j], scale = inv[j], size = 1)[0]
            # calculate logposterior for the current value of paramters
                lpost = KDE_model_logposterior(j, current, data)
            # set the proposal vector to the current set of paramters
                proposal_vec = current.copy()
            # swap the current parameter with the proposed value
                proposal_vec[j] = proposal
            # calulate the log of the acceptance probability which is the minimum value between 0 and the difference bewetween the log posteriors at the current and proposed values
                logaccprob = KDE_model_logposterior(j, proposal_vec, data) - lpost
            # generate a random number between 0 and 1 with equal probability
                u = np.random.random()
            # if the log acceptance probability is greater than the log of the random number accept the proposal, set the next row of the output for the parameter to be the proposal value and increase the acceptace count
                if math.log(u) < logaccprob:
                    current[j] = proposal
                    acceptance_count[j] += 1
                    if iters % thin == 0 and i > burn_in:
                        mat[int((i-burn_in)/thin),j] = proposal
            # otherwise make the next row of output for the parameter the same as the current value
                else:
                    if iters % thin == 0 and i > burn_in:
                        mat[int((i-burn_in)/thin),j] = current[j].copy()
    # create a dataframe of the output
    KDE_model_df = pd.DataFrame(data = mat)
    # caclulate the accepance rate by divinding acceptance count by the number of iterations
    KDE_acceptance_rate = (acceptance_count/iters)*100
    # return the dataframe and acceptance rate
    return(KDE_model_df, KDE_acceptance_rate)

KDE_model_df, KDE_acceptance_rate = KDE_model_met_hast(iters = 50000, inv = np.array([0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.1, 0.1, 0.1, 0.01, 0.1, 0.001, 0.1, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001, 0.001, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.001, 0.1, 0.1, 0.001, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.1, 0.1, 0.001, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.1, 0.1, 0.1, 0.040]), data = data, burn_in = 500, thin = 5, b = 0.1)
# check output dataframe
KDE_model_df
# check acceptace rate is in acceptable region (approx 20-40)
KDE_acceptance_rate
# check trace plots of the seperate parameters to see chain convergence
plt.plot(KDE_model_df[0])
plt.ylabel('Phi 1')
plt.xlabel('Iterations')
plt.title('Traceplot of Phi 1')
plt.plot(KDE_model_df[13])
plt.ylabel('Sig 2')
plt.xlabel('Iterations')
plt.title('Traceplot of Sig 2')
plt.plot(KDE_model_df[63])
plt.ylabel('Tau')
plt.xlabel('Iterations')
plt.title('Traceplot of Tau')
# save output as a csv file
KDE_model_df.to_csv('/Users/thomasoman/Desktop/Projects/Bayesian_models_outputs/KDE_model_output.csv', index = False)
