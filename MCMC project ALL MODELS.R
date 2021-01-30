##############Read and Clean Function for all models###################
read_and_clean = function(datalocation) # input is the location of the data stored on the computer i.e C:/...
{
  data = read.table(datalocation) #read in the data to a data frame called data
  colnames(data) <- c("Month",1:51)
  month = vector() #initialize empty vectors
  rate = vector()
  zone = vector()
  index = 1 # initalize index position of vectors
  column_count = 1 # initalize column count
  for (rows in 1:nrow(data)+1){ # loop over each row in data.
    for (value in data[rows,2:52]){ # loop over each value in row
      if (!is.na(value)){ # if the value is not NA 
        month[index] = data[rows, 1] # set month at position index equal to month of that row
        rate[index] = value # set rate at position index to be the value
        zone[index] = column_count # set zone at position index to be zone of that column
        cleaned_data = data.frame(month, rate, zone) # create data frame with 3 columns
        index = index + 1 # move index position for vectors along by 1
      }
      column_count = column_count + 1 # change column count every time a new value is used
      if(column_count == 52){ # if column count surpasses number of columns reset to 1
        column_count = 1
      }
    }
  }
  return(cleaned_data)
}
###################################
#########NULL MODEL################
###################################

#########LogPosterior for Null Model###################
Lposterior = function(j, theta, data)
{
  mu = theta[1] # labels of theta[1], theta[2]
  tau = exp(theta[2])
  if(j == 1){ #set counter j =1
    logprior = dnorm(mu,0,1000,log=T) #log of prior for mu using vague normal
  }else{ # if j!= 1
    logprior = dnorm(tau,0,1000,log=T) # log of prior for tau using vague normal
  }
  loglike = sum(dnorm(log(data$rate+0.001),mu,1/tau,log=T)) # log likelihood of normal using both mu and tau
  logpost = logprior+loglike # log of posterior
  return(logpost)
}
#############Null Model Function###############
data = read_and_clean("/Users/thomasoman/Documents/MmathStat/florida_acc.txt")
MetHast = function(iters,inv,data)
{
  mat= matrix(0,ncol = 2,nrow = iters) # matrix to store mu and tau
  theta = c(0,0)#initalise at prior mean
  mat[1,] = theta # set first row of matrix to inital theta
  acceptance_count = c(0,0) # # initalise acceptance count for both mu and tau
  current = mat[1,]
  for (i in 2:iters) { #loop over number of iterations
    for(j in 1 : 2){ # loop of counter for j
      proposal = rnorm(1,mat[i-1,j],inv[j]) # generate single proposal for mu using normal with mean at existing mu and variance inv
      Lpost = Lposterior(j,current, data) # calculate logposterior using prior for mu and existing mu and tau for likelihood
      proposal_vec = current
      proposal_vec[j] = proposal
      laprob = min(0, Lposterior(j,proposal_vec,data)-Lpost) # calculate logaccprob with logposterior using prior for mu but likelihood at proposal mu and existing tau
      u = runif(1) # random number between 0 and 1 with equal probability
      if (log(u) < laprob){ #if logof number between 1 and 0 is less than log of acceptance probability
        mat[i,j] = proposal # set ith row in matrix as proposal jth element of theta.
        acceptance_count[j] = acceptance_count[j]+1 # increase acceptance count of theta[j] by 1
        current[j] = proposal
      } else{
        mat[i,j] = mat[i-1,j] # otherwise set ith row of matrix to jth element of i-1th row
      }
    }
  }
  return(list(M = mat, AP =(acceptance_count/iters)*100)) # return list of matrix, acceptance percentage, rejection count.
}
x1 = MetHast(10000,c(0.025,0.025), data) # run trial function with the data.
plot(ts(x1$M))
x1$AP
summary(x1$M)
#######Null Model DIC Calculations####################
x1$M = x1$M[201:10000,]
posteriormean1 = c(rep(0,2))
for (i in 1:2){
  posteriormean1[i] = mean(x1$M[,i])
}
deviances1 = c(rep(0,9800))
for (i in 1:9800){
  deviances1[i] = -2*sum(dnorm(log(data$rate+0.001),x1$M[i,1],1/exp(x1$M[i,2]),log = T))
}
dbar1 = mean(deviances1)
dpbar1 =-2*sum(dnorm(log(data$rate+0.001), posteriormean1[1],1/exp(posteriormean1[2]),log = T))
DIC1 = 2*dbar1 - dpbar1
DIC1

#########################################
############SEASONAL MODEL###############
#########################################

#########LogPosterior for seasonal Model##############
Lposterior = function(j, theta, data)
{
  phi = theta[1:12] # labels of theta[1:12], theta[13]
  tau = exp(theta[13])
  if(j <= 12){ #set counter j <=12
    logprior = dnorm(phi,0,1000,log=T) #log of prior for mu using vague normal
  }else{ # if j!= 1
    logprior = dnorm(tau,0,1000,log=T) # log of prior for tau using vague normal
  }
  loglike = sum(dnorm(log(data$rate+0.001),phi[data$month],1/tau,log=T)) # log likelihood of normal using both mu and tau
  logpost = logprior+loglike # log of posterior
  return(logpost)
}
###########Seasonal Model function#################
MetHast = function(iters,inv,data)
{
  mat= matrix(0,ncol = 13,nrow = iters) # matrix to store mu and tau
  theta = c(rep(0,13))#initalise at prior mean
  mat[1,] = theta # set first row of matrix to inital theta
  acceptance_count = rep(0,13) # # initalise acceptance count for both phi and tau
  current = mat[1,]
  for (i in 2:iters) { #loop over number of iterations
    for(j in 1 : 13){ # loop of counter for j
      proposal = rnorm(1,mat[i-1,j],inv[j]) # generate single proposal for phi using normal with mean at existing phi and variance inv
      Lpost = Lposterior(j,current, data) # calculate logposterior using prior for phi and existing phi and tau for likelihood
      proposal_vec = current
      proposal_vec[j] = proposal
      laprob = min(0,Lposterior(j,proposal_vec,data)-Lpost) # calculate logaccprob with logposterior using prior for mu but likelihood at proposal mu and existing tau
      u = runif(1) # random number between 0 and 1 with equal probability
      if (log(u) < laprob){ #if logof number between 1 and 0 is less than log of acceptance probability
        mat[i,j] = proposal # set ith row in matrix as proposal jth element of theta.
        acceptance_count[j] = acceptance_count[j]+1 # increase acceptance count of theta[j] by 1
        current[j] = proposal
      } else{
        mat[i,j] = mat[i-1,j] # otherwise set ith row of matrix to jth element of i-1th row
      }
    }
  }
  return(list(M = mat, AP =(acceptance_count/iters)*100)) # return list of matrix, acceptance percentage, rejection count.
}
x2 = MetHast(10000,c(rep(0.1,12),0.05), data) # run trial function witht the data.
x2$AP
summary(x2$M)
########Seasonal Model DIC Calculations################
x2$M = x2$M[201:10000,]
posteriormean2 = c(rep(0,13))
for (i in 1:13){
  posteriormean2[i] = mean(x2$M[,i])
}
deviances2 = c(rep(0,9800))
for (i in 1:9800){
  deviances2[i] = -2*sum(dnorm(log(data$rate+0.001),x2$M[i,1:12][data$month],1/exp(x2$M[i,13]),log = T))
}
dbar2 = mean(deviances2)
dpbar2 =-2*sum(dnorm(log(data$rate+0.001), posteriormean2[1:12][data$month],1/exp(posteriormean2[13]),log = T))
DIC2 = 2*dbar2 - dpbar2
DIC2

#######################################
########ZONAL MODEL####################
#######################################

###########LogPosterior for Zonal Model####################
Lposterior = function(j, theta, data)
{
  sig = theta[1:51] # labels of theta[1:51], theta[13]
  tau = exp(theta[52])
  if(j <= 51){ #set counter j <=12
    logprior = dnorm(sig,0,1000,log=T) #log of prior for mu using vague normal
  }else{ # if j!= 1
    logprior = dnorm(tau,0,1000,log=T) # log of prior for tau using vague normal
  }
  loglike = sum(dnorm(log(data$rate+0.001),sig[data$zone],1/tau,log=T)) # log likelihood of normal using both mu and tau
  logpost = logprior+loglike # log of posterior
  return(logpost)
}
###########Zonal Model function#################
MetHast = function(iters,inv,data)
{
  mat= matrix(0,ncol = 52,nrow = iters) # matrix to store mu and tau
  theta = c(rep(0,52))#initalise at prior mean
  mat[1,] = theta # set first row of matrix to inital theta
  acceptance_count = rep(0,52) # # initalise acceptance count for both phi and tau
  current = mat[1,]
  for (i in 2:iters) { #loop over number of iterations
    for(j in 1 : 52){ # loop of counter for j
      proposal = rnorm(1,mat[i-1,j],inv[j]) # generate single proposal for phi using normal with mean at existing phi and variance inv
      Lpost = Lposterior(j,current, data) # calculate logposterior using prior for phi and existing phi and tau for likelihood
      proposal_vec = current
      proposal_vec[j] = proposal
      laprob = min(0,Lposterior(j,proposal_vec,data)-Lpost) # calculate logaccprob with logposterior using prior for mu but likelihood at proposal mu and existing tau
      u = runif(1) # random number between 0 and 1 with equal probability
      if (log(u) < laprob){ #if logof number between 1 and 0 is less than log of acceptance probability
        mat[i,j] = proposal # set ith row in matrix as proposal jth element of theta.
        acceptance_count[j] = acceptance_count[j]+1 # increase acceptance count of theta[j] by 1
        current[j] = proposal
      } else{
        mat[i,j] = mat[i-1,j] # otherwise set ith row of matrix to jth element of i-1th row
      }
    }
  }
  return(list(M = mat, AP =(acceptance_count/iters)*100)) # return list of matrix, acceptance percentage
}
x3 = MetHast(10000,c(rep(0.2,51),0.05), data) # run trial function witht the data.
x3$AP
summary(x3$M)
###################Zonal Model DIC Calculations####################
x3$M=x3$M[201:10000,]
posteriormean3 = c(rep(0,52))
for (i in 1:52){
  posteriormean3[i] = mean(x3$M[,i])
}
deviances3 = c(rep(0,9800))
for (i in 1:9800){
  deviances3[i] = -2*sum(dnorm(log(data$rate+0.001),x3$M[i,1:51][data$zone],1/exp(x3$M[i,52]),log = T))
}
dbar3 = mean(deviances3)
dpbar3 =-2*sum(dnorm(log(data$rate+0.001), posteriormean3[1:51][data$zone],1/exp(posteriormean3[52]),log = T))
DIC3 = 2*dbar3 - dpbar3
DIC3

############################################
############FULL MODEL######################
############################################

###########LogPosterior Function for S&S Model#########################
Lposterior = function(j, theta, data)
{
  phi = theta[1:12] # labels of theta
  sig = theta[13:63]
  tau = exp(theta[64])
  if(j <= 12){ #set counter j <=12
    logprior = dnorm(phi,0,1000,log=T) #log of prior for phi using vague normal
  }else if(j == 64){ # if j==64
    logprior = dnorm(tau, 0,1000,log=T) # log of prior for tau using vague normal
  } else{
    logprior = dnorm(sig,0,1000,log=T) # log of prior for sig using vague normal
  }
  loglike = sum(dnorm(log(data$rate+0.001),(phi[data$month]+sig[data$zone]),1/tau,log=T)) # log likelihood of normal using both phi+sigma and tau
  logpost = logprior+loglike # log of posterior
  return(logpost)
}
###############Seasonal and Site Model#########################
iters = 50000
inv = c(rep(0.05,12),rep(0.2,9),0.05,rep(0.1,8),0.05,rep(0.1,6),0.25,0.05,0.2,0.2,0.05,0.3,0.05,0.25,0.15,0.25,0.15,0.25,0.2,0.2,0.05,rep(0.18,4),0.15,0.05,rep(0.17,4),0.2,0.05)
mat= matrix(0,ncol = 64,nrow = iters) # matrix to store phi sigma and tau
current = c(rep(0,64))#initalise at prior mean
acceptance_count = rep(0,64) # # initalise acceptance count for all phi sig and tau
mat[1,] = current
for (i in 2:iters) { #loop over number of iterations
  for(j in 2 : 64){ # loop of counter for j
    proposal = rnorm(1,mat[i-1,j],inv[j]) # generate single proposal using normal with mean at existing phi or sigma and variance inv
    Lpost = Lposterior(j,current, data) # calculate logposterior using prior for phi or sigma and existing phi or sigma and tau for likelihood
    proposal_vec = current
    proposal_vec[j] = proposal
    laprob = min(0,Lposterior(j,c(proposal_vec),data)-Lpost) # calculate logaccprob with logposterior using prior for phi or sig but likelihood at proposal phi or sig and existing tau
    u = runif(1) # random number between 0 and 1 with equal probability
    if (log(u) < laprob){ #if logof number between 1 and 0 is less than log of acceptance probability
      mat[i,j] = proposal # set ith row in matrix as proposal jth element of theta.
      acceptance_count[j] = acceptance_count[j]+1 # increase acceptance count of theta[j] by 1
      current[j] = proposal
    }else{
      mat[i,j] = mat[i-1,j]#otherwise set ith row of matrix to jth element of i-1th row
    }
  }
}
x4 = list(M = mat, AP =(acceptance_count/iters)*100)
x4$AP
summary(x4$M)
x4$M = x4$M[seq(0,50000,length.out = 10000),]
plot(ts(x4$M[100:9900,13]), main = "Trace plot for the zonal effect at site 1", ylab = expression(sigma[1]))
plot(ts(x4$M[100:9900,10]), main = "Trace plot for the seasonal effect in October", ylab = expression(phi[10]))
hist(x4$M[100:9900,13], main = "Posterior distribution of the zonal effect at site 1", xlab = expression(sigma[1]))
hist(x4$M[100:9900,10], main = "Posterior distribution of the seasonal effect in October", xlab = expression(phi[10]))

x = seq(-2, 2, 0.01)
plot(x, dnorm(x, 0, 1000), type = "l", ylim = c(0,30), xlab = expression(phi[12]), ylab = "Density", main = "Prior and posterior distribution of seasonal effect in Decemeber")
lines(density(x4$M[100:9900,12]), col = "red")
plot(x, dnorm(x, 0, 1000), type = "l", ylim = c(0,30), xlab = expression(sigma[51]), ylab = "Density", main = "Prior and posterior distribution of zonal effect at site 51")
lines(density(x4$M[100:9900,63]), col = "red")


output = x4$M[100:9900,]
a = output[,2]+output[,13]
postmeana = mean(a)
uppercia= quantile(a,0.975)
lowercia= quantile(a, 0.025)
uppercia
lowercia


##########DIC Value Calculations#####################
posteriormean4 = c(rep(0,64))
for (i in 1:64){
  posteriormean4[i] = mean(x4$M[,i])
}
deviances4 = c(rep(0,9800))
for (i in 1:9800){
  deviances4[i] = -2*sum(dnorm(log(data$rate+0.001),x4$M[i,1:12][data$month]+x4$M[i,13:63][data$zone],1/exp((x4$M[i,64])),log = T))
}
dbar4 = mean(deviances4)
dpbar4 =-2*sum(dnorm(log(data$rate+0.001), posteriormean4[1:12][data$month]+posteriormean4[13:63][data$zone],1/exp(posteriormean4[64]),log = T))
DIC4 = 2*dbar4 - dpbar4
DIC4
#############Test/Train data split##########################
samp = sample(nrow(data),500) # sample 500 rows from data without replacement
train = data[-samp,] # set training set to all other rows of data
for(i in 1:51){ # check no zones have been lost.
  if( i %in% train$zone){
    print("yes")
  }else{
    print("no")

  }
}
test = data[samp,] #set test data
data = train# Run model with train data.
iters = 10000
inv = c(rep(0.05,12),rep(0.2,51),0.05)
mat= matrix(0,ncol = 64,nrow = iters) # matrix to store phi sigma and tau
theta = c(rep(0,64))#initalise at prior mean
mat[1,] = theta # set first row of matrix to inital theta
acceptance_count = rep(0,64) # # initalise acceptance count for all phi sig and tau
current = mat[1,]
for (i in 2:iters) { #loop over number of iterations
  for(j in 1 : 64){ # loop of counter for j
    proposal = rnorm(1,mat[i-1,j],inv[j]) # generate single proposal using normal with mean at existing phi or sigma and variance inv
    Lpost = Lposterior(j,current, data) # calculate logposterior using prior for phi or sigma and existing phi or sigma and tau for likelihood
    proposal_vec = current
    proposal_vec[j] = proposal
    laprob = min(0,Lposterior(j,c(proposal_vec),data)-Lpost) # calculate logaccprob with logposterior using prior for phi or sig but likelihood at proposal phi or sig and existing tau
    u = runif(1) # random number between 0 and 1 with equal probability
    if (log(u) < laprob){ #if logof number between 1 and 0 is less than log of acceptance probability
      mat[i,j] = proposal # set ith row in matrix as proposal jth element of theta.
      acceptance_count[j] = acceptance_count[j]+1 # increase acceptance count of theta[j] by 1
      current[j] = proposal
    } else{
      mat[i,j] = mat[i-1,j]#otherwise set ith row of matrix to jth element of i-1th row
    }
  }
}
x4 = list(M = mat, AP =(acceptance_count/iters)*100)
for(i in 1:64){
  plot(ts(x4$M[1:10000,i]))
}
x4$AP
summary(x4$M)
rates = vector() # initalise vec
vec_count = 1 #initalise count
for(month in 1:12){ # calculate vector of all possible rates for each month with each site
  for(site in 13:63){
    rates[vec_count] = mean(x4$M[,month]+x4$M[,site])
    vec_count = vec_count+1
  }
}
rates
test_index= vector()# initalise vec
test_index_count = 1 # initalise count
for(i in 1:500){ # calculate vector of corresponding indexes to test set
  test_index[test_index_count] = (test$month[i]-1)*51 + test$zone[i]
  test_index_count = test_index_count + 1
}
plot(test$rate,rates[test_index],xlim = c(0,20),ylim = c(0,20), ylab = "Predicted rate value", xlab = "Actual rate value") # create scatterplot of predicted value/actual value for all rates in test set.
lines(c(-20,20),c(-20,20)) # add y=x line
predictedrates = rates[test_index] # set predicted rates as the posterior rates at the corresponding indexes to the test set.
sqaures_vec = vector()# initalise vec
sqaures_count = 1 # initalise count
for(i in 1:500){ # calculate the square of the difference between the actual rates and the predicted rates
  sqaures_vec[sqaures_count] = (test$rate[i] - predictedrates[i])^2
  sqaures_count = sqaures_count +1
}
sumsqaures = sum(sqaures_vec) #sum the sqaured differences.
sumsqaures

##########################################################
###############SUMMER EFFECT MODEL########################
##########################################################

##############LogPosterior of Summer Effect Model###################
splitdatasummer = function(data)
{
  month = vector()
  rate = vector()
  zone = vector()
  index = 1
  count = 0
  for (value in data$month){
    count = count+1
    if (value <=10){
      if(value >=6){
        month[index] = data$month[count]
        rate[index] = data$rate[count]
        zone[index] = data$zone[count]
        cleaned_data = data.frame(month, rate, zone)
        index = index + 1
      }
    }
  }
  return(cleaned_data)
}
splitdatawinter = function(data)
{
  month = vector()
  rate = vector()
  zone = vector()
  index = 1
  count = 0
  for (value in data$month){
    count = count+1
    if (value >10 | value < 6){
      month[index] = data$month[count]
      rate[index] = data$rate[count]
      zone[index] = data$zone[count]
      cleaned_data = data.frame(month, rate, zone)
      index = index + 1
    }
  }
  return(cleaned_data)
}
summerdata = splitdata(data)
winterdata = splitdatawinter(data)
Lposterior = function(j, theta, summerdata,winterdata)
{
  summereffect= theta[1] # labels of theta
  sig = theta[2:52]
  tau = exp(theta[53])
  if(j == 1){ #set counter j <=12
    logprior = dnorm(summereffect,0,1000,log=T) #log of prior for phi using vague normal
  }else if(j == 53){ # if j==64
    logprior = dnorm(tau, 0,1000,log=T) # log of prior for tau using vague normal
  } else{
    logprior = dnorm(sig,0,1000,log=T) # log of prior for sig using vague normal
  }
  loglike1 = sum(dnorm(log(summerdata$rate+0.001),(summereffect+sig[summerdata$zone]),1/tau,log=T))
  loglike2 = sum(dnorm(log(winterdata$rate+0.001),(sig[winterdata$zone]),1/tau,log=T)) # log likelihood of normal using both phi+sigma and tau
  logpost = logprior+loglike1+loglike2 # log of posterior
  return(logpost)
}
#############Summer effect model##################
iters = 10000
inv = c(rep(0.05,1),rep(0.2,51),0.05)
mat= matrix(0,ncol = 53,nrow = iters) # matrix to store phi sigma and tau
theta = c(rep(0,53))#initalise at prior mean
mat[1,] = theta # set first row of matrix to inital theta
acceptance_count = rep(0,53) # # initalise acceptance count for all phi sig and tau
current = mat[1,]
for (i in 2:iters) { #loop over number of iterations
  for(j in 1 : 53){ # loop of counter for j
    proposal = rnorm(1,mat[i-1,j],inv[j]) # generate single proposal using normal with mean at existing phi or sigma and variance inv
    Lpost = Lposterior(j,current, summerdata, winterdata) # calculate logposterior using prior for phi or sigma and existing phi or sigma and tau for likelihood
    proposal_vec = current
    proposal_vec[j] = proposal
    laprob = min(0,Lposterior(j,proposal_vec,summerdata,winterdata)-Lpost) # calculate logaccprob with logposterior using prior for phi or sig but likelihood at proposal phi or sig and existing tau
    u = runif(1) # random number between 0 and 1 with equal probability
    if (log(u) < laprob){ #if logof number between 1 and 0 is less than log of acceptance probability
      mat[i,j] = proposal # set ith row in matrix as proposal jth element of theta.
      acceptance_count[j] = acceptance_count[j]+1 # increase acceptance count of theta[j] by 1
      current[j] = proposal
    } else{
      mat[i,j] = mat[i-1,j]#otherwise set ith row of matrix to jth element of i-1th row
    }
  }
}
x5 = list(M = mat, AP =(acceptance_count/iters)*100)
x5$AP
summary(x5$M)
###############Summer Effect Model DIC Calculations####################
x5$M = x5$M[201:10000,]
posteriormean5 = c(rep(0,53))
for (i in 1:53){
  posteriormean5[i] = mean(x5$M[,i])
}
deviances5 = c(rep(0,9800))
for (i in 1:9800){
  deviances5[i] = -2*sum(dnorm(log(data$rate+0.001),x5$M[i,1]+x5$M[i,2:52][data$zone],1/exp((x5$M[i,53])),log = T))
}
dbar5 = mean(deviances5)
dpbar5 =-2*sum(dnorm(log(data$rate+0.001), posteriormean5[1]+posteriormean5[2:52][data$zone],1/exp(posteriormean5[53]),log = T))
DIC5 = 2*dbar5 - dpbar5
DIC5

###############################################
##########CAR 1 MODEL##########################
###############################################

###############LogPosterior of CAR1 Model########################
Lposterior = function(j, theta, data)
{
  phi = theta[1:12] # labels of theta
  sig = theta[13:63]
  tau = exp(theta[64])
  if(j <= 12){ #set counter j <=12
    logprior = dnorm(phi,0,1000,log=T) #log of prior for phi using vague normal
  }else if(j == 64){ # if j==64
    logprior = dnorm(tau, 0,1000,log=T) # log of prior for tau using vague normal
  } else{
    logprior = dnorm(sig,0,1000,log=T) # log of prior for sig using vague normal
  }
  loglike = sum(dnorm(log(data$rate+0.001),(phi[data$month]+sig[data$zone]),1/tau,log=T)) # log likelihood of normal using both phi+sigma and tau
  logpost = logprior+loglike # log of posterior
  return(logpost)
}
###############CAR1 Model#####################
iters = 10000
inv = c(rep(0.05,12),rep(0.2,51),0.05)
mat= matrix(0,ncol = 64,nrow = iters) # matrix to store phi sigma and tau
theta = c(rep(0,64))#initalise at prior mean
mat[1,] = theta # set first row of matrix to inital theta
acceptance_count = rep(0,64) # # initalise acceptance count for all phi sig and tau
current = mat[1,]
for (i in 2:iters) { #loop over number of iterations
  for(j in 1 : 64){ # loop of counter for j
    if (j==1){
      proposal = rnorm(1,(0.5*mat[i-1,12]+0.5*mat[i-1,j+1]),inv[j])
    }else if (j==12){
      proposal = rnorm(1,(0.5*mat[i,1]+0.5*mat[i,j-1]),inv[j])
    }else if(j > 1 & j < 12){
      proposal = rnorm(1,(0.5*mat[i-1,j+1]+0.5*mat[i,j-1]),inv[j]) # generate single proposal using normal with mean at existing phi or sigma and variance inv
    }else{
      proposal = rnorm(1,mat[i-1,j],inv[j])
    }
    Lpost = Lposterior(j,current, data) # calculate logposterior using prior for phi or sigma and existing phi or sigma and tau for likelihood
    proposal_vec = vector(length = 64) # generate proposal vector length 64
    proposal_vec = current
    proposal_vec[j] = proposal
    laprob = min(0,Lposterior(j,c(proposal_vec),data)-Lpost) # calculate logaccprob with logposterior using prior for phi or sig but likelihood at proposal phi or sig and existing tau
    u = runif(1) # random number between 0 and 1 with equal probability
    if (log(u) < laprob){ #if logof number between 1 and 0 is less than log of acceptance probability
      mat[i,j] = proposal # set ith row in matrix as proposal jth element of theta.
      acceptance_count[j] = acceptance_count[j]+1 # increase acceptance count of theta[j] by 1
      current[j] = proposal
    } else{
      mat[i,j] = mat[i-1,j]#otherwise set ith row of matrix to jth element of i-1th row
    }
  }
}
x6 = list(M = mat, AP =(acceptance_count/iters)*100)
x6$AP
summary(x6$M)
################CAR1 Model DIC Calculations######################
x6$M = x6$M[201:10000,]
posteriormean6 = c(rep(0,64))
for (i in 1:64){
  posteriormean6[i] = mean(x6$M[,i])
}
deviances6 = c(rep(0,9800))
for (i in 1:9800){
  deviances6[i] = -2*sum(dnorm(log(data$rate+0.001),x6$M[i,1:12][data$month]+x6$M[i,13:63][data$zone],1/exp((x6$M[i,64])),log = T))
}
dbar6 = mean(deviances6)
dpbar6 =-2*sum(dnorm(log(data$rate+0.001), posteriormean6[1:12][data$month]+posteriormean6[13:63][data$zone],1/exp(posteriormean6[64]),log = T))
DIC6 = 2*dbar6 - dpbar6
DIC6

######################################################
##################CAR2 MODEL##########################
######################################################

##############LogPosterior for CAR2 Model###################
Lposterior = function(j, theta, data)
{
  phi = theta[1:12] # labels of theta
  sig = theta[13:63]
  tau = exp(theta[64])
  if(j <= 12){ #set counter j <=12
    logprior = dnorm(phi,0,1000,log=T) #log of prior for phi using vague normal
  }else if(j == 64){ # if j==64
    logprior = dnorm(tau, 0,1000,log=T) # log of prior for tau using vague normal
  } else{
    logprior = dnorm(sig,0,1000,log=T) # log of prior for sig using vague normal
  }
  loglike = sum(dnorm(log(data$rate+0.001),(phi[data$month]+sig[data$zone]),1/tau,log=T)) # log likelihood of normal using both phi+sigma and tau
  logpost = logprior+loglike # log of posterior
  return(logpost)
}
###############CAR2 Model#####################
iters = 10000
inv = c(0.1,0.1,0.1,0.0001,0.1,0.1,0.1,0.1,0.0001,0.1,0.1,0.1,rep(0.15,51),0.05)
data = read_and_clean("C:/Users/Thomas/florida_acc.txt")
mat= matrix(0,ncol = 64,nrow = iters) # matrix to store phi sigma and tau
theta = c(rep(0,64))#initalise at prior mean
mat[1,] = theta # set first row of matrix to inital theta
acceptance_count = rep(0,64) # # initalise acceptance count for all phi sig and tau
current = mat[1,]
for (i in 2:iters) { #loop over number of iterations
  for(j in 1 : 64){ # loop of counter for j
    if (j==1){
      proposal = rnorm(1,((1/3)*mat[i-1,12]+(1/3)*mat[i-1,j+1]+(1/6)*mat[i-1,11]+(1/6)*mat[i-1,j+2]),inv[j])
    }else if (j==2){
      proposal = rnorm(1,((1/3)*mat[i,1]+(1/3)*mat[i-1,j+1]+(1/6)*mat[i-1,12]+(1/6)*mat[i-1,j+2]),inv[j])
    }else if (j==12){
      proposal = rnorm(1,((1/3)*mat[i,1]+(1/3)*mat[i,j-1]+(1/6)*mat[i,j-2]+(1/6)*mat[i,2]),inv[j])
    }else if (j==11){
      proposal = rnorm(1,((1/3)*mat[i-1,12]+(1/3)*mat[i,j-1]+(1/6)*mat[i,1]+(1/6)*mat[i,j-2]),inv[j])
    }else if(j > 2 & j < 11){
      proposal = rnorm(1,((1/3)*mat[i-1,j+1]+(1/3)*mat[i,j-1]+(1/6)*mat[i-1,j+2]+(1/6)*mat[i,j-2]),inv[j]) # generate single proposal using normal with mean at existing phi or sigma and variance inv
    }else{
      proposal = rnorm(1,mat[i-1,j],inv[j])
    }
    Lpost = Lposterior(j,current, data) # calculate logposterior using prior for phi or sigma and existing phi or sigma and tau for likelihood
    proposal_vec = current
    proposal_vec[j] = proposal
    laprob = min(0,Lposterior(j,c(proposal_vec),data)-Lpost) # calculate logaccprob with logposterior using prior for phi or sig but likelihood at proposal phi or sig and existing tau
    u = runif(1) # random number between 0 and 1 with equal probability
    if (log(u) < laprob){ #if logof number between 1 and 0 is less than log of acceptance probability
      mat[i,j] = proposal # set ith row in matrix as proposal jth element of theta.
      acceptance_count[j] = acceptance_count[j]+1 # increase acceptance count of theta[j] by 1
      current[j] = proposal
    } else{
      mat[i,j] = mat[i-1,j]#otherwise set ith row of matrix to jth element of i-1th row
    }
  }
}
x7 = list(M = mat, AP =(acceptance_count/iters)*100)
x7$AP
summary(x7$M)
###############CAR2 Model DIC Calculations####################
x7$M = x7$M[201:10000,]
posteriormean7 = c(rep(0,64))
for (i in 1:64){
  posteriormean7[i] = mean(x7$M[,i])
}
deviances7 = c(rep(0,9800))
for (i in 1:9800){
  deviances7[i] = -2*sum(dnorm(log(data$rate+0.001),x7$M[i,1:12][data$month]+x7$M[i,13:63][data$zone],1/exp((x7$M[i,64])),log = T))
}
dbar7 = mean(deviances7)
dpbar7 =-2*sum(dnorm(log(data$rate+0.001), posteriormean7[1:12][data$month]+posteriormean7[13:63][data$zone],1/exp(posteriormean7[64]),log = T))
DIC7 = 2*dbar7 - dpbar7
DIC7

################################################
###############Heir Model#######################
################################################

###########LogPosterior Function for Heir Model#########################
Lposterior = function(j, theta, data,mu.sig,tau.sig,tau.phi)
{
  phi = theta[1:12] # labels of theta
  sig = theta[13:63]
  tau = exp(theta[64])
  if(j <= 12){ #set counter j <=12
    logprior = dnorm(phi,0,sqrt(1/tau.phi),log=T) #log of prior for phi using vague normal
  }else if(j == 64){ # if j==64
    logprior = dnorm(tau, 0,1000,log=T) # log of prior for tau using vague normal
  } else{
    logprior = dnorm(sig,mu.sig,sqrt(1/tau.sig),log=T) # log of prior for sig using vague normal
  }
  loglike = sum(dnorm(log(data$rate+0.001),(phi[data$month]+sig[data$zone]),1/tau,log=T)) # log likelihood of normal using both phi+sigma and tau
  logpost = logprior+loglike # log of posterior
  return(logpost)
}
###############Heir Model#########################
iters = 10000
inv = c(rep(0.05,12),rep(0.2,51),0.05)
a = 0
b = 1000
e = 0.001
f = 0.001
g = 0.001
h = 0.001
mat= matrix(0,ncol = 64,nrow = iters) # matrix to store phi sigma and tau
priormat = matrix(0,ncol = 3,nrow = iters)
priormat[1,] = c(a,e/f,g/h)
theta = c(rep(0,64))#initalise at prior mean
mat[1,] = theta # set first row of matrix to inital theta
acceptance_count = rep(0,64) # # initalise acceptance count for all phi sig and tau
current = mat[1,]
for (i in 2:iters) { #loop over number of iterations
  priormat[i,1] = rnorm(1,(1/(51*priormat[i-1,2]+(1/b)))*(priormat[i-1,2]*sum(mat[i-1,13:63])),1/(51*priormat[i-1,2]+(1/b)))
  priormat[i,2] = rgamma(1,e+(51/2),f+0.5*sum((mat[i-1,13:63] - priormat[i,1])^2))
  priormat[i,3] = rgamma(1,g+6,h+0.5*(sum((mat[i-1,1:12])^2)))
  for(j in 1 : 64){ # loop of counter for j
    proposal = rnorm(1,mat[i-1,j],inv[j]) # generate single proposal using normal with mean at existing phi or sigma and variance inv
    Lpost = Lposterior(j,current, data,priormat[i,1],priormat[i,2],priormat[i,3]) # calculate logposterior using prior for phi or sigma and existing phi or sigma and tau for likelihood
    proposal_vec = current
    proposal_vec[j] = proposal
    laprob = min(0,Lposterior(j,c(proposal_vec),data,priormat[i,1],priormat[i,2],priormat[i,3])-Lpost) # calculate logaccprob with logposterior using prior for phi or sig but likelihood at proposal phi or sig and existing tau
    u = runif(1) # random number between 0 and 1 with equal probability
    if (log(u) < laprob){ #if logof number between 1 and 0 is less than log of acceptance probability
      mat[i,j] = proposal # set ith row in matrix as proposal jth element of theta.
      acceptance_count[j] = acceptance_count[j]+1 # increase acceptance count of theta[j] by 1
      current[j] = proposal
    } else{
      mat[i,j] = mat[i-1,j]#otherwise set ith row of matrix to jth element of i-1th row
    }
  }
}
x8 = list(M = mat, AP =(acceptance_count/iters)*100)
x8$AP
summary(x8$M)
##########DIC Value Calculations#####################
x8$M = x8$M[201:10000,]
posteriormean8 = c(rep(0,64))
for (i in 1:64){
  posteriormean8[i] = mean(x8$M[,i])
}
deviances8 = c(rep(0,9800))
for (i in 1:9800){
  deviances8[i] = -2*sum(dnorm(log(data$rate+0.001),x8$M[i,1:12][data$month]+x8$M[i,13:63][data$zone],1/exp((x8$M[i,64])),log = T))
}
dbar8 = mean(deviances8)
dpbar8 =-2*sum(dnorm(log(data$rate+0.001), posteriormean8[1:12][data$month]+posteriormean8[13:63][data$zone],1/exp(posteriormean8[64]),log = T))
DIC8 = 2*dbar8 - dpbar8
DIC8

###############KDE Model#########################
coord = read.table("C:/Users/Thomas/MmathStat/florida_coord.txt")
distancemat = matrix(0,nrow = 51, ncol = 51)
for (i in 1:51){
  for (j in 1:51){
    distancemat[i,j] = sqrt((coord[i,1] - coord[j,1])^2+(coord[i,2]-coord[j,2])^2)
  }
}



Lposterior = function(j, theta, data)
{
  phi = theta[1:12] # labels of theta
  sig = theta[13:63]
  tau = exp(theta[64])
  if(j <= 12){ #set counter j <=12
    logprior = dnorm(phi,0,1000,log=T) #log of prior for phi using vague normal
  }else if(j == 64){ # if j==64
    logprior = dnorm(tau, 0,1000,log=T) # log of prior for tau using vague normal 
  }else{
    logprior = dnorm(sig,0,1000,log=T) # log of prior for sig using vague normal
  }
  loglike = sum(dnorm(log(data$rate+0.001),(phi[data$month]+sig[data$zone]),1/tau,log=T)) # log likelihood of normal using both phi+sigma and tau
  logpost = logprior+loglike # log of posterior 
  return(logpost)
}

b = 0.1
weights = exp(-(distancemat^2)/b)
standardisedweight = matrix(0,ncol = 51, nrow = 51)
for(k in 1:51){
  standardisedweight[k,] = weights[k,]/sum(weights[k,])
}
iters = 10000
inv = c(rep(0.05,12),rep(0.001,51),0.04)
mat= matrix(0,ncol = 64,nrow = iters) # matrix to store phi sigma and tau and b 
current = c(rep(0,12),rep(1,51),1)#initalise at prior mean 
mat[1,] = current # set first row of matrix to inital theta
acceptance_count = rep(0,64) # # initalise acceptance count for all phi sig and tau
loglike = vector()
for (i in 2:iters) { #loop over number of iterations
  for(j in 2 : 64){ # loop of counter for j
    if( j > 12 & j <= 63){
      proposal = rnorm(1,sum(standardisedweight[j-12,]*current[13:63]),inv[j])
      proposal_vec = current
      proposal_vec[j] = proposal
      laprob = min(0,Lposterior(j,c(proposal_vec),data)-Lposterior(j,current,data)) # calculate logaccprob with logposterior using prior for phi or sig but likelihood at proposal phi or sig and existing tau
      u = runif(1) # random number between 0 and 1 with equal probability 
      if (log(u) < laprob){ #if logof number between 1 and 0 is less than log of acceptance probability
        mat[i,j] = proposal # set ith row in matrix as proposal jth element of theta.
        acceptance_count[j] = acceptance_count[j]+1 # increase acceptance count of theta[j] by 1
        current[j] = proposal
      }else{
        mat[i,j] = mat[i-1,j]#otherwise set ith row of matrix to jth element of i-1th row 
      }
    }else{
      proposal = rnorm(1,current[j],inv[j]) # generate single proposal using normal with mean at existing phi or sigma and variance inv
      Lpost = Lposterior(j,current, data) # calculate logposterior using prior for phi or sigma and existing phi or sigma and tau for likelihood
      proposal_vec = current
      proposal_vec[j] = proposal
      laprob = min(0,Lposterior(j,proposal_vec,data)-Lpost) # calculate logaccprob with logposterior using prior for phi or sig but likelihood at proposal phi or sig and existing tau
      u = runif(1) # random number between 0 and 1 with equal probability 
      if (log(u) < laprob){ #if logof number between 1 and 0 is less than log of acceptance probability
        mat[i,j] = proposal # set ith row in matrix as proposal jth element of theta.
        acceptance_count[j] = acceptance_count[j]+1 # increase acceptance count of theta[j] by 1
        current[j] = proposal
      } else{
        mat[i,j] = mat[i-1,j]#otherwise set ith row of matrix to jth element of i-1th row 
      }
    }
  }
}
x9 = list(M = mat, AP =(acceptance_count/iters)*100)
x9$AP
summary(x9$M)
plot(ts(x9$M[,18]))
plot(ts(x9$M[100:9900,13]), main = "Trace plot for the zonal effect at site 1", ylab = expression(sigma[1]))
plot(ts(x9$M[100:9900,10]), main = "Trace plot for the seasonal effect in October", ylab = expression(phi[10]))
hist(x9$M[100:9900,13], main = "Posterior distribution of the zonal effect at site 1", xlab = expression(sigma[1]))
hist(x9$M[100:9900,10], main = "Posterior distribution of the seasonal effect in October", xlab = expression(phi[10]))

x = seq(-2, 2, 0.01)
plot(x, dnorm(x, 0, 1000), type = "l", ylim = c(0,30), xlab = expression(phi[12]), ylab = "Density", main = "Prior and posterior distribution of seasonal effect in Decemeber")
lines(density(x9$M[100:9900,12]), col = "red")
plot(x, dnorm(x, 0, 1000), type = "l", ylim = c(0,30), xlab = expression(sigma[51]), ylab = "Density", main = "Prior and posterior distribution of zonal effect at site 51")
lines(density(x9$M[100:9900,63]), col = "red")


output = x9$M[100:9900,]
a = output[,10]+output[,60] 
postmeana = mean(a)
uppercia= quantile(a,0.975)
lowercia= quantile(a, 0.025)
uppercia
lowercia


posteriormean9 = c(rep(0,64))
for (i in 1:64){
  posteriormean9[i] = mean(x9$M[,i])
}
deviances9 = c(rep(0,9800)) 
for (i in 1:9800){
  deviances9[i] = -2*sum(dnorm(log(data$rate+0.001),x9$M[i,1:12][data$month]+x9$M[i,13:63][data$zone],1/exp((x9$M[i,64])),log = T))
}
dbar9 = mean(deviances9)
dpbar9 =-2*sum(dnorm(log(data$rate+0.001), posteriormean9[1:12][data$month]+posteriormean9[13:63][data$zone],1/exp(posteriormean9[64]),log = T))
DIC9 = 2*dbar9 - dpbar9
DIC9
