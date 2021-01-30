# Bayesian-modelling-of-road-collisions
This project was completed as part of my Masters program and involved creating, coding and evaluating different bayesian models of road collisions acorss 51 traffic zones in Florida. 

This was a 9 month project initally coded in R and later translated into python for learning purposes. In this project we aim to use Bayesian methods to model the rate of road collisions in the state of Florida using spatial and seasonal effects and then more advanced modeling techniques such as kernel density estimate and conditional auto regressive models to reduce the uncertainty in the posterior distributions for the effects in the models so that the models could be used more generally for road collision data world wide.

Florida_acc.txt is the orignal data I recieved from the department of transportation and it was accompanied by a file, Florida_coord.txt, containing the co-ordinates of the centriods of each of the 51 traffic zones. The file contains a lot of missing data but its assumed to be missing at random so it is able to be dropped without consequence to the models. 
