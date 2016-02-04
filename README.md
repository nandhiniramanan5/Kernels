# Kernels
linear, Polynomial,Gaussian, Sigmoidal Kernels

Xtrain is nxd dimension
I am selecting centers C randomly from Xtrain such that C is kxd dimensions. I observe that I get better results when k gets large as in some where close to n but not n. also The Linear kernel is the simplest kernel function. It is given by the inner product <x,y> plus an optional constant c ( which is optional).
 
C here is x-coordinate of the point that all the lines in the posterior go though. At this point, the function will have zero variance. If C=0 as in our case, the set of points x such that wTx = 0 are all points that are perpendicular to w and go through the origin. 
The output reported is when c=0 and Center has d= 150 and are selected arbitrarily. 

Solution 2.c
Implemented 1.linear 2. Polynomial 3.guassian 4.Sigmoid kernels on the data set blogData_train.csv.
 

My l2err_squared error rate is improved on applying kernel function to the linear regression algorithm.
Polynomial Kernel :  
Where Alpha is the slope which is adjustable, Constant c=1 and degree d=2, hence giving 2 degree kernel function. It performs better than the linear polynomial on the data set. 
Gaussian Kernel:  
Sigma here is the variance calculated over center matrix which is adjustable for better results. I noticed that on overestimating sigma, exponential performs like linear kernel. Neither the underestimate of sigma gives me a good result, hence chose to calculating c as the variance over Center matrix.

Sigmoid Kernel:  
Two adjustable parameters are alpha and intercept c here. I took c=0, as c here is x-coordinate of the point that all the lines in the posterior go though, assuming it to be origin. Alpha is tuned accordingly to avoid over fitting or underfitting.
My Gaussian and Sigmoid kernels tend to give me the most desirable results on the Dataset. I have reported one of the outputs for your reference.
