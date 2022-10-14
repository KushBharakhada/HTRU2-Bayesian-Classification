'''

Classification.py

Main program where Bayes Classification will be used to classify the Pulsar's.

Author: Kush Bharakhada

Here the legitimate pulsar examples are a minority positive class, and spurious examples
the majority negative class.

The data set shared here contains 16,259 spurious examples caused by RFI/noise, and 1,639
real pulsar examples. These examples have all been checked by human annotators. Each
candidate is described by 8 continuous variables. The first four are simple statistics
obtained from the integrated pulse profile (folded profile). This is an array of continuous
variables that describe a longitude-resolved version of the signal that has been averaged
in both time and frequency (see [3] for more details). The remaining four variables are
similarly obtained from the DM-SNR curve (again see [3] for more details). These are
summarised below:

1. Mean of the integrated profile.
2. Standard deviation of the integrated profile.
3. Excess kurtosis of the integrated profile.
4. Skewness of the integrated profile.
5. Mean of the DM-SNR curve.
6. Standard deviation of the DM-SNR curve.
7. Excess kurtosis of the DM-SNR curve.
8. Skewness of the DM-SNR curve.

'''

import numpy as np
from scipy.stats import multivariate_normal

# Loading the data file
data = np.loadtxt(open("./data/HTRU_2.csv", "r"), delimiter=',')

CLASS_COLUMN_INDEX = 8

# Extracting the positive and negative data
positive_data = data[data[:, CLASS_COLUMN_INDEX] == 1, :]
negative_data = data[data[:, CLASS_COLUMN_INDEX] == 0, :]

# METHOD 1 (Splitting data into training and testing)

# Even rows represent training data
positive_data_train = positive_data[0::2, :]
negative_data_train = negative_data[0::2, :]

# Odd rows represent testing data
positive_data_test = positive_data[1::2, :]
negative_data_test = negative_data[1::2, :]

# Creating the new dataset for training and testing
pulsar_train = np.vstack((positive_data_train, negative_data_train))
pulsar_test = np.vstack((positive_data_test, negative_data_test))

# Calculating mean and covariance matrix
# Excluding last column which represents the class
positive_train_mean = np.mean(positive_data_train[:, 0:CLASS_COLUMN_INDEX], axis=0)
negative_train_mean = np.mean(negative_data_train[:, 0:CLASS_COLUMN_INDEX], axis=0)

positive_train_cov = np.cov(positive_data_train[:, 0:CLASS_COLUMN_INDEX], rowvar=0)
negative_train_cov = np.cov(negative_data_train[:, 0:CLASS_COLUMN_INDEX], rowvar=0)

# METHOD 2 (Leave-one-out testing)









