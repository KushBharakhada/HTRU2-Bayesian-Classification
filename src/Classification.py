'''

Classification.py

Main program where Bayes Classification will be used to classify the Pulsar's.
Candidates must be classified in to pulsar and non-pulsar classes.

Author: Kush Bharakhada

Citation:
R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection:
From simple filters to a new principled real-time classification approach, Monthly Notices of the Royal Astronomical
Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656

Dataset Information:
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

# Priors
POSITIVE_PRIOR = positive_data.shape[0]
NEGATIVE_PRIOR = negative_data.shape[0]

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

# Actual class values
real_class_values = pulsar_test[:, CLASS_COLUMN_INDEX]

# Calculating mean and covariance matrix
# Excluding last column which represents the class
positive_train_mean = np.mean(positive_data_train[:, 0:CLASS_COLUMN_INDEX], axis=0)
negative_train_mean = np.mean(negative_data_train[:, 0:CLASS_COLUMN_INDEX], axis=0)

positive_train_cov = np.cov(positive_data_train[:, 0:CLASS_COLUMN_INDEX], rowvar=0)
negative_train_cov = np.cov(negative_data_train[:, 0:CLASS_COLUMN_INDEX], rowvar=0)

# Creating the distribution from the trained data, using its mean and covariance matrix
pos_distribution = multivariate_normal(mean=positive_train_mean, cov=positive_train_cov)
# Each row from the test is a point in the multi-dimensional distribution, and each
# row will return a single value e.g. the height/likelihood
p1 = pos_distribution.pdf(pulsar_test[:, 0:CLASS_COLUMN_INDEX]) * POSITIVE_PRIOR

neg_distribution = multivariate_normal(mean=negative_train_mean, cov=negative_train_cov)
p2 = neg_distribution.pdf(pulsar_test[:, 0:CLASS_COLUMN_INDEX]) * NEGATIVE_PRIOR

# Stacking the scores from the distribution, top row is from the negative distribution
# and bottom row is from the positive distribution.
# Ordered this way so when the index of the higher value from top or bottom is retrieved,
# index 0 represents negative and 1 represents positive.
p = np.vstack((p2, p1))

# Gives the index's of which score is higher from the top and bottom row
index_higher_score = np.argmax(p, axis=0)

# Number of classes that were test which have been classified correctly
number_correct = np.count_nonzero(index_higher_score == real_class_values)
percentage_correct = (number_correct / pulsar_test.shape[0]) * 100

# Information
print("Actual Number of negatives in test:", np.count_nonzero(real_class_values == 0))
print("Number of negatives the model got in test:", np.count_nonzero(index_higher_score == 0))
print("Actual Number of positives in test:", np.count_nonzero(real_class_values == 1))
print("Number of positives the model got in test:", np.count_nonzero(index_higher_score == 1))
print("Percentage correct:", percentage_correct)

'''
Actual Number of negatives in test: 8129
Number of negatives the model got in test: 8088
Actual Number of positives in test: 819
Number of positives the model got in test: 860
Percentage correct: 96.5690657130085
'''

# METHOD 2 (Leave-one-out testing)

# Train a classifier using all the data in 'data' except the sample test_index. Then tests the
# sample test_index. Returns True if the sample is classified correctly otherwise returns False.
def train_and_test(test_index):
    # Splitting data: training and test value
    test_data = data[test_index, :]
    removed_index_sample = np.delete(data, test_index, axis=0)

    # Training

    # Splitting data into positive and negative
    positive_data_sample = removed_index_sample[removed_index_sample[:, CLASS_COLUMN_INDEX] == 1, :]
    negative_data_sample = removed_index_sample[removed_index_sample[:, CLASS_COLUMN_INDEX] == 0, :]

    # Covariances
    positive_sample_cov = np.cov(positive_data_sample[:, 0:CLASS_COLUMN_INDEX], rowvar=0)
    negative_sample_cov = np.cov(negative_data_sample[:, 0:CLASS_COLUMN_INDEX], rowvar=0)

    # Means
    positive_sample_mean = np.mean(positive_data_sample[:, 0:CLASS_COLUMN_INDEX], axis=0)
    negative_sample_mean = np.mean(negative_data_sample[:, 0:CLASS_COLUMN_INDEX], axis=0)

    # Testing

    pos_sample_distribution = multivariate_normal(mean=positive_sample_mean, cov=positive_sample_cov)
    np1 = pos_sample_distribution.pdf(test_data[:CLASS_COLUMN_INDEX]) * POSITIVE_PRIOR

    neg_sample_distribution = multivariate_normal(mean=negative_sample_mean, cov=negative_sample_cov)
    np2 = neg_sample_distribution.pdf(test_data[:CLASS_COLUMN_INDEX]) * NEGATIVE_PRIOR

    p_stack = np.vstack((np2, np1))

    # Checking which class gives the maximum
    # Using Bayes Decision Theory
    max_class = np.argmax(p_stack, axis=0)
    # Returns an array of a single boolean hence the index 0
    return (test_data[CLASS_COLUMN_INDEX] == max_class)[0]


# Classifying every sample using leave-one-out training
def classify():
    n_correct = 0
    n_total = data.shape[0]
    for index in range(n_total):
        n_correct = n_correct + train_and_test(index)

    percent_correct = n_correct * 100.0 / n_total
    return percent_correct


percent_correct_classifications = classify()
print("Leave-one-out percentage correct:", percent_correct_classifications)
# Leave-one-out percentage correct: 96.73147837747234

