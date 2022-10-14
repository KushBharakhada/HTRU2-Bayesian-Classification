# HTRU2 Classification

## Data Set Information

HTRU2 is a data set which describes a sample of pulsar candidates collected during the High Time Resolution Universe Survey.
Candidates must be classified into pulsar and non-pulsar classes to aid discovery.

Pulsars are a rare type of Neutron star that produce radio emission detectable here on Earth. As pulsars rotate, their emission beam sweeps across the sky, and when this crosses our line of sight, produces a detectable pattern of broadband radio emission. As pulsars rotate rapidly, this pattern repeats periodically. Thus pulsar search involves looking for periodic radio signals with large radio telescopes.

Each pulsar produces a slightly different emission pattern, which varies slightly with each rotation. Thus a potential signal detection known as a 'candidate', is averaged over many rotations of the pulsar, as determined by the length of an observation. In the absence of additional info, each candidate could potentially describe a real pulsar. However in practice almost all detections are caused by radio frequency interference (RFI) and noise, making legitimate signals hard to find.

## Attribute Information

1. Mean of the integrated profile.
2. Standard deviation of the integrated profile.
3. Excess kurtosis of the integrated profile.
4. Skewness of the integrated profile.
5. Mean of the DM-SNR curve.
6. Standard deviation of the DM-SNR curve.
7. Excess kurtosis of the DM-SNR curve.
8. Skewness of the DM-SNR curve.
9. Class

## HTRU2 Summary

- 17,898 total examples.
- 1,639 positive examples.
- 16,259 negative examples.

## Code

Uses Bayesian Classification to classify Pulsae and Non-Pulsar.
There are two methods, they differ in the proportion of data used for Testing and Training.
The priors are calculated using the whole data set.

### Method 1

The data is split into positive (1) and negative (0) data. Half the positive data goes into the training data and the other half goes into the testing data. Half the negative data also goes into the training data, and the other half into the testing data. This creates the training and testing data sets. 

#### Training

The mean vector and covariance matrix is calculated for the positive and negative training data. A multivariate Gaussian distribution is created for the positive and negative training data using the mean and covariance.

#### Testing

The likelihoods are calculated for the testing data set. Each row is a feature vector of values (excluding the class column) which is a data point in the multivariate distribution. Therefore each row gives a single value back, p(x|w). The testing data set is calculated on each of the distributions, the trained positive and negative distributions and the priors are taken into account inorder to calculate p(w|x) (Bayes Rule). Using Bayes Decision Theorem, check which class, w, the feature vector x belongs to. Compare the resulting classifications with the actual classifications in the testing data.

#### Method 2

Idea is to use more data for training. Uses 'leave-one-out' testing. Uses just the first sample for testing and train using all the remaining N−1 samples. Then, repeat the this using the 2nd sample for testing and the other N−1 for training, and then again using the 3rd sample for testing and so on until all N samples have been tested. The number of correct classifications is calculated after.

## Results

### Method 1

- Actual Number of negatives in test: 8129
- Number of negatives the model got in test: 8088
- Actual Number of positives in test: 819
- Number of positives the model got in test: 860
- Percentage correct: 96.5690657130085

### Method 2
- Actual Number of negatives in data: 16259
- Number of negatives the model got in Leave-one-out: 16152
- Actual Number of positives in data: 1639
- Number of positives the model got in Leave-one-out: 1746
- Leave-one-out percentage correct: 96.73147837747234

## More Information

More information about the dataset is located in the information.txt and the citations below.

## Citation

Dataset: https://archive.ics.uci.edu/ml/datasets/HTRU2
R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach, Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656
R. J. Lyon, HTRU2, DOI: 10.6084/m9.figshare.3080389.v1.




