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

The actual data is split into positive and negative based on their class value (0 = negative, 1 = positive).

(Needs finishing)

## Results

(Needs finishing)

## More Information

More information about the dataset is located in the information.txt and the citations below.

## Citation

Dataset: https://archive.ics.uci.edu/ml/datasets/HTRU2
R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach, Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656
R. J. Lyon, HTRU2, DOI: 10.6084/m9.figshare.3080389.v1.




