# Statistics and Probability

## Terminology

**Population** - A collection or set of individuals or objects or events whose properties are to be analyzed.

**Sample** - A subset of population is called _Sample_. A well chosen sample will contains most of the information about a particular population parameter.

## Sampling Techniques

- Probability

  - Random
  - Systematic
  - Stratified

- Non-probability
  - Snowball
  - Quota
  - Judgement
  - Convenience

### Random

Each member of the population has equal chance of being selected in the sample.

### Systematic

In _Systematic_ sampling every nth record is chosen from the population to be a part of the sample.

### Stratified

- A stratum is a subset of the population that shares at least one common characteristic.
- Random sampling is used to select a sufficient of subjects from each stratum.

## Descriptive statistics

**Descriptive statistics** uses the data to provide descriptions of the population, either through numerical calculations or graphs or tables. (Maximum, Average, Minimum, etc.)

Descriptive statistics is mainly focused upon the main characteristics of data. It provides graphical summary of the data.

Descriptive statistics is a method used to describe and understand the features of a specific data set by giving short summaries about the sample and measures of the data.

**Categories**:

- Measures of Central tendancy

  - Mean
  - Median
  - Mode

- Measures of Variability (spread)
  - Range
  - Inter Quartile Range
  - Variance
  - Standard Deviation

## Inferential statistics

**Inferential statistics** makes inferences and predictions about a population based on a sample of data taken from the population in question. (Large, Medium, Small)

Inferential statistics, generalizes a large dataset and applies probability to draw a conclusion. It allows us to infer data parameters based on a statistical model using a sample data.

## Information Gain and Entropy

---

???

---

## Confuision Matrix

A _confusion matrix_ is a table that is often used to describe the perfomance of a classification model (or "classifier") on a set of test data for thich the true values are known.

```
Model Accuracy = True Positives + True Negatives / True Positives + True Negatives + False Positives + False Negatives
```

## Terminologies in Probability

**Random Experiment** An experiment or a process for which the outcome cannot be predicted with certainty.

**Sample Space** The entire possible set of outcomes of a random experiment is the sample space (S) of that experiment.

**Event** One or more outcomes of an experiment. It is a subset of sample space (S).

### Types of Events

- **Disjoint** - Mutually exclusive outcomes (dead and alive at same time)
- **Non-Disjoint** - Can have common outcomes

### Probability Distribution

- **Probability Density Function** - The equation describing a continuous probability distribution.
- **Normal Distribution** - Is a probability that associates the normal random variable X with a cumulative probability.
- **Central Limit Theorem** states that the sampling distribution of the mean of any independent, random variable well be normal or nearly normal, if sample size is large enough.

**(Note)** Normal Random variable is variable with mean at 0 and variance equal to 1.

### Types of Probability

- **Marginal Probability** - probability of occurrence of a single event.
- **Joint Probability** - measure of two events happening at the same time.
- **Conditional Probability** - probability of an event or outcome om the occurrence of a previous event or outcome.

## Bayes' Theorem

---

???

---

## Point Estimation

_Point estimation_ involves the use of sample data to calculate a single value _(known as a point estimate since it identifies a point in some parameter space)_ which is to serve as a "best guess" or "best estimate" of an unknown population parameter.

### Finding the Estimates

- Method of Moments
- Maximum of Likelihood
- Bayes' Estimators
- Best Unbiased Estimators

An Interval, or range of values, used to estimate a population parameter is called **Interval Estimate**.

### Margin of Error

Difference between the point estimate and the actual population parameter value is called the **Sampling Error**. 

Margin of Error for a given level of confidence is the greatest possible distance between the point estimate and the value of the parameter it is estimating.
