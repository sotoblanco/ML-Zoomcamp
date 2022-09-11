# Week 1

## Supervised Machine Learning

We show examples to the algorithm, so it can learn from it and identify patters that can be extrapolated to unseen data. 

We have features store in a matrix X and the target vector denoted as y. 

g(X) = y

The function g is our model in which it takes as input our features and came up with a prediction that, if done appropriately, will get a close value to the target value "y". 

The type of output allows to classified supervised machine learning in two major categories:

1. Regression: Provides a numerical outcome (Housing price, car's price)
2. Classification
	2.2. Binary: Returns a category (Spam not spam)
	2.1. Multi-class: More than two classes
3. Ranking: Creates probability for multiple items


## CRISP-DM

![image](https://user-images.githubusercontent.com/46135649/188970370-d384b750-054a-40a3-bc99-34e62448df12.png)


Methodology on how Machine Learning Projects should be organize. 

### Business understanding

#### Identify the problem we want to solve

- Do we actually need Machine Learning?
- Analyze to what extent it's a problem
- Will Machine Learning help?
- If not: propose an alternative approach

#### Define the goal:
- Reduce the number of spam by 50% - KPI

### Data understanding

Analyze available data sources, decide if we need to get more data.

Analyzing the data is an essential step to understand the problem before even go to thinking on a solution. How the data is collected, the time it was collected and features around collection of data will help us to identify if the data is reliable or not. 

### Data preparation

Transform the data, so it can be put in a ML algorithm

- Clean the data
- Build the pipelines
- Convert into tabular form

### Data modeling

- Try different models
- Select the best one

Sometimes we may go back to data preparation

- Add features
- Fix data issues

### Evaluation

Measure how well the model solves the business problem

Is the model good enough?

Have we reached the goal?
Do our metrics improve?

Go back to data understanding for restropective:
Was the goal achievable?
Did we solve/measure the right thing?

After that. we may decide to:
- Go back and adjust the goal
- Roll the model to more users/all users
- Stop working on the project

### Evaluation + Deployment

- Online evaluation: evaluation of live users
- It means: deploy the model, evaluate it

- Roll the model to all users
- Proper monitoring
- Ensuring the quality and maintainability

### Iterate!

ML projects require many iterations!

Start simple
Learn from feedback
Improve


There are 6 steps to develop a machine learning project from end-to-end. Each of these steps should answer specific questions. 

1. What's the problem I am trying to solve? 

Usually this requires to understand the business and the metrics used to evaluate their performance. 

2. What data do I have?
 
This can be answered in many ways, format of the data, when has been collected, how it was collected? Is the data reliable?

3. Is the data useful for Machine Learning?

This question involves cleaning the data, do we have missing data? Is the data missing at random? Can we improve our features? What format of data my ML framework needs?

4. What is the best algorithm for my business model, data and the problem I am trying to answer?

This question and the selection of the algorithm needs to think about the ultimate customer, the data and the problem you're trying to answer. Some algorithms may have an easier interpretation than others, other works better on tabular data.

5. Was my model good enough to solve the business problem?

This question involves researching the potential impact of your model and how likely is to solve the business problem. Sometimes requires going back to previous step to have a better understanding on how good or bad is this model to your business requirements. 

6. Are we ready to test in production?

Now is the deployment phase, in which you test your model in real environments and get feedback from users. This phase is where MLOps is just starting!

7. Iterate
Start simple, learn from feedback, improve

Answer each of this question will improve your chances to make your model into production, in real situations your model won't make it if you are not able to communicate your findings. 


## Model Selection Process

Split the data into training and testing set. As a general guideline we can split the data into 80% for training the model and 20% to test the model performance. 

Let's say we follow this process and use 4 different models, and we compare them base on the accuracy

Logistic regression 66%
Decision Tree 60%
Random Forest 67%
Neural Network 80%

Multiple comparison problem

The accuracy of the models can be just by chance, to account for this problem we use a validation and testing set. 

Training data: 60%
Validation data: 20%
Testing data: 20%


We evaluate our results on the validation data, and then we compare to the testing data. 

Steps to develop a machine learning model

1. Split
2. Train
3. Validate
4. Select the best model
5. Test the model
6. Check

One approach that can be use is to after testing and checking your model performance is to use the validation data into the training data and run the model again, which is expected to give better results. 


## Linear algebra refresher

### Matrix-Vector multiplication


![matrix_vector](https://github.com/sotoblanco/ML-Zoomcamp/blob/main/gif/matrix_vector.gif)


### Code implementation Matrix-Vector

```python
def matrix_vector_multiplication(U, v):
    assert U.shape[1] == v.shape[0]
    
    num_rows = U.shape[0]
    
    result = np.zeros(num_rows)
    
    for i in range(num_rows):
        result[i] = vector_vector_multiplication(U[i], v)
    
    return result
```

### NumPy Matrix-Vector

```python
import numpy as np

U.dot(v)
```

### Matrix-Matrix Multiplication

![matrix_matrix](https://github.com/sotoblanco/ML-Zoomcamp/blob/main/gif/matrix_matrix.gif)

### Code implementation Matrix-Matrix multiplication

```python
def matrix_matrix_multiplication(U, V):
    assert U.shape[1] == V.shape[0]
    
    num_rows = U.shape[0]
    num_cols = V.shape[1]
    
    result = np.zeros((num_rows, num_cols))
    
    for i in range(num_cols):
        vi = V[:, i]
        Uvi = matrix_vector_multiplication(U, vi)
        result[:, i] = Uvi
    
    return result
```


### Matrix-Matrix Identity

Why matrix identity returns the same matrix

![matrix_matrix_identity](https://github.com/sotoblanco/ML-Zoomcamp/blob/main/gif/matrix_matrix_indentity.gif)

### Code implementation Matrix-Matrix Identity

```python
I = eye(3)
V = np.array([
    [1, 1, 2],
    [0, 0.5, 1], 
    [0, 2, 1],
])
inv = np.linalg.inv(V)

```



