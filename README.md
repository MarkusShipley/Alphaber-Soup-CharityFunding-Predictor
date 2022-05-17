## Deep Learning Homework: Charity Funding Predictor
Note:  Observations also captured at the end of the each Jupyter Notebook as well.
## Details and Observations
The assignment consists of three Jupyter Notebooks: 1) AlphabetSourCharityModel.ipynb  2) AlphabetSourCharity-OptimizeModel.ipynb 3) AlphabetSourCharity-OptimizeModel2.ipynb

1 and 2 met the requirements of the homework.  However, I wanted to "play around" with some of the settings of the Optimization Model and analyze the resuls.

## AlphabetSourCharityModel.ipynb CONCLUSION: 
CONCLUSION: Accuracy is 72.43% and loss was 56.31%  Accuracy could or may have gone up with more epochs and time to learn.  Howevver, with this model we are/were able to predict approximately 72% of the campaigns or organizations that would be successful.  In step 3 (next part of the homework assignment), we are going to attempt to optimize the model.

## AlphabetSourCharity-OptimizeModel.ipynb CONCLUSION
CONCLUSION: Accuracy for the opti,ized model is 79.00% and loss was 46.16%  With this model we are/were able to predict approximately 79% of the campaigns or organizations that would be successful. We achieved a target predictive accuracy higher than 75% by dropping the "STATUS" and "SPECIAL CONSIDERATIONS" Columns.  However, we retained the "NAME" column in case name sake or reputation were part of the success of the campaign.  Several changes were made to the model: 1) Changed the layer1Nodes to 100  2) Added a third set of nodes with a value of 30.  Thus, we had three hidden nodes with node values for 100, 30 and 10, respectively; 3) The activation for the second hidden layer was change from 'relu" to 'sigmoid'.

## AlphabetSourCharity-OptimizeModel2.ipynb CONCLUSION
CONCLUSION: Accuracy for the second/"fun" optimized model is 79.43 and loss was 46.27%  With this model we are/were able to predict approximately 79.4% of the campaigns or organizations that would be successful. As written earlier, the other/first optimization met the homework requirements of generating a model wtih accuracy greater than 75%.  This model is just for fun and to "learn"/"see" the impact of making changed to the just some of the model settings.  

Changes were made to the model as follows: 1) layer1Nodes = 200,  2) layer2Nodes = 75,  3) layer3Nodes = 50  4) epochs = 300.
I do find it interesting that the substantial increase in nodes and increasing the epochs from 100 to 300 yielded a .04% less  accurate model.  I wish I had more time to "play" with this homework.  I would determine other columns to drop or add back, add additional hidden layers and used differing activations.  I wish this was a year long boot camp so we had the time to start getting into the math, etc. behind all of this. 

I write this with every homework, but this one was my favorite so far.  Only one more homework and the final project.  I wish I could have had Dr Arrington for the full term.  I have learned a great deal from him and will always be grateful for that which he has taught..

## Homework Requirements
From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

* **EIN** and **NAME**—Identification columns
* **APPLICATION_TYPE**—Alphabet Soup application type
* **AFFILIATION**—Affiliated sector of industry
* **CLASSIFICATION**—Government organization classification
* **USE_CASE**—Use case for funding
* **ORGANIZATION**—Organization type
* **STATUS**—Active status
* **INCOME_AMT**—Income classification
* **SPECIAL_CONSIDERATIONS**—Special consideration for application
* **ASK_AMT**—Funding amount requested
* **IS_SUCCESSFUL**—Was the money used effectively

## Instructions

### Step 1: Preprocess the data

Using your knowledge of Pandas and the Scikit-Learn’s `StandardScaler()`, you’ll need to preprocess the dataset in order to compile, train, and evaluate the neural network model later in Step 2

Using the information we have provided in the starter code, follow the instructions to complete the preprocessing steps.

1. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
  * What variable(s) are considered the target(s) for your model?
  * What variable(s) are considered the feature(s) for your model?
2. Drop the `EIN` and `NAME` columns.
3. Determine the number of unique values for each column.
4. For those columns that have more than 10 unique values, determine the number of data points for each unique value.
6. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.
7. Use `pd.get_dummies()` to encode categorical variables

### Step 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1. Continue using the jupter notebook where you’ve already performed the preprocessing steps from Step 1.
2. Create a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.
3. Create the first hidden layer and choose an appropriate activation function.
4. If necessary, add a second hidden layer with an appropriate activation function.
5. Create an output layer with an appropriate activation function.
6. Check the structure of the model.
7. Compile and train the model.
8. Create a callback that saves the model's weights every 5 epochs.
9. Evaluate the model using the test data to determine the loss and accuracy.
10. Save and export your results to an HDF5 file, and name it `AlphabetSoupCharity.h5`.

### Step 3: Optimize the Model

Using your knowledge of TensorFlow, optimize your model in order to achieve a target predictive accuracy higher than 75%. If you can't achieve an accuracy higher than 75%, you'll need to make at least three attempts to do so.

Optimize your model in order to achieve a target predictive accuracy higher than 75% by using any or all of the following:

* Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model, such as:
  * Dropping more or fewer columns.
  * Creating more bins for rare occurrences in columns.
  * Increasing or decreasing the number of values for each bin.
* Adding more neurons to a hidden layer.
* Adding more hidden layers.
* Using different activation functions for the hidden layers.
* Adding or reducing the number of epochs to the training regimen.

**NOTE**: You will not lose points if your model does not achieve target performance, as long as you make three attempts at optimizing the model in your jupyter notebook.

1. Create a new Jupyter Notebook file and name it `AlphabetSoupCharity_Optimzation.ipynb`.
2. Import your dependencies, and read in the `charity_data.csv` to a Pandas DataFrame.
3. Preprocess the dataset like you did in Step 1, taking into account any modifications to optimize the model.
4. Design a neural network model, taking into account any modifications that will optimize the model to achieve higher than 75% accuracy.
5. Save and export your results to an HDF5 file, and name it `AlphabetSoupCharity_Optimization.h5`.

### Step 4: Write a Report on the Neural Network Model

For this part of the Challenge, you’ll write a report on the performance of the deep learning model you created for AlphabetSoup.

The report should contain the following:

1. **Overview** of the analysis: Explain the purpose of this analysis.

2. **Results**: Using bulleted lists and images to support your answers, address the following questions.

  * Data Preprocessing
    * What variable(s) are considered the target(s) for your model?
    * What variable(s) are considered to be the features for your model?
    * What variable(s) are neither targets nor features, and should be removed from the input data?
  * Compiling, Training, and Evaluating the Model
    * How many neurons, layers, and activation functions did you select for your neural network model, and why?
    * Were you able to achieve the target model performance?
    * What steps did you take to try and increase model performance?

3. **Summary**: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.

- - -

## Rubric

[Unit 21 - Deep Learning Homework Rubric - Charity Funding Predictor](https://docs.google.com/document/d/1SLOROX0lqZwa1ms-iRbHMQr1QSsMT2k0boO9YpFBnHA/edit?usp=sharing)

___
© 2021  Trilogy Education Services, a 2U, Inc. brand. All Rights Reserved.	
