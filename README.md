# Project 4: Finance Loan Approval Prediction

## Presented by Group 4
* Brent Beachtel
* Tico Brown
* Jim Haugen
* Celina K
* Justin
* Nataliia Shevchenko


### Purpose of the Analysis 
The purpose of this analysis is to develop a machine learning model capable of predicting loan approval of applicant(s) seeking a mortgage.  By leveraging neural networks, the goal is to create a binary classifier that can determine whether applicants will be successful in receiving a loan. This model aims to help loan providers to make data-driven decisions in selecting applicants, thereby providing one source for predicting the approval of mortgage applicants.



### Overview of the Analysis

#### Data Preprossing
The original data was provided in a .csv.  It was read into a Pandas DataFrame and found to comprise 614 rows of data representative of previous mortgage applicants comprised of the following features:
- **Loan_ID:** Unique identifier for each row. 
- **gender:** Gender of the applicant.
- **Dependents:** Number of dependents of the applicant.
- **Education:** College or non-college education of the applicant.
- **Self_Employed:** Self-employment of non-self-employment of the applicant.
- **ApplicantIncome:** Income of the applicant.
- **CoapplicantIncome:** Income of a co-applicant (if present).
- **LoanAmount:** Loan amount of the mortgage requested by the applicant (in thousands).
- **Loan_Amount_Term:** Number of months for the term of the mortgage (in months).
- **Credit_History:** Previous credit history of the applicant's repayment of debts.
- **Property_Area:** Area of the property to which mortgage will be applied, i.e., Urban/Rural/Semiurban.
- **Loan_Status:** Approval or non-approval of the mortgage for the applicant, i.e., Y-Yes, N-No.

After examination of the data, the following actions were performed:
- Loan_ID: was dropped.
- Loan_Status: Data type 'N' and 'Y' were transformed to '0' and '1', respectively.    
- The data was split into a train subset (80%) and a test subset (20%); the random state for reproducibility was set at 1.
- Categorical columns were defined: gender, Married, Dependents, Education, Self_Employed, and Property_Area.
- Numerical columns were defined: ApplicantIncome, CoapplicantIncome, LoanAmount, LoanAmountTerm, CreditHistory, and Loan_status.
- Missing values of the numerical columns of the train and test subsets were imputed with an imputer value defined by 5 neighbors.
- Categorical columns of the train and test subsets were converted into dummy/indicator variables.
- Loan_Status was considered "label" (y) because it is data which will indicate approval of the mortage applicant.
- All variables except Loan_Status were considered "features" (X) they are the data used to predict the approval of the mortage applicant.
- Test and train subsets were divided into the following:
  - First train and test subsets (i.e., y_train and y_test) were created comprising the "label" column.
  - Second train and test subsets (i.e., X_train and X_test) were created comprising the "features" column.  

#### Create a Logistic Regression Models
**First Logistic Regression Model:**
- A logistic regression model comprising reproducibility random state of 1 was created and fitted as a function of X_train and y_test data.
- Prediction data (i.e. y_pred) was created as a function of X_test data.
- A confusion matrix is a useful tool for evaluating the performance of a classification model. The matrix provides information about true positives (TPs), true negatives (TNs), false positives (FPs), and false negatives (FNs).  The following Confusion Matrix of this logistic regression model was generated as a function of the y_test and y_pred data:

![The optimized model](Screenshots/Picture3.png)

- A classification report is commonly used to evaluate the performance of a classification model in terms of precision, recall, F1-score, and support. The following Classfication Report of this logistic regression model was generated as a function of the y_test and y_pred data:

![image](https://github.com/NataliiaShevchenko620/Finance_loan_approval_prediction_data/tree/main/Screenshots/Picture4.png)

**Second Logistic Regression Model:**

- Polynomial features are employed so the model could be trained better on a relatively large set of features.  Here, polynomical features (i.e., X_train_poly, X_test_poly) were created and fitted as a function of X_train and X_test.  
- A second logistic regression model was created and fitted with X_train_poly and y_train data.
- Prediction data (i.e., y_pred) was re-created as a function of X_test_poly data.
- The following Classfication Report of the second logistic regression model was generated as a function of the y_test and y_pred data:

![image](https://github.com/NataliiaShevchenko620/Finance_loan_approval_prediction_data/tree/main/Screenshots/Picture5.png)

**Conclusion:** The first logistic regression model reveals a high precision for the positive class (0.78) and a high recall (0.98); the second model shows a decrease in precision (0.71) but an improvement in recall (0.51) for the negative class. The overall accuracy of the first model is 80%, while the second model's accuracy is 78%. Based on this analysis, the polynomial features and L1 regularization included in the second model did not significantly improve the model's performance and even reduced its precision in some cases. 

####  Create a Neural Network (NN) Models 
**First NN Model**
- A first NN model of a sequential NN model comprising two hidden layers and one output layer was created.
- The first hidden layer was added for receiving the input data of the features in the X_train data and comprised of 64 neurons and a ReLU activation function.
- A regularization technique for preventing overfitting of the model was added so that 50% of the neutrons were randomly dropped out (i.e., set to zero) to preventing co-adaptation of neurons and encourage the model to learn more robust features.
- The second hidden layer comprised of 10 neurons and a ReLU activation function was added. 
- The regularization technique was applied.
- The output layer comprised of 1 fully-connected neuron and a sigmoid activation function was added for squashing the output between 0 and 1. 
- The following presents the structure of the NN model:
    
[image](https://github.com/NataliiaShevchenko620/Finance_loan_approval_prediction_data/tree/main/Screenshots/Picture6.png)

**Second NN Model**
- A second NN model was complied.
- A loss function of cross-entropy was employed. In binary classification tasks, it is commonly employed to measure the difference between predicted and actual labels.
- An adaptive learning rate optimization algorithm was employed to determine how the model's weights are updated during training.
- The evaluation metric of accuracy was employed to measure how well the model predicted the correct class by representing the prorortion of correctly predicted samples.
- X_train and X_test were amended as a function of a scaling processing technique (i.e., StandardScaler()) so that the features were standardized. StandardScaler() contributes to the robustness, interpretability, and performance of machine learning models trained on diverse datasets.
- The second NN model was fitted as a function of X_train, y_train and 100 epochs, where the weights of the model are adjusted to minimize the loss function during each epoch.  
- The following presents the model's loss and accuracy as a function of X_test and y_test:  

![image](https://github.com/NataliiaShevchenko620/Finance_loan_approval_prediction_data/tree/main/Screenshots/Picture7.png)

**Third NN Model**
- A third NN model of a second sequential NN model comprising two hidden layers and one output layer was created.
- The first hidden layer was added for receiving the input data corresponding to the X_train features and comprised of 64 neurons and a ReLU activation function.
- The regularization technique discussed above was applied.
- The second hidden layer comprised of 32 neurons and a ReLU activation function was added. 
- The regularization technique was applied.
- The output hidden layer comprised of 1 fully-connected neuron and a sigmoid activation function was added. 
- The model was compiled using the Adam optimizer, binary_crossentropy loss function, and the evaluation metric of accuracy (discusssed above). The Adam optimizer is an adaptive learning rate optimization algorithm for adjusting the learning rate during training to improve convergence speed and performance. The binary_crossentropy loss function measures the difference between predicted probabilities and actual class labels.
- The third NN model was fitted as a function of X_train, y_train, 50 epochs, 32 samples used per iteration, and a 20% proportion of training data to use for validation during fitting.  
- The third NN model delivered an accuracy of 0.80.


**Fourth NN Model**
- A fourth NN model of a third sequential NN model comprising four hidden layers and one output layer was created.
- The first hidden layer was added for receiving the input data corresponding to the X_train features and comprised of 256 neurons, a ReLU activation function, and an L2 regularizer with a coefficient of 0.001 applied to the layer’s weights.
- A normalization technique was applied to improve convergence and generalization by ensuring that the mean activation is close to zero and the standard deviation is close to one.
- The regularization technique discussed above was applied.
- The second hidden layer comprised of 128 neurons, a ReLU activation function, and the L2 regularizer was added.
- The normalization and regularization techniques were applied.
- The third hidden layer comprised of 64 neurons, a ReLU activation function, and the L2 regularizer was added. 
- The normalization and regularization techniques were applied.
- The fourth hidden layer comprised of 32 neurons, a ReLU activation function, and the L2 regularizer was added. 
- The normalization and regularization techniques were applied.
- The output layer comprised of 1 fully-connected neuron and a sigmoid activation function was added. 
- The model was compiled using the Root Mean Square Propogation (i.e., RMSProp) optimizer, binary_crossentropy loss function, and the evaluation metric of accuracy (discusssed above). The RMSProp optimizer is an adaptive learning rate optimization algorithm for adjusting the learning rate during training based upon the gradient history corresponding to stochastic gradient descent in the training of deep enural networks.
- The fourth NN model delivered an accuracy of 0.8049.

**Summary:** 
- First NN Model
  - Accuracy: 0.80 The first neural network model, using two hidden layers with 15 and 10 neurons respectively demonstrated excellent performance, achieving an accuracy of 80%. This confirms that even a relatively simple neural network can effectively solve the binary classification task.

- Third NN Model
  - Accuracy: 0.80 The second neural network model with an increased number of neurons in the hidden layers (64 and 32 neurons) and the use of Dropout layers to prevent overfitting also achieved a high accuracy of 80%. This underscores that a more complex architecture can consistently maintain a high level of performance.

- Fourth NN Model
  - Accuracy: 0.80 The third neural network model, featuring a multi-layer architecture with a large number of neurons (256, 128, 64, 32), BatchNormalization, and Dropout in each layer, along with the RMSprop optimizer with a lower learning rate, also showed high accuracy at 80%. This demonstrates that a model with a more complex architecture and additional measures to prevent overfitting can maintain stable performance.

**Conclusion:** Neural networks have demonstrated their ability to solve the loan status classification task with high accuracy of 80%. Even with the use of different architectures and hyperparameters, the models consistently showed high performance. This confirms that neural networks are a powerful tool for data analysis and can be successfully used to solve binary classification problems. These results provide confidence that neural networks can effectively work with various types of data and can be adapted to solve more complex tasks in the future.


####  Create a K-Nearest Neighbors (KNN) Models
**First KNN Model**
- A first KNN model was created with the number of neighbors parameter set to 30.
- A cross-validated accuracy score for the model was calculated as a cross-validation function spitting the X_train and y_train data into 5 validation folds and evaluating the model's performance on each fold.
- An accuracy score list is appended with the mean score of the cross-validated accuracy score.
- The following presents an elbow plot of the model:
    
![image](https://github.com/NataliiaShevchenko620/Finance_loan_approval_prediction_data/tree/main/Screenshots/Picture10.png)

**Second KNN Model**
- A second KNN model was created with the number of neighbors parameter set to 8.
- The model was fitted as a function of X_train and y_train data.
- Prediction data (i.e. y_pred) was created as a function of X_test data.
- The model was evaluated using a confusion matrix generated as a function of y_test data, y_pred data, and the number of unique labels of the y_train data.
- The following presents the Confusion Matrix evaluation of the model:

![image](https://github.com/NataliiaShevchenko620/Finance_loan_approval_prediction_data/tree/main/Screenshots/Picture11.png)

**Third KNN Model**
- A third KNN model was created with grid search.
- A parameter grid for hyperparameter tuning of the model was defined with 30 neighbors (either uniform or distance-based weights), and distance metrics comprised of Euclidean distance, Manhattan distance (city block distance), and Minkowski distance for measuring similarity between data points.
- A grid search using cross-validation was set up to find the best hyperparameters of the model as a function of, in part, the parameter grid, 5-fold cross-validaation, and accuracy optimization.  
- The grid search was fitted with X_train and y_train data.  
- The best hyperparameters were determined.  
- A best model was trained with the best hyperparameters.
- Prediction data (i.e. y_pred) was created as a function of X_test data and the best model.
- The model was evaluated using a confusion matrix generated as a function of y_test data, y_pred data, and the number of unique labels of the y_train data.
- The following presents the Confusion Matrix evaluation of the model:
    
![image](https://github.com/NataliiaShevchenko620/Finance_loan_approval_prediction_data/tree/main/Screenshots/Picture12.png)

**Summary** 
- Second KNN Model
  - This model demonstrates its ability to correctly classify the majority of positive cases, showcasing its potential in this binary classification task.

- Third KNN Model
  - This model demonstrates a high precision for predicting positive loan statuses and an excellent recall, indicating that it effectively identifies approved loans.  
  
**Conclusion** 
-  Generally, KNN models are considered fairly simple models, so it is logical that the accuracy turned out to be lower than that of NN models discusssed above.


####  Create a Gradient-Boosting (GB) Models
**First GB Model**
- A Light Gradient-Boosting Machine (LightGBM) model was created and fitted as a function of X_train and y_train data.
- Prediction data (i.e. y_pred) was created as a function of X_test data.
- The model was evaluated using a confusion matrix generated as a function of y_test and y_pred data.
- The following presents the Confusion Matrix evaluation of the model:
  
![image](https://github.com/NataliiaShevchenko620/Finance_loan_approval_prediction_data/tree/main/Screenshots/Picture13.png)

**Second GB Model**
- A Histogram-Based Gradient Boosting Classifier was created and fitted as a function of X_train and y_train data.
- Prediction data (i.e. y_pred) was created as a function of X_test data.
- The model was evaluated using a confusion matrix generated as a function of y_test data and y_pred data.
- This model delivered an accuracy of 0.7642.
  
**Conclusion** 
-  The LightGBM model demonstrates a balanced performance with good precision and recall for both classes, making it a reliable choice for this classification problem.  When compared with the KNN models, all have shown strong performance in predicting loan statuses but still lower than NN models. The Third KNN Model particularly excels in identifying approved loans with high precision and recall, while the LightGBM model offers balanced and reliable predictions for both approved and rejected loans. These results highlight the effectiveness of both models in handling the binary classification task, providing a solid foundation for further improvements and applications in loan status prediction.


####  Create Random Forest Model
- A random forest model comprised of 500 estimators and a reproducibility random state of 78 was created and fitted as a function of X_train and y_train data.
- Prediction data (i.e. y_pred) was created as a function of X_test data.
- The model was evaluated using a confusion matrix and a classification report generated as a function of y_test and y_pred data.
- Feature importances were determined.
- The following presents the Confusion Matrix evaluation, Classfication Report evaluation, and Feature Importances of the model:   
  
![image](https://github.com/NataliiaShevchenko620/Finance_loan_approval_prediction_data/tree/main/Screenshots/Picture16.png)
    
![image](https://github.com/NataliiaShevchenko620/Finance_loan_approval_prediction_data/tree/main/Screenshots/Picture17.png)


**Summary** 
- Random Forest and Feature Evaluations
  - The Random Forest model demonstrated solid performance with an accuracy of 76%, effectively identifying the majority of approved loans while maintaining a balanced precision and recall for both classes. 
  - The Random Forest model provides insights into feature importance, highlighting which features contribute most to the prediction.
  
- Top Features
  - **Self_Employed:** Most significant feature with the highest relative importance indicating that whether an applicant is self-employed greatly influences loan approval. 
  - **Gender:** Second most important feature indicating that the applicant's gender plays a substantial role in the model's predictions. 
  - **Dependents:** The number of dependents an applicant is another critical factor in the model's predictions. 
  - **Married:** Marital status also significantly affects the prediction outcome. 
  - **Education:** The applicant's educational background is another important factor.
  
**Conclusion** 
-  The Random Forest model successfully demonstrates its capability in predicting loan statuses with a solid accuracy of 76%. It effectively identifies the majority of approved loans and provides valuable insights into the most influential features affecting loan approval. The feature importance analysis highlights the key factors such as self-employment status, gender, number of dependents, marital status, and education level, which can guide further decision-making and model improvements. These results showcase the robustness and interpretability of the Random Forest model, making it a reliable choice for loan status prediction tasks.


####  Create a Linear Regression Model
- A heat map of the original data was created to determine the correlations of the numerical columns.
- There are two positive correlations.
  - ApplicantIncome v. LoanAmount
  - CoapplicantIncome v. LoanAmount
- The following presents the heat map of the original data:   

![image](https://github.com/NataliiaShevchenko620/Finance_loan_approval_prediction_data/tree/main/Screenshots/Picture19.png)

- The following presents a histogram of ApplicantIncome v. Count and a boxplot of ApplicantIncome:     

![image](hhttps://github.com/NataliiaShevchenko620/Finance_loan_approval_prediction_data/tree/main/Screenshots/Picture20.png)

- The following presents a sactterplot of Applicant Income v. Loan Amount:     
   
![image](https://github.com/NataliiaShevchenko620/Finance_loan_approval_prediction_data/tree/main/Screenshots/Picture21.png)

- The following presents a scatterplot of Total Applicant Income (ApplicantIncome + CoapplicantIncome) v. Loan Amount:     

![image](https://github.com/NataliiaShevchenko620/Finance_loan_approval_prediction_data/tree/main/Screenshots/Picture22.png)

- The following presents a second scatterplot of Total Applicant Income (ApplicantIncome + CoapplicantIncome) v. Loan Amount:     
    
![image](hhttps://github.com/NataliiaShevchenko620/Finance_loan_approval_prediction_data/tree/main/Screenshots/Picture23.png)

- A linear regression model was created and fitted as a function of ApplicantIncome and LoanAmount.
- The model coeffient (i.e., slope) of the regression line is 0.007927.
- The model intercept (i.e., y-intercept) of the regression line is 103.57.   
- The Mean Absolute Percentage Error (MAPE) is 0.3387.
- The maximum loan amount for a mortgage = 103.57 + (0.007927 * Total Income).