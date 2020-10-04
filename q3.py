import numpy as np
import pandas as pd
import csv
import random
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from pprint import pprint

class DecisionTree:

	def adjust_data(self, datafile):
		df = pd.read_csv(datafile)

		df = df.drop("Id", axis=1)
		df = df.drop("Alley", axis=1)
		df = df.drop("PoolQC", axis=1)
		df = df.drop("MiscFeature", axis=1)
		df = df.drop("Fence", axis=1)

		df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mode()[0])
		df['MasVnrType'] = df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
		df['MasVnrArea'] = df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
		df['BsmtQual'] = df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
		df['BsmtCond'] = df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
		df['BsmtExposure'] = df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
		df['BsmtFinType1'] = df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])
		df['BsmtFinType2'] = df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0]) 
		df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
		df['FireplaceQu'] = df['FireplaceQu'].fillna('NA')
		df['GarageType'] = df['GarageType'].fillna('NA')
		df['GarageYrBlt'] = df['GarageYrBlt'].fillna(2020.0)
		df['GarageFinish'] = df['GarageFinish'].fillna('NA')
		df['GarageQual'] = df['GarageQual'].fillna('NA')
		df['GarageCond'] = df['GarageCond'].fillna('NA')

		df = df.rename(columns={"SalesPrice" : "label"})

		return df

	def check_purity(self, data):
	    
	    label_column = data[:, -1]
	    unique_classes = np.unique(label_column)

	    if len(unique_classes) == 1:
	        return True
	    else:
	        return False

	def create_leaf(self, data, ml_task):
	    
	    label_column = data[:, -1]
	    if ml_task == "regression":
	        leaf = np.mean(label_column)
	        
	    # classfication    
	    else:
	        unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
	        index = counts_unique_classes.argmax()
	        leaf = unique_classes[index]
	    
	    return leaf
	
	def get_potential_splits(self, data):
	    
	    potential_splits = {}
	    _, n_columns = data.shape
	    for column_index in range(n_columns - 1):
	        values = data[:, column_index]
	#         print(values)
	        unique_values = np.unique(values)
	        
	        potential_splits[column_index] = unique_values
	    
	    return potential_splits

	def split_data(self, data, split_column, split_value):
	    
	    split_column_values = data[:, split_column]

	    type_of_feature = FEATURE_TYPES[split_column]
	    if type_of_feature == "continuous":
	        data_below = data[split_column_values <= split_value]
	        data_above = data[split_column_values >  split_value]
	    
	    # feature is categorical   
	    else:
	        data_below = data[split_column_values == split_value]
	        data_above = data[split_column_values != split_value]
	    
	    return data_below, data_above

	def calculate_mse(self, data):
	    actual_values = data[:, -1]
	    if len(actual_values) == 0:   # empty data
	        mse = 0
	        
	    else:
	        prediction = np.mean(actual_values)
	        mse = np.mean((actual_values - prediction) **2)
	    
	    return mse

	def calculate_entropy(self, data):
	    
	    label_column = data[:, -1]
	    _, counts = np.unique(label_column, return_counts=True)

	    probabilities = counts / counts.sum()
	    entropy = sum(probabilities * -np.log2(probabilities))
	     
	    return entropy

	def calculate_overall_metric(self, data_below, data_above, metric_function):
	    
	    n = len(data_below) + len(data_above)
	    p_data_below = len(data_below) / n
	    p_data_above = len(data_above) / n

	    overall_metric =  (p_data_below * metric_function(data_below) 
	                     + p_data_above * metric_function(data_above))
	    
	    return overall_metric

	def determine_best_split(self, data, potential_splits, ml_task):
	    
	    first_iteration = True
	    for column_index in potential_splits:
	        for value in potential_splits[column_index]:
	            data_below, data_above = self.split_data(data, split_column=column_index, split_value=value)
	            
	            if ml_task == "regression":
	                current_overall_metric = self.calculate_overall_metric(data_below, data_above, metric_function=self.calculate_mse)
	            
	            # classification
	            else:
	                current_overall_metric = self.calculate_overall_metric(data_below, data_above, metric_function=self.calculate_entropy)

	            if first_iteration or current_overall_metric <= best_overall_metric:
	                first_iteration = False
	                
	                best_overall_metric = current_overall_metric
	                best_split_column = column_index
	                best_split_value = value
	    
	    return best_split_column, best_split_value

	def determine_type_of_feature(self, df):
	    
	    feature_types = []
	    n_unique_values_treshold = 15
	    for feature in df.columns:
	        if feature != "label":
	            unique_values = df[feature].unique()
	            example_value = unique_values[0]

	            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
	                feature_types.append("categorical")
	            else:
	                feature_types.append("continuous")
	    
	    return feature_types

	def decision_tree_algorithm(self, df, ml_task, counter=0, min_samples=2, max_depth=5):
	    if counter == 0:
	        global COLUMN_HEADERS, FEATURE_TYPES
	        COLUMN_HEADERS = df.columns
	        FEATURE_TYPES = self.determine_type_of_feature(df)
	        data = df.values
	    else:
	        data = df           

	    if (self.check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
	        leaf = self.create_leaf(data, ml_task)
	        return leaf

	    else:    
	        counter += 1

	        potential_splits = self.get_potential_splits(data)
	        split_column, split_value = self.determine_best_split(data, potential_splits, ml_task)
	        data_below, data_above = self.split_data(data, split_column, split_value)

	        if len(data_below) == 0 or len(data_above) == 0:
	            leaf = self.create_leaf(data, ml_task)
	            return leaf

	        feature_name = COLUMN_HEADERS[split_column]
	        type_of_feature = FEATURE_TYPES[split_column]
	        if type_of_feature == "continuous":
	            question = "{} <= {}".format(feature_name, split_value)
	            
	        else:
	            question = "{} = {}".format(feature_name, split_value)
	        
	        sub_tree = {question: []}

	        yes_answer = self.decision_tree_algorithm(data_below, ml_task, counter, min_samples, max_depth)
	        no_answer = self.decision_tree_algorithm(data_above, ml_task, counter, min_samples, max_depth)
	        
	        if yes_answer == no_answer:
	            sub_tree = yes_answer
	        else:
	            sub_tree[question].append(yes_answer)
	            sub_tree[question].append(no_answer)
	        
	        return sub_tree

	def predict_example(self, example, tree):
	    question = list(tree.keys())[0]
	    feature_name, comparison_operator, value = question.split(" ")

	    # ask question
	    if comparison_operator == "<=":
	        if example[feature_name] <= float(value):
	            answer = tree[question][0]
	        else:
	            answer = tree[question][1]
	    
	    # feature is categorical
	    else:
	        if str(example[feature_name]) == value:
	            answer = tree[question][0]
	        else:
	            answer = tree[question][1]

	    # base case
	    if not isinstance(answer, dict):
	        return answer
	    
	    # recursive part
	    else:
	        residual_tree = answer
	        return self.predict_example(example, residual_tree)

	def train(self, datafile):
		df = self.adjust_data(datafile)

		train_df = df.iloc[:-122]
		val_df = df.iloc[-122:]

		tree = self.decision_tree_algorithm(train_df, ml_task="regression", max_depth=3)
		pprint(tree)


	def predict(self, datafile):
		df = self.adjust_data(datafile)
		l = df.shape
		n = l[0]

		train_df = self.adjust_data('./Datasets/q3/train.csv')
		tree = self.decision_tree_algorithm(train_df, ml_task="regression", max_depth=3)

		y_predicted = []

		for i in range(0, n):
			example = df.iloc[i]

			y_predicted.append(self.predict_example(example, tree))

		return y_predicted
