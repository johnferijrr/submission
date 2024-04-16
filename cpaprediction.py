import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import math
from sklearn.metrics import mean_squared_error
from numpy import array
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
import csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_log_error
from numpy import array
from scipy.stats import kurtosis, skew
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[out_end_ix - 1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
def stats_features(input_data):
    inp = list()
    for i in range(len(input_data)):
        inp2=list()
        inp2=input_data[i]
        min=float(np.min(inp2))
        max=float(np.max(inp2))
        diff=(max-min)
        std=float(np.std(inp2))
        mean=float(np.mean(inp2))
        median=float(np.median(inp2))
        kurt=float(kurtosis(inp2))
        sk=float(skew(inp2))
        inp2=np.append(inp2,min)
        inp2=np.append(inp2,max)
        inp2=np.append(inp2,diff)
        inp2=np.append(inp2,std)
        inp2=np.append(inp2,mean)
        inp2=np.append(inp2,median)
        inp2=np.append(inp2,kurt)
        inp2=np.append(inp2,sk)
        #print(list(inp2))
        inp=np.append(inp,inp2)
    inp=inp.reshape(len(input_data),-1)
    #print(inp)
    return inp
import pandas as pd
zymuno_df = pd.read_csv('https://raw.githubusercontent.com/johnferijrr/submission/main/1%20year%20zymuno.csv', delimiter=',')
df_ori = zymuno_df
df_ori['Date'] = pd.to_datetime(df_ori['Date'])
df_X = df_ori[['Cost','CPC (Destination)','CPM','Impression','Clicks (Destination)','CTR (Destination)','Conversions','CPA','CPA']]
in_seq = df_X.astype(float).values
#out_seq = df_y.astype(float).values

#in_seq1 = in_seq.reshape(in_seq.shape[0], in_seq.shape[1])
#out_seq = out_seq.reshape((len(out_seq), 1))

#from numpy import hstack
#dataset = hstack((in_seq1, out_seq))


n_steps_in, n_steps_out = 4, 1
X, y = split_sequences(in_seq, n_steps_in, n_steps_out)

n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = False)
X_train = stats_features(X_train)
X_test = stats_features(X_test)


df_new=df_ori[['Date','CPA']]
df_new.set_index('Date')

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
X_train_no_nan = X_train[~np.isnan(X_train).any(axis=1)]
X_test_no_nan = X_test[~np.isnan(X_test).any(axis=1)]

y_train_no_nan = y_train[~np.isnan(y_train)]
y_test_no_nan = y_test[~np.isnan(y_test)]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_no_nan = X_train[~np.isnan(X_train).any(axis=1)]
X_test_no_nan = X_test[~np.isnan(X_test).any(axis=1)]

y_train_no_nan = y_train[~np.isnan(y_train)]
y_test_no_nan = y_test[~np.isnan(y_test)]
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
# Create the title and description
st.set_page_config(page_title="CPA Prediction App", page_icon="🔎")
st.title("CPA Prediction App 🔎")
st.write("""
This is a CPA Prediction App that uses machine learning algorithms to predict the Cost Per Acquisition (CPA) for a given set of input features (Cost, CPC (Destination), CPM, Impression, Clicks (Destination), CTR (Destination), Conversions, CPA) for the 4 days before tomorrow.
""")
st.write("""
Enter the Cost, CPC (Destination), CPM, Impression, Clicks (Destination), CTR (Destination), Conversions, and CPA at Day 1 until Day 4 (Don't forget to recheck again before click the button!):
""")
# Create the input widgets for the new name
new_name_inputs = []
with st.form("cpa_form"):
    for i in range(32):
        day = (i // 8) + 1
        metric = i % 8
        if metric == 0:
            metric = "Cost"
        elif metric == 1:
            metric = "CPC (Destination)"
        elif metric == 2:
            metric = "CPM"
        elif metric == 3:
            metric = "Impression"
        elif metric == 4:
            metric = "Clicks (Destination)"
        elif metric == 5:
            metric = "CTR (Destination)"
        elif metric == 6:
            metric = "Conversions"
        else:
            metric = "CPA"
        new_name_input = st.text_input(label=f'{metric} at Day {day}:', key=f'input_{i+32}')
        new_name_inputs.append(new_name_input)
    if st.form_submit_button("Predict The CPA!"):
        # Get the input values
        new_name = np.array([float(new_name_input) for new_name_input in new_name_inputs]).reshape(-1, X_test.shape[1])

        # Initialize the Random Forest Regressor model
        model = RandomForestRegressor(random_state=42)

        # Perform hyperparameter tuning using RandomizedSearchCV
        model.fit(X_train_no_nan, y_train_no_nan)

        # Make predictions on the test data
        y_pred = model.predict(new_name)
        y_pred = np.round(y_pred, 0)

        # Display the predictions in the sidebar
        st.sidebar.write("Tomorrow's CPA Prediction:")
        st.sidebar.write(y_pred)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.write("""
Please refresh the website if you want input new values
""")
# Create the description
st.write("""
Enter the CPA at day 1 until day 5 (Tomorrow's CPA Prediction) as CPA at Day 1 until CPA at Day 5 (Don't forget to recheck again before click the button!):
""")

# Create the input widgets for the new name
new_name_inputs_2 = []
with st.form("cpa_form_2"):
    for i in range(5):
        new_name_input = st.text_input(label=f'CPA at Day {i+1}:', key=f'input_{i+1}')
        new_name_inputs_2.append(new_name_input)
    if st.form_submit_button("Show Line Chart!"):
        # Get the input values
        new_name_2 = np.array([float(new_name_input) for new_name_input in new_name_inputs_2]).reshape(-1, 1)

        # Create the line chart
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 6), new_name_2, label='CPA Values')
        plt.xlabel('Day')
        plt.xticks(range(1, 6))
        plt.ylabel('CPA')
        plt.title('CPA Prediction Chart')
        plt.legend()
        plt.grid(False) # Set grid to False to remove the grid
        plt.scatter(range(1, 6), new_name_2, label='CPA Values')
        for i, txt in enumerate(new_name_2.flatten()):
            plt.annotate(txt, (i+1, txt))

        # Plot a red line from day 7 to 8
        plt.plot(range(4, 6), [new_name_2[3][0], new_name_2[4][0]], 'r-', label='Day 4 to 5')
        st.pyplot(plt)
	    
st.caption('Copyright (c) John Feri Jr. Ramadhan 2024')
