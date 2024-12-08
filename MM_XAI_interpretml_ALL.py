#!/usr/bin/env python
# coding: utf-8

# ### This version is the pipeline to do the following analysis:
# - Use the following methods to create classifier for LEMON data: scikit Logistic Regression, EBM, EDT (explainable decision tree), ELR (explainable Logistic Regression)
# - Find the F1 score, AUC and other metrics for the methods.
# - Use "intersecction" method to find common channels
# - ALL: Consider ALL common channels
# - PART 3: TOP 3 channels (PO3, PO8, P6)
# - PART 1: TOP 1 channels

# In[3]:


# import packages
import mne
import glob
import warnings
import time
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from interpret.glassbox import ExplainableBoostingClassifier
from interpret.glassbox import ClassificationTree



# In[ ]:



start = time.time()

# In[6]:


# Ignore specific warning
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# In[8]:


raw={}
fpath = "MENTAL-MATH"
for name in glob.glob(fpath+'/Subject*.edf'):
    #extract the unique code from imput file name (.set) in 2 parts
    fname = name.split("\\")[-1].replace(".edf", "").replace("Subject", "")
    #print(fname)
    exp = fname.split("_")[1] 
    sub = fname.split("_")[0]
    print(sub, exp)
    # #read the input file (.set) and store it in the "raw" dictionary whose key is a tuple (exp_type, subject)
    raw[(sub, exp)] = mne.io.read_raw_edf(name)


# In[10]:


eeg_dict = {}
ch_dict = {}
eeg_list = []

for key, val in raw.items():
    #Read the channel names from 1 of the .set (input) file
    chl_names = val.ch_names
    ch_dict[key] = chl_names

    # Read EEG
    eeg_df = val.to_data_frame().drop(columns='ECG ECG')   #convert raw data object to pandas dataframe  
    eeg_df['sub'] = key[0]
    eeg_df['exp'] = key[1]
    eeg_dict[key] = eeg_df
    eeg_list.append(eeg_df)
    # print(key,'\n', chl_names)
    # print('\n Number of channels are', len(chl_names))


# In[16]:


#eeg_list[0]


# #### Find the common channels in all eeg recordings

# In[12]:


# Find the intersection of columns across all DataFrames
common_columns = set(eeg_list[0].columns)
for df in eeg_list[1:]:
    common_columns &= set(df.columns)

# Output the result
print("Columns present in all DataFrames:", common_columns)
print("Number of columns present in all DataFrames:", len(common_columns))


# #### Select only those channels which are present in all the subjects

# In[15]:


# Filter each DataFrame to retain only common columns
eeg_common_chlist = [df[list(common_columns)] for df in eeg_list]
eeg_all_df = pd.concat(eeg_common_chlist, ignore_index=True)


# In[17]:


print(" The final dataframe with 20 channels, time, sub, and exp column:")
print(eeg_all_df.head())


# In[19]:Remove "EEG" from each channel name
eeg_all_df.columns = eeg_all_df.columns.str.replace("EEG", "")
eeg_all_df.head()


print(len(eeg_all_df))


# #### Another way to find common channels

# In[22]:


pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)


# In[24]:


eeg_all_df_float = eeg_all_df.drop(columns=['sub', 'exp']).apply(pd.to_numeric, downcast='float')


# In[26]:


print(eeg_all_df_float.dtypes)


# In[28]:


eeg_all_df_float['sub'] = eeg_all_df['sub']
eeg_all_df_float['exp'] = eeg_all_df['exp']


# In[30]:


print(eeg_all_df_float[eeg_all_df_float['exp'] == '1'].head())


# #### Number of instances of 'EO' ans 'EC'

# In[33]:


print('Number of samples for 1 = ', len(eeg_all_df_float[eeg_all_df_float['exp'] == '1']))
print('Number of samples for 2 = ', len(eeg_all_df_float[eeg_all_df_float['exp'] == '2']))


# ## PART 1: Consider ALL common channels

# In[36]:


# Split the dataset into features (X) and labels (y)
X = eeg_all_df_float.drop(columns=['sub', 'exp', 'time'])
y = eeg_all_df_float['exp']

# # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[38]:


print(X.head())
print("Columns in X are below:")
print(X.columns)


# In[40]:


print(y.head())


# **Scikit Logistic Regression**

# In[ ]:


# # Initialize and train a multi-class classifier (e.g., Logistic Regression)
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)


# In[ ]:


# # Make predictions on the test set
y_pred = model.predict(X_test)


# In[ ]:


# Evaluate the classifier
print("model performance for logistic regression \n")
print(classification_report(y_test, y_pred))
auc_lr_scikit = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print("AUC for logistic regression from scikit is: {:.3f}".format(auc_lr_scikit))


# ## Using interpretML 

# #### EXPLAINABLE Boosting Classifier (EBM)¶

# In[ ]:


ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)
# # Make predictions on the test set
y_pred_ebm = ebm.predict(X_test)


# In[ ]:


# # Evaluate the classifier
print("model performance for EBM is \n")
print(classification_report(y_test, y_pred_ebm))
auc_ebm = roc_auc_score(y_test, ebm.predict_proba(X_test)[:, 1])
print("AUC for EBM is: {:.4f}".format(auc_ebm))


# In[ ]:


ebm_global = ebm.explain_global()
ebm_local = ebm.explain_local(X_test[:5], y_test[:5])


# #### EXPLAINABLE Logistic Regression

# In[ ]:

from interpret.glassbox import LogisticRegression
lr = LogisticRegression(max_iter=3000, random_state=42)
lr.fit(X_train, y_train)

# # Make predictions on the test set
y_pred_lr = lr.predict(X_test)


# In[ ]:


# # Evaluate the classifier
print("model performance for Explainable LR is \n")
print(classification_report(y_test, y_pred_lr))

auc_lr = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
print("AUC for logistic regression explainable: {:.4f}".format(auc_lr))


# In[ ]:


## Explainable features
lr_global = lr.explain_global()
lr_local = lr.explain_local(X_test[:5], y_test[:5])


# #### EXPLAINABLE Decision trees

# In[ ]:


dt = ClassificationTree(random_state=42)
dt.fit(X_train, y_train)

# # Make predictions on the test set
y_pred_dt = dt.predict(X_test)


# In[ ]:


# # Evaluate the classifier
print("model performance for Explainable DT is \n")
print(classification_report(y_test, y_pred_dt))

auc_dt = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])
print("AUC for decsion trees explainable: {:.4f}".format(auc_dt))


# In[ ]:


# Explainable features
dt_global = dt.explain_global()
dt_local = dt.explain_local(X_test[:5], y_test[:5])
#print("Local for first 5 features in decision trees\n")


# #### Pickle the charts 

# In[ ]:

end = time.time()

# For EBM
#with open('MM_EBM_G1.pkl', 'wb') as f:
#	pickle.dump(ebm_global,f, protocol=4)
#with open('MM_EBM_L1.pkl', 'wb') as f1:  # open a text file
#    pickle.dump(ebm_local, f1, protocol=4) # serialize the list
##------------------------------
# FOR EXP-LOGISTIC REGRESSION
#with open('MM_ELR_G1.pkl', 'wb') as f:
#	pickle.dump(lr_global,f, protocol=4)
#with open('MM_ELR_L1.pkl', 'wb') as f1:  # open a text file
#    pickle.dump(lr_local, f1, protocol=4) # serialize the lis
#------------------------------
## FOR EXP-DECISION TREES
#with open('MM_EDT_G1.pkl', 'wb') as f:
#	pickle.dump(dt_global,f, protocol=4)
#with open('MM_EDT_L1.pkl', 'wb') as f1:  # open a text file
##    pickle.dump(dt_local, f1, protocol=4) # serialize the lis

#f.close()
#f1.close()

print("The time of execution of above program is :",(end-start))


# In[ ]:




