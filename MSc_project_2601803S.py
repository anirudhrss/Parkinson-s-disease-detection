#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import wave
from IPython.display import Audio
#
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import os
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
#
import random
import itertools
import librosa as librosa
import IPython.display as ipd
get_ipython().run_line_magic('matplotlib', 'inline')
from logging import error
from pydub import AudioSegment
from pydub.silence import split_on_silence
import glob
import os.path
#
import opensmile
#
from sklearn import svm
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[6]:


#Check if the audio files are loaded correcctly
Audio(r'D:\Projet data set\26-29_09_2017_KCL\SpontaneousDialogue\PD\ID04_pd_2_0_1.wav')


# In[8]:


#Creating spectograms of the dataset
for filee in path:
    sampleR, samples2 = wavfile.read(os.path.join(r"C:\Users\Anirudh Sharma\Project_work\26-29_09_2017_KCL\SpontaneousDialogue\PD", filee))
    frequencies, times, spectrogram = signal.spectrogram(samples2, sample_R)

    plt.pcolormesh(times, frequencies, spectrogram,cmap="rainbow", shading = "auto")
    np.log(spectrogram)
    plt.savefig('sp_xyz.png',
                dpi=10,
                frameon='false',
                bbox_inches='tight',
                pad_inches=0) 
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


# In[9]:


get_ipython().system('pip install opensmile')


# In[10]:


#Extracting features using opensmile to know their labels and actual meaning for Spontaneous Dialogue of healthy controls
df_list = []
hc_path = "C:/Users/Anirudh Sharma/Project_work/26-29_09_2017_KCL/SpontaneousDialogue/HC"
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.GeMAPSv01b,
    feature_level=opensmile.FeatureLevel.Functionals,
)
files = os.listdir(hc_path)
for file in files:
    y = smile.process_file(file)
    df_list.append(y)
    
df_list


# In[11]:


HC_df = pd.concat(df_list)
HC_df['Label'] = 0
HC_df


# In[12]:


#Extracting features using opensmile to know their labels and actual meaning for Spontaneous Dialogue of Parkinson's disease
df_list1 = []
hc_path1 = "C:/Users/Anirudh Sharma/Project_work/26-29_09_2017_KCL/SpontaneousDialogue/PD/"
smile1 = opensmile.Smile(
    feature_set=opensmile.FeatureSet.GeMAPSv01b,
    feature_level=opensmile.FeatureLevel.Functionals,
)
files1 = os.listdir(hc_path1)
for file1 in files1:
    y1 = smile1.process_file(file1)
    df_list1.append(y1)
    
df_list1


# In[13]:


PD_df = pd.concat(df_list1)
PD_df['Label'] = 1
PD_df


# In[14]:


final = pd.concat([HC_df[:15], PD_df])
final


# In[24]:


X_train, x_test, y_train, y_test = train_test_split(final[['F0semitoneFrom27.5Hz_sma3nz_amean']], final[['Label']], test_size=0.2)


# In[25]:


clf = svm.SVC()
clfnew = clf.fit(X_train,y_train.values.ravel())
clfnew


# In[26]:


y_pred = clf.predict(x_test)
y_pred


# In[27]:


tscore = abs(clf.score(x_test,y_test))
tscore


# In[28]:


#Final score
print("Score is: ", tscore)
#For mean squared error
print("The mean squared error is:", mean_squared_error(y_test, y_pred, squared=False))

#For R2 Score
print("The R square score is:",r2_score(y_test, y_pred))

#For classification report
print(classification_report(y_test, y_pred,  digits=3, zero_division=0))


# In[29]:


#PART 2


# In[30]:


#Extracting features using opensmile to know their labels and actual meaning for read text HC
df_list2 = []
hc_path2 = "C:/Users/Anirudh Sharma/Project_work/26-29_09_2017_KCL/ReadText/HC"
smile2 = opensmile.Smile(
    feature_set=opensmile.FeatureSet.GeMAPSv01b,
    feature_level=opensmile.FeatureLevel.Functionals,
)
files2 = os.listdir(hc_path2)
for file2 in files2:
    y2 = smile2.process_file(file2)
    df_list2.append(y2)
    
df_list2


# In[31]:


HC_df2 = pd.concat(df_list2)
HC_df2['Label'] = 0
HC_df2


# In[32]:


#Extracting features using opensmile to know their labels and actual meaning for read text PD
df_list3 = []
hc_path3 = "C:/Users/Anirudh Sharma/Project_work/26-29_09_2017_KCL/ReadText/PD"
smile3 = opensmile.Smile(
    feature_set=opensmile.FeatureSet.GeMAPSv01b,
    feature_level=opensmile.FeatureLevel.Functionals,
)
files3 = os.listdir(hc_path3)
for file3 in files3:
    y3 = smile3.process_file(file3)
    df_list3.append(y3)
    
df_list3


# In[33]:


PD_df3 = pd.concat(df_list3)
PD_df3['Label'] = 1
PD_df3


# In[34]:


final2 = pd.concat([HC_df2[:15], PD_df3])
final2


# In[35]:


#For read only files
X_train2, x_test2, y_train2, y_test2 = train_test_split(final2[['F0semitoneFrom27.5Hz_sma3nz_amean']], final2[['Label']], test_size=0.2)


# In[36]:


clf2 = svm.SVC()
clfnew2 = clf2.fit(X_train2,y_train2.values.ravel())
clfnew2


# In[37]:


y_pred2 = clf2.predict(x_test2)
y_pred2


# In[38]:


tscore2 = abs(clf2.score(x_test2,y_test2))
tscore2


# In[39]:


#score for model
print("Score is:", tscore2)
#For mean squared error
print("The mean squared error score is:", mean_squared_error(y_test2, y_pred2, squared=False))

#For R2 Score
print("The R Square score is:",r2_score(y_test2, y_pred2))

#CLASSIFICATION REPORT ( F1, RECALL, PRECISION)
print(classification_report(y_test2, y_pred2,  digits=3, zero_division=0))


# In[40]:


#New modeling for shuffled data


# In[41]:


#For read only files
X_train3, x_test3, y_train3, y_test3 = train_test_split(final2[['F0semitoneFrom27.5Hz_sma3nz_percentile20.0']], final2[['Label']], test_size=0.2, shuffle = True)


# In[42]:


#SELECTING A MODEL TO TRAIN
clf3 = svm.SVC()
clfnew3 = clf3.fit(X_train3,y_train3.values.ravel())
clfnew3

#For prediction 
y_pred3 = clf3.predict(x_test3)
y_pred3

#Score
tscore3 = abs(clf3.score(x_test3,y_test3))


# In[43]:


#Score
print("Score is:", tscore3)

#MEAN SQUARED ERROR
print("The mean squared error score is:", mean_squared_error(y_test3, y_pred3, squared=False))

#R2 SCORE
print("The R Square score is:", r2_score(y_test3, y_pred3))

#CLASSIFICATION REPORT ( F1, RECALL, PRECISION)
print(classification_report(y_test3, y_pred3,  digits=3, zero_division=0))


# In[44]:


#For Spontaneous dialogue only files
X_train33, x_test33, y_train33, y_test33 = train_test_split(final[['F0semitoneFrom27.5Hz_sma3nz_percentile20.0']], final[['Label']], test_size=0.2, shuffle = True)


# In[45]:


#Implying a model
clf33 = svm.SVC()
clfnew33 = clf33.fit(X_train33,y_train33.values.ravel())
clfnew33

#Making predictions
y_pred33 = clf33.predict(x_test33)
y_pred33

#score
tscore4 = abs(clf33.score(x_test33,y_test33))


# In[46]:


#Final score
print("Score is:", tscore4)

#MEAN SQUARED ERROR
print("The mean squared error score is:", mean_squared_error(y_test33, y_pred33, squared=False))

#R2 SCORE
print("The R Square score is:", r2_score(y_test33, y_pred33))

#CLASSIFICATION REPORT ( F1, RECALL, PRECISION)
print(classification_report(y_test33, y_pred33,  digits=3, zero_division=0))


# In[ ]:


#AFTER USING ALL THE FEATURES


# In[67]:


#For Spontaneous dialogue files
X_train7, x_test7, y_train7, y_test7 = train_test_split(final.iloc[:,:-1], final.iloc[:,-1], test_size=0.2)


# In[68]:


#For model training
clf7 = svm.SVC()
clfnew7 = clf7.fit(X_train7,y_train7.values.ravel())
clfnew7


# In[69]:


#For prediction
y_pred7 = clf7.predict(x_test7)
y_pred7


# In[70]:


tscore9 = abs(clf7.score(x_test7,y_test7))


# In[72]:


#Final score
print("Score is:",tscore9)

#MEAN SQUARED ERROR
print("The mean squared error score is:", mean_squared_error(y_test7, y_pred7, squared=False))

#R2 SCORE
print("The R Square score is:", r2_score(y_test7, y_pred7))

#CLASSIFICATION REPORT ( F1, RECALL, PRECISION)
print(classification_report(y_test7, y_pred7,  digits=3, zero_division=0))


# In[48]:


#For read only files
X_train5, x_test5, y_train5, y_test5 = train_test_split(final2.iloc[:,:-1], final2.iloc[:,-1], test_size=0.2)


# In[49]:


#For model training
clf5 = svm.SVC()
clfnew5 = clf5.fit(X_train5,y_train5.values.ravel())
clfnew5


# In[50]:


#For prediction
y_pred5 = clf5.predict(x_test5)
y_pred5


# In[51]:


tscore10 = abs(clf5.score(x_test5,y_test5))


# In[52]:


#Final score
print("Score is:",tscore10)

#MEAN SQUARED ERROR
print("The mean squared error score is:", mean_squared_error(y_test5, y_pred5, squared=False))

#R2 SCORE
r11 = print("The R Square score is:", r2_score(y_test5, y_pred5))

#CLASSIFICATION REPORT ( F1, RECALL, PRECISION)
print(classification_report(y_test5, y_pred5,  digits=3, zero_division=0))


# In[ ]:


#After using different feature


# In[73]:


#For Spontaneous files
X_train8, x_test8, y_train8, y_test8 = train_test_split(final[['hammarbergIndexUV_sma3nz_amean']], final[['Label']], test_size=0.2)


# In[74]:


#For model training
clf8 = svm.SVC()
clfnew8 = clf8.fit(X_train8,y_train8.values.ravel())
clfnew8


# In[75]:


#For prediction
y_pred8 = clf8.predict(x_test8)
y_pred8


# In[76]:


tscore8 = abs(clf8.score(x_test8,y_test8))


# In[77]:


#Final score
print("Score is:",tscore8)

#MEAN SQUARED ERROR
print("The mean squared error score is:", mean_squared_error(y_test8, y_pred8, squared=False))

#R2 SCORE
print("The R Square score is:", r2_score(y_test8, y_pred8))

#CLASSIFICATION REPORT ( F1, RECALL, PRECISION)
print(classification_report(y_test8, y_pred8,  digits=3, zero_division=0))


# In[78]:


#For Read only text data
X_train11, x_test11, y_train11, y_test11 = train_test_split(final2[['hammarbergIndexUV_sma3nz_amean']], final2[['Label']], test_size=0.2)


# In[79]:


#For model training
clf11 = svm.SVC()
clfnew11 = clf11.fit(X_train11,y_train11.values.ravel())
clfnew11


# In[80]:


#For prediction
y_pred11 = clf11.predict(x_test11)
y_pred11


# In[81]:


tscore11 = abs(clf11.score(x_test11,y_test11))


# In[82]:


#Final score
print("Score is:",tscore11)

#MEAN SQUARED ERROR
print("The mean squared error score is:", mean_squared_error(y_test11, y_pred11, squared=False))

#R2 SCORE
print("The R Square score is:", r2_score(y_test11, y_pred11))

#CLASSIFICATION REPORT ( F1, RECALL, PRECISION)
print(classification_report(y_test11, y_pred11,  digits=3, zero_division=0))


# In[53]:


#NEW LOGISTIC REGRESSION MODEL


# In[54]:


#For read only files
X_train4, x_test4, y_train4, y_test4 = train_test_split(final2[['F0semitoneFrom27.5Hz_sma3nz_percentile20.0']], final2[['Label']], test_size=0.2, shuffle = True)


# In[55]:


clf4 = LogisticRegression(random_state=0).fit(X_train4, y_train4)
clf4


# In[56]:


clf4pred = clf4.predict(x_test4)
clf4pred


# In[57]:


# abs(clf4.score(x_test4,y_test4))


# In[58]:


#MEAN SQUARED ERROR
print("The mean squared error score is:", mean_squared_error(y_test4, clf4pred, squared=False))

#R2 SCORE
print("The R Square score is:", r2_score(y_test4, clf4pred))

#CLASSIFICATION REPORT ( F1, RECALL, PRECISION)
print(classification_report(y_test4, clf4pred,  digits=3, zero_division=0))


# In[59]:


#For logestic regression of Spontaneous data


# In[60]:


#For read only files
X_train6, x_test6, y_train6, y_test6 = train_test_split(final[['F0semitoneFrom27.5Hz_sma3nz_percentile20.0']], final[['Label']], test_size=0.2, shuffle = True)


# In[61]:


clf6 = LogisticRegression(random_state=0).fit(X_train6, y_train6)
clf6


# In[64]:


clf5pred = clf6.predict(x_test6)
clf5pred


# In[66]:


#MEAN SQUARED ERROR
print("The mean squared error score is:", mean_squared_error(y_test6, clf5pred, squared=False))

#R2 SCORE
print("The R Square score is:", r2_score(y_test6, clf5pred))

#CLASSIFICATION REPORT ( F1, RECALL, PRECISION)
print(classification_report(y_test6, clf5pred,  digits=3, zero_division=0))


# In[ ]:


#CODE REFERENCES 
#1. openSMILE
#2. SKLEARN FOR R2, SVM.SVC, LOGISTIC REGRESSION, MEAN SQUARE ERROR, CLASSIFICATION REPORT

