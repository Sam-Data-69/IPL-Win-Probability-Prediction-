#!/usr/bin/env python
# coding: utf-8

# In[91]:


import numpy as np 
import pandas as pd 


# In[92]:


match = pd.read_csv(r"C:\Users\Sambhav\Downloads\matches (1).csv")
delivery = pd.read_csv(r"C:\Users\Sambhav\Downloads\deliveries.csv")


# In[93]:


match.head()


# In[94]:


delivery.head()


# In[95]:


match.shape


# In[96]:


delivery.shape


# In[97]:


total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
total_score_df


# In[98]:


total_score_df = total_score_df[total_score_df['inning'] == 1]
total_score_df


# In[99]:


match_df = match.merge(total_score_df[['match_id','total_runs']], left_on='id', right_on='match_id')
match_df.head()


# In[100]:


match_df['team1'].unique()


# In[101]:


teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]


# In[102]:


match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')


# In[103]:


match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]


# In[104]:


match_df.shape


# In[105]:


match_df.head()


# In[106]:


match_df['dl_applied'].value_counts()


# In[107]:


match_df = match_df[match_df['dl_applied'] == 0]
match_df


# In[108]:


match_df = match_df[['match_id', 'city', 'winner', 'total_runs']]


# In[109]:


delivery_df = match_df.merge(delivery, on='match_id')


# In[110]:


delivery_df.head()


# In[111]:


delivery_df = delivery_df[delivery_df['inning'] == 2]


# In[112]:


delivery_df.head()


# In[113]:


delivery_df.shape


# In[114]:


delivery_df.groupby('match_id')['total_runs_y'].cumsum()


# In[115]:


delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()


# In[116]:


delivery_df


# In[117]:


delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']


# In[118]:


delivery_df


# In[119]:


126 - (delivery_df['over']*6 + delivery_df['ball'])


# In[120]:


delivery_df['balls_left'] = 126 - (delivery_df['over']*6 + delivery_df['ball'])


# In[121]:


delivery_df


# In[122]:


delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x:x if x == "0" else "1")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')

wickets = delivery_df.groupby('match_id')['player_dismissed'].cumsum().values
delivery_df['wickets'] = 10 - wickets

delivery_df.head()


# In[123]:


# Creating crr column (current runrate (crr) = runs/over)

(delivery_df['current_score']*6) / (120 - delivery_df['balls_left'])


# In[124]:


delivery_df['crr'] = (delivery_df['current_score']*6) / (120 - delivery_df['balls_left'])


# In[125]:


delivery_df


# In[126]:


# Creating rrr column (Requaried runrate)

delivery_df['rrr'] = (delivery_df['runs_left']*6)/delivery_df['balls_left']


# In[127]:


delivery_df


# In[128]:


def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0


# In[129]:


delivery_df.apply(result, axis=1)


# In[130]:


delivery_df['result'] = delivery_df.apply(result, axis=1)


# In[131]:


delivery_df


# In[132]:


# Joo Joo hume columns cahiye vo hum nikal lenge alag se 
final_df = delivery_df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x',
                        'crr', 'rrr', 'result']]


# In[133]:


final_df


# In[134]:


# Shuffling the data 
final_df = final_df.sample(final_df.shape[0])


# In[135]:


final_df


# In[136]:


# Example

final_df.sample()


# In[137]:


final_df.isnull().sum()


# In[138]:


final_df.dropna(inplace=True)


# In[139]:


final_df.isnull().sum()


# In[140]:


final_df = final_df[final_df['balls_left'] != 0]


# In[141]:


# Building model 

X = final_df.iloc[:,:-1]  # --> every rows & every columns except last column (multiple indipendent variable)
y = final_df.iloc[:,-1]   # --> every rows & only last column (feature column)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[142]:


X_train


# In[143]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first'),
     ['batting_team', 'bowling_team', 'city'])
],
remainder='passthrough')


# In[144]:


# Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# In[145]:


pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])


# In[146]:


pipe.fit(X_train, y_train)


# In[147]:


X_train.describe()


# In[148]:


y_pred = pipe.predict(X_test)


# In[149]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[150]:


pipe.predict_proba(X_test)[7]


# In[151]:


# Random Forest 

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# In[152]:


pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',RandomForestClassifier())
])


# In[153]:


pipe.fit(X_train, y_train)


# In[154]:


y_pred = pipe.predict(X_test)


# In[155]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[169]:


pipe.predict_proba(X_test)[7]


# In[170]:


def match_progression(x_df, match_id, pipe):
    # Filter data for the given match
    match = x_df[x_df['match_id'] == match_id]

    # Consider data only after the 6th ball
    match = match[(match['ball'] == 6)]

    # Select required features
    temp_df = match[['batting_team', 'bowling_team', 'city', 
                     'runs_left', 'balls_left', 'wickets', 'total_runs_x','rrr', 'crr']]

    # Filter rows where balls_left is not zero
    temp_df = temp_df[temp_df['balls_left'] != 0]

    # Predict win probabilities using the trained model
    result = pipe.predict_proba(temp_df)

    # Add probabilities into dataframe
    temp_df['lose'] = np.round(result.T[0] * 100, 1)
    temp_df['win'] = np.round(result.T[1] * 100, 1)

    # Add over progression
    temp_df['end_of_over'] = range(1, temp_df.shape[0] + 1)

    # Target score
    target = temp_df['total_runs_x'].values[0]

    # Runs left
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0, target)

    # Calculate runs scored in each over
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)

    # Wickets fallen
    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0, 10)
    wickets.append(0)

    # Convert to numpy arrays
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[:temp_df.shape[0]]
    
    print('Target-', target)

    # fixed column names
    temp_df = temp_df[['end_of_over', 'runs_after_over', 'wickets_in_over', 'lose', 'win']]
    return temp_df, target


# In[176]:


temp_df,target = match_progression(delivery_df,7,pipe)
temp_df


# In[179]:


import matplotlib.pyplot as plt 

plt.figure(figsize=(18,8))
plt.plot(temp_df['end_of_over'],temp_df['wickets_in_over'],color='yellow',linewidth=3)
plt.plot(temp_df['end_of_over'],temp_df['win'],color='#00a65a',linewidth=4)
plt.plot(temp_df['end_of_over'],temp_df['lose'],color='red',linewidth=4)
plt.bar(temp_df['end_of_over'], temp_df['runs_after_over'])
plt.title(f"Target - {target}")


# In[178]:


import matplotlib.pyplot as plt 

plt.figure(figsize=(18,8))

# wickets progression
plt.plot(temp_df['end_of_over'], temp_df['wickets_in_over'], 
         color='yellow', linewidth=3, label="Wickets in Over")

# win probability
plt.plot(temp_df['end_of_over'], temp_df['win'], 
         color='#00a65a', linewidth=4, label="Win %")

# lose probability
plt.plot(temp_df['end_of_over'], temp_df['lose'], 
         color='red', linewidth=4, label="Lose %")

# runs per over
plt.bar(temp_df['end_of_over'], temp_df['runs_after_over'], 
        alpha=0.4, label="Runs per Over")

# add title
plt.title(f"Target - {target}", fontsize=18)
plt.xlabel("Over", fontsize=14)
plt.ylabel("Value", fontsize=14)

plt.legend(fontsize=12)
plt.show()


# In[ ]:




