#!/usr/bin/env python
# coding: utf-8

# In[8]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


df = pd.read_csv("Downloads//Data-Collisions.csv")


# In[11]:


df.head(5)


# In[13]:


df.shape


# In[14]:


df.info()


# In[20]:


c1 = df[df['SEVERITYCODE'] == 1].count()['SEVERITYCODE']
c1


# In[21]:


c2 = df[df['SEVERITYCODE'] == 2].count()['SEVERITYCODE']
c2


# In[22]:



import seaborn as sns


# In[23]:


plt.figure(figsize = (12,6))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[25]:


df.drop(['SPEEDING','SDOTCOLNUM','PEDROWNOTGRNT','INATTENTIONIND'
                    ,'EXCEPTRSNDESC','EXCEPTRSNCODE', 'INTKEY'], axis=1,inplace=True)


# In[26]:


plt.figure(figsize = (12,6))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[27]:


new_collision = df.dropna()


# In[28]:



plt.figure(figsize = (12,6))
sns.heatmap(new_collision.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[29]:


new_collision.shape


# In[30]:



new_collision['INCDATE'] = pd.to_datetime(new_collision['INCDATE'])
new_collision['Hour'] = new_collision['INCDATE'].apply(lambda time: time.hour)
new_collision['Month'] = new_collision['INCDATE'].apply(lambda time: time.month)
new_collision['Day of Week'] = new_collision['INCDATE'].apply(lambda time: time.dayofweek)


# In[31]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
new_collision['Day of Week'] = new_collision['Day of Week'].map(dmap)


# In[32]:


new_collision['Day of Week'].value_counts().plot(kind = 'line')


# In[33]:


new_collision.columns


# In[34]:


fig, ax=plt.subplots()
new_collision['WEATHER'].value_counts().sort_values(ascending=False).head(10).plot.bar(width=0.5, align='center')
plt.xlabel('Weather Conditions')
plt.ylabel('Number of Accidents')
ax.tick_params()
plt.title('Top 10 Weather Condition for accidents')


# In[35]:


plt.figure(figsize = (12,6))
sns.countplot(x = 'ROADCOND', data = new_collision , 
              hue = 'SEVERITYCODE', palette = 'viridis')


# In[36]:


plt.figure(figsize = (20,6))
sns.countplot(x = 'WEATHER', data = new_collision , 
              hue = 'SEVERITYCODE', palette = 'viridis')


# In[37]:


plt.figure(figsize = (10,6))
sns.countplot(x = 'Day of Week', data = new_collision , 
              hue = 'SEVERITYCODE', palette = 'viridis')


# In[38]:



plt.figure(figsize = (20,6))
sns.countplot(x = 'Month', data = new_collision , 
              hue = 'SEVERITYCODE', palette = 'viridis')


# In[39]:


new_collision['Total'] = new_collision['INCKEY'].count()


# In[40]:


plt.figure(figsize = (20,8))
sns.countplot(x = 'COLLISIONTYPE', data = new_collision , 
              hue = 'SEVERITYCODE', palette = 'viridis')


# In[41]:



plt.figure(figsize = (7,6))
set = sns.countplot(x = 'PERSONCOUNT', data = new_collision , 
              hue = 'SEVERITYCODE', palette = 'viridis')
set.set(xlim=(-1,9))


# In[42]:


new_collision.columns


# In[43]:


plt.figure(figsize = (18,6))
set = sns.countplot(x = 'LIGHTCOND', data = new_collision , 
              hue = 'SEVERITYCODE', palette = 'viridis')


# In[44]:


plt.figure(figsize = (7,6))
set = sns.countplot(x = 'PEDCYLCOUNT', data = new_collision , 
              hue = 'SEVERITYCODE', palette = 'viridis')


# In[45]:


plt.figure(figsize = (7,6))
set = sns.countplot(x = 'VEHCOUNT', data = new_collision , 
              hue = 'SEVERITYCODE', palette = 'viridis')
set.set(xlim=(0,6))


# In[46]:


plt.figure(figsize = (7,6))
set = sns.countplot(x = 'Day of Week', data = new_collision , 
              hue = 'ADDRTYPE', palette = 'viridis')


# In[47]:


plt.figure(figsize = (7,6))
set = sns.countplot(x = 'HITPARKEDCAR', data = new_collision , 
              hue = 'SEVERITYCODE', palette = 'viridis')


# In[48]:


new_collision.columns


# In[49]:


log_df = new_collision[['SEVERITYCODE','Month', 'Day of Week','COLLISIONTYPE', 'PERSONCOUNT', 
          'PEDCOUNT', 'PEDCYLCOUNT', 'VEHCOUNT', 'ADDRTYPE','LIGHTCOND','WEATHER', 'ROADCOND',
          'HITPARKEDCAR']]


# In[50]:


log_df.head()


# In[51]:


dmap_day = {'Mon':1,'Tue':2,'Wed':3,'Thu':4,'Fri':5, 'Sat':6,'Sun':7}


# In[52]:



log_df['Day of Week'] = new_collision['Day of Week'].map(dmap_day)


# In[53]:


dmap_collision = {'Parked Car':1,'Angles':2,'Rear Ended':3,'Other':4,'Sideswipe':5, 
                  'Left Turn':6,'Left Turn':7, 'Pedestrian':8, 'Cycles':9, 'Right Turn': 10,
                 'Head On': 11}

log_df['COLLISIONTYPE'] = log_df['COLLISIONTYPE'].map(dmap_collision)
dmap_address = {'Block':1,'Intersection':2}

log_df['ADDRTYPE'] = log_df['ADDRTYPE'].map(dmap_address)
log_df['ADDRTYPE'].value_counts()


# In[54]:


dmap_light = {'Daylight':1,'Dark - Street Lights On':2,'Unknown':3,'Dusk':4,'Dawn':5, 
                  'Dark - No Street Lights':6,'Dark - Street Lights Off':7, 
              'Other':8, 'Dark - Unknown Lighting':9}

log_df['LIGHTCOND'] = log_df['LIGHTCOND'].map(dmap_light)
log_df['WEATHER'].value_counts()


# In[55]:



dmap_weather = {'Clear':1,'Raining':2,'Overcast':3,'Unknown':4,'Snowing':5, 
                  'Other':6,'Fog/Smog/Smoke':7, 
              'Sleet/Hail/Freezing Rain':8, 'Blowing Sand/Dirt':9, 'Severe Crosswind':10,
               'Partly Cloudy': 11}

log_df['WEATHER'] = log_df['WEATHER'].map(dmap_weather)
dmap_roadcond = {'Dry':1,'Wet':2,'Unknown':3,'Ice':4,'Snow/Slush':5, 
                  'Other':6,'Standing Water':7, 
              'Sand/Mud/Dirt':8, 'Oil': 9}

log_df['ROADCOND'] = log_df['ROADCOND'].map(dmap_roadcond)
dmap_hitcar = {'N':1,'Y':2}

log_df['HITPARKEDCAR'] = log_df['HITPARKEDCAR'].map(dmap_hitcar)


# In[56]:


log_df.head()


# In[57]:



log_df.columns


# In[58]:


#predictive model
from sklearn.model_selection import train_test_split


# In[59]:



X = log_df[['Month', 'Day of Week', 'COLLISIONTYPE', 'PERSONCOUNT',
       'PEDCOUNT', 'PEDCYLCOUNT', 'VEHCOUNT', 'ADDRTYPE', 'LIGHTCOND',
       'WEATHER', 'ROADCOND', 'HITPARKEDCAR']]
y = log_df['SEVERITYCODE']


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[61]:


from sklearn.linear_model import LogisticRegression


# In[62]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[63]:


predictions = logmodel.predict(X_test)


# In[64]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))


# In[65]:


print(confusion_matrix(y_test,predictions))


# In[66]:


logmodel.score(X_test, y_test)


# In[ ]:




