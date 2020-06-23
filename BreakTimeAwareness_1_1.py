#!/usr/bin/env python
# coding: utf-8

# In[192]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mat
from pymongo import MongoClient
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np


# In[193]:


##Vi opretter forbindelsen til vores MongoDB server hvor vi peger på en specifik collection.
cluster = MongoClient("mongodb+srv://thejokerd3:sxp78gaf@breaktimeawarenesscluster-rwpry.mongodb.net/test?retryWrites=true&w=majority")
db = cluster["BreakTimeDB"]
collection = db["BreakTimeCollectionV2"]


# In[194]:


##Så henter vi vores datasæt ned som et pandas dataframe
df = pd.DataFrame(list(collection.find()))


# In[195]:


##For at teste at vi har korrekt hentet vores datasæt bruger vi head for at få de første 5 rækker ud
df.head(5)

##Printer en summary statistik
df_description = df.describe()

print("\n")

##Printer dataframe information
df_info = df.info()

print("\n")


# In[196]:


##Da vi printede de 5 første rækker ud så vi at der var en kolonne ID - den vil vi gerne fjerne
del df['_id']


# In[197]:


##Vi printer ud igen for at se om det ser korrekt ud vi printer 17 ud denne gang
df.head(17)


# In[198]:


##Vi kan se at der er nogle missing values - de er erstattet med NaN.
##Vi finder først ude af hvilken features indeholder NaN værdier

##df['sex'].isnull().values.any()
##df['hours_slept'].isnull().values.any() -> indeholder
##df['noise_level'].isnull().values.any() -> indeholder 
##df['light_level'].isnull().values.any() -> indeholder
##df['temperature_level'].isnull().values.any() -> indeholder
##df['needs_pause'].isnull().values.any()


##Vi vil gerne finde median så vi kan erstatte alle NaN værdier
hourssleptmedian = df['hours_slept'].mean()
noiselevelmedian = df['noise_level'].mean()
lightlevelmedian = df['light_level'].mean()
temperaturelevelmedian = df['temperature_level'].mean()
print('This is the median for hours slept')
print(hourssleptmedian)
print('')
print('This is the median for noise level')
print(noiselevelmedian)
print('')
print('This is the median for light level')
print(lightlevelmedian)
print('')
print('This is the median for temperature level')
print(temperaturelevelmedian)
print('')

## efter at have fundet medianen for de forskellige nan værdier i de forskellige features vil vi gerne erstatte dem

df['hours_slept'] = df['hours_slept'].fillna(7)
df['noise_level'] = df['noise_level'].fillna(56)
df['light_level'] = df['light_level'].fillna(1233)
df['temperature_level'] = df['temperature_level'].fillna(26)

## så printer vi de først 20 rækker ud for at tjekke at de missing values er blevet replaced - erstattet
df.head(20)

##Derefter tæller vi antallet af NaN's vi har i vores datasæt og printer counten ud for at verificere
print("Number of NaNs in dataset")
print(df.isnull().values.sum())


# In[199]:


##Da vi arbejder med numeriske værdier kan vi se at i vores kolonne for sex indeholder tekst værdier - dem vil vi gerne erstatte
##Med numeriske værdier
df["sex"].replace({"Female": 0, "Male": 1}, inplace=True)

##Vi printer ud for at se om det lykkes
df.head(20)


# In[200]:


##Vi vil gerne visualisere vores data for at finde sammenhæng mellem de enkelte features samt om der sammenhæng
## mellem featursenes værdier og om der skal holdes pause

##Så vi laver et array med 50 variabler om der holdes pause
y = df['needs_pause'].head(50)
##så laver vi et nyt dataframe med de første 50 rækker fordi at datasættet er for stort til at det giver en ordentlig overblik
dfv2 = df.head(50)
## vi kan ikke bruge de her 3 featurs da de første 50 datapunkter er for den samme person og samme dag.
## og vi fjerne needs_pause label fordi vi allerede har det som vores y-værdi
del dfv2['needs_pause']
del dfv2['sex']
del dfv2['hours_slept']
##vi normalisere vores datasæt
df_norm_col=(dfv2-dfv2.mean())/dfv2.std()
##Anvender panda library til at lave et scatter matrix for at se om der er sammenhæng mellem lys niveauet, temperature niveauet og lyd niveauet
pd.plotting.scatter_matrix(df_norm_col, c=y, figsize = [8, 8], s=150, marker='D')


# In[201]:


##Vi laver derefter et heapmap for at finde ude af hvornår folk holder pause.
dfv3 = df.head(25)
del dfv3['sex']
del dfv3['hours_slept']
df_norm_col=(dfv3-dfv3.mean())/dfv3.std()
sns.heatmap(df_norm_col, cmap='coolwarm',
                annot=True,
                fmt=".1f",
                annot_kws={'size':9},
                cbar=False)
## som der kan ses på heatmappet fandt vi ude af at der er tendens til at holde pause når værdierne er i den lave ende


# In[202]:


##Herefter vil vi statistisk set kigge på korrelationen mellem de 3 features.
dfv4 = df.head(48000)
del dfv4['time_at_reading']
dfv4.corr()


# In[203]:


##Vi har færdiggjortt visualiserings delen. Grunden til at dette blev lavet er for at finde sammenhæng mellem features
## som skulle give os et grundlag for hvilken algoritme vi gerne vil anvende

##Hvorfor vi ikkke har tænkt os at vælge Logistick regression:
#       Der er ikke god sammenhæng mellem nogle af vores features og vores label udfaldet variere meget

##Hvorfor har vi ikke tænkt os at bruge KNN:
#       KNN angiver hvilken kategori et punkt høre til udefra de tætteste omkringlæggende punkter
#       efter vores data består af ca 95% punkter hvor der ikke skal holdes pause vil udfaldet i næsten alle tilfælde være
#       at der ikke skal holdes pause. Der er ikke nogle gruppering i de punkter hvor der holdes pause
#       det kan ses på vores scattermatrix
#       vi fandt frem til at fordi at dataen er så randomized / tilfædigt vil det gøre det svært at bruge KNN
#       Dataen er for tilfædigt og det kan ses også på heatmappet hvor selvom at 2 steder kan man se at lave tal giver en pause
#       så er der også mange andre lave tal hvor det ikke gøre. 

##Hvorfor vi ikke bruger desicion trees
#       svært at gennemskue og forstå hvorfor modellen genere et specifik udfald
#       er sårbar over for overfitting og i vores tilfælde vil der altid være at der ikke skal holdes pause


# In[204]:


##Vi har derfor valgt et neuralt netværk. Der er komplekse sammenhæng mellem mange af vores features.
## Vi arbejder med neurale netværk når vi har komplekse relationer samt kan vi tune vores model så vi kan undgå overfitting/underfitting
## Også effektiv i tilfælde hvor der er komplekse relationer mellem features og outputs og det kan man se i vores corr table
## hvor alle features har corr under 0.1 til vores output


# In[205]:


## Nu vil vi gerne opdele vores datasæt i træningsdata og testdata - vi bruger træningsdata til at træne vores model
## hvor testdata er for at afprøve vores model og finde ude af hvor præcis den er til at forudsige. 

##Vores x variable indeholder vores 5 første features i vores tabel
## y variablen indeholder vores label

del df['sex']
del df['hours_slept']

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

## Vi instansiere et train_split objekt fra sklearn biblioteket som opdeler vores datasæt
## de parametre den får er featurs og labels som X og y samt procent opdeling som er 30% test 70% træning
## derefter angiver man en random state som er hvordan den bliver opdelt så vi er konsistent med det så vi har den samme opdeling hver gang
## stratify gøre at labels og features matcher med det oprindelig datasæt

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify=y)

adam = Adam(learning_rate=0.001)


##vi har anvendt de forskellige activation function som: selu, elu, softmax, softplus
model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer=adam, metrics="AUC")

history = model.fit(X_train, y_train, validation_split=0.2, batch_size=128, epochs=100)

scores = model.evaluate(X_test, y_test)

print("\n")
print(scores)


# In[206]:


plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()



# In[207]:


## Nu vil vi gerne opdele vores datasæt i træningsdata og testdata - vi bruger træningsdata til at træne vores model
## hvor testdata er for at afprøve vores model og finde ude af hvor præcis den er til at forudsige. 

##Vores x variable indeholder vores 5 første features i vores tabel
## y variablen indeholder vores label

#del df['sex']
#del df['hours_slept']

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

## Vi instansiere et train_split objekt fra sklearn biblioteket som opdeler vores datasæt
## de parametre den får er featurs og labels som X og y samt procent opdeling som er 30% test 70% træning
## derefter angiver man en random state som er hvordan den bliver opdelt så vi er konsistent med det så vi har den samme opdeling hver gang
## stratify gøre at labels og features matcher med det oprindelig datasæt

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify=y)

adam = Adam(learning_rate=0.001)


##vi har anvendt de forskellige activation function som: selu, elu, softmax, softplus
model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

##accuracy er en dårlig måde at måle på fordi at der er et stort overtal hvor der ikke holdes pause
##så vil den få en høj accuracy selvom den sagde den aldrig skal holde pause
model.compile(loss="binary_crossentropy", optimizer=adam, metrics="accuracy")

history = model.fit(X_train, y_train, validation_split=0.2, batch_size=128, epochs=100)

scores = model.evaluate(X_test, y_test)

print("\n")
print(scores)


# In[208]:


predictions_NN_prob = model.predict(X_test)
predictions_NN_prob = predictions_NN_prob[:,0]

false_positive_rate, recall, thresholds = roc_curve(y_test, predictions_NN_prob)
roc_auc = auc(false_positive_rate, recall)
plt.figure()
plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')
plt.show()

##den røde linje repræsentere fuld random så jo tætte på røde linje jo dårlgiere er den 


# In[212]:


predictions_NN_01 = np.where(predictions_NN_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output

cm = confusion_matrix(y_test, predictions_NN_01)
labels = ['negative', 'positive']
plt.figure(figsize=(8,6))
sns.heatmap(cm,xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cmap="Blues", vmin = 0.2);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()





