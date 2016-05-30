from __future__ import division
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import train_test_split, cross_val_score, LeaveOneOut,cross_val_predict
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import StratifiedKFold


mydir = os.path.expanduser("~/github/DataSciNotes/Kaggle/2016_US_Election/")

countyPath = mydir + 'data/2016_presidential_election/county_facts.csv'
resultsPath = mydir + 'data/2016_presidential_election/primary_results.csv'

importCounty = pd.read_csv(countyPath)
importResults = pd.read_csv(resultsPath)

importCounty_VA =  importCounty.loc[importCounty['state_abbreviation'] == 'VA']

importResults_VA = importResults.loc[importResults['state_abbreviation'] == 'VA']
importResults_VA = importResults_VA.drop(['state', 'state_abbreviation'], axis=1)

#print importResults_VA.pivot_table(values=['votes','fraction_votes'], index='county', columns=['party','candidate'])
#print importResults_VA.pivot(index=None, columns=['party'], values=['votes', 'fraction_votes'])

def tableSubset(importResults_VA):
    countyNames = set(importResults_VA['county'].values)
    dictSub = {}
    for x in countyNames:
        rep = importResults_VA.loc[(importResults_VA['county'] == x) & (importResults_VA['party'] == 'Republican')]
        rep_cand =  rep.loc[rep['fraction_votes'].idxmax()][3]
        rep_fract =  rep.loc[rep['fraction_votes'].idxmax()][5]
        dem = importResults_VA.loc[(importResults_VA['county'] == x) & (importResults_VA['party'] == 'Democrat')]
        dem_cand =  dem.loc[dem['fraction_votes'].idxmax()][3]
        dem_fract =  dem.loc[dem['fraction_votes'].idxmax()][5]
        returnList = [dem_cand, dem_fract,rep_cand, rep_fract]
        county_name = x + ' County'
        dictSub[county_name] = returnList
    df =  pd.DataFrame.from_dict(dictSub)
    df = df.transpose()
    df.reset_index(level=0, inplace=True)
    df.columns = ['area_name', 'dem_cand', 'dem_fract', 'rep_cand', 'rep_fract']
    return df

def plotResiduals(y_train, y_test, y_train_pred, y_test_pred, filename = 'Fig1'):
    plt.scatter(y_train_pred, y_train_pred - y_train, \
        c = 'blue', marker = 'o', label = 'Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, \
        c = 'lightgreen', marker = 's', label = 'Test data')
    plt.xlabel('Predicted expression')
    plt.ylabel('Residuals')
    plt.legend(loc = 'upper left')
    plt.hlines(y = 0, xmin = -10, xmax = 50, lw =2, color = 'red')
    plt.xlim([-5, 5])
    plt.savefig(str(mydir) + 'figs/' +  filename + ".png")
    plt.close()

def pairPlot(df, filename = 'Fig2'):
    cols = list(df.columns.values)
    sns_plot = sns.pairplot(df[cols], size=2.5)
    sns_plot.savefig(str(mydir) + 'figs/' + filename + ".png")

def log10(x):
    return np.log10(x)

def replace(g):
    mask = g == 0
    g.loc[mask] = g[~mask].mean()
    return g

dfReturn = tableSubset(importResults_VA)
dfFinal = pd.merge(importCounty_VA, dfReturn, on='area_name')

class_map_dem = {label:idx for idx,label in enumerate(np.unique(dfFinal['dem_cand']))}
class_map_rep = {label:idx for idx,label in enumerate(np.unique(dfFinal['rep_cand']))}

dfFinal['dem_cand'] = dfFinal['dem_cand'].map(class_map_dem)
dfFinal['rep_cand'] = dfFinal['rep_cand'].map(class_map_rep)

#print dfFinal
X = dfFinal.iloc[:, 3:54]
y_dem = dfFinal.iloc[:, 54]
y_repub = dfFinal.iloc[:, 56]

toTransform = ['POP060210', 'BPS030214', 'AFN120207', 'RTN130207', 'WTN220207', 'MAN450207', \
    'NES010213' ,'SBO001207', \
    'BZA010213', 'BZA110213', 'HSD410213', 'HSG096213', 'HSG010214', \
    'VET605213', 'PST045214', 'PST040210', 'POP010210']

X[toTransform] = X[toTransform].apply(log10)

X = X.replace([np.inf, -np.inf], np.nan)
# replace inf with mean
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp = imp.fit(X)
# Impute our data, then train
X_imp = imp.transform(X)
X_imp = pd.DataFrame(X_imp)
X_imp.columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(X_imp, y_repub, test_size = 0.3, \
    random_state = 0)

slr = LogisticRegression()

slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)
plotResiduals(y_train, y_test, y_train_pred,  y_test_pred, filename = 'Fig2')

rfecv = RFECV(estimator=slr, step=1, cv=StratifiedKFold(y_repub, 2))
rfecv = rfecv.fit(X_imp, y_repub)
print("Optimal number of features : %d" % rfecv.n_features_)
print "RFECV ranking: " + str(rfecv.ranking_)
print "RFECV support: " + str(rfecv.support_)


def test():
    toSlice = []
    for i, j in enumerate(rfecv.ranking_):
        if j == 1:
            toSlice.append(i)
    return toSlice

# slice columsn with important features

X_imp_features = X_imp.iloc[:, test()]

y_repub = y_repub.to_frame()
#RHI125214 = percent white along
#RHI125214 = percent white along, excluding hispanics
# POP815213 = language other than english spoken at home
# EDU635213 = percent high school graduate or higher
# EDU685213 = percent bachelors or higher
# LFE305213 = mean communting to work time
#HSG445213 = homeownership rate
# PVY020213 = persons below poverty level

#print X_imp_features['POP815213']
X_imp_features['Republican'] = y_repub

trump = np.asarray(X_imp_features.query('Republican == 0')['POP815213'])
rubio = np.asarray(X_imp_features.query('Republican == 1')['POP815213'])
toPlot = [trump, rubio]
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(toPlot)
ax.set_xticklabels(['Trump', 'Rubio'])
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.set_ylabel('Language other than English spoken at home, pct age 5+, 2009-2013')
# Save the figure
fig.savefig('nonEnglish.png', bbox_inches='tight')
