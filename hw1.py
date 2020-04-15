import os
import gzip
import numpy as np
from sklearn.linear_model import *
from sklearn.preprocessing import StandardScaler

import pickle

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import joblib

def optimize(model, x_train, y_train, x_test, y_test, cv=None):
    '''
    model = model instance to use with specified solver, regularization etc.
    x = samples, features; these are observations (images) with # samples = # measurements and features being the attributes of that measurement
    y = samples; these are the _labels_ for the observations of same size as samples
    '''
    model.fit(x_train, y_train)
    predictions=model.predict(x_test)
    score=model.score(x_test, y_test)
    
    if cv is not None:
        from sklearn.model_selection import cross_val_score
        # Run cv number of simulations using k-fold cross validation approach
        cv_scores=cross_val_score(model, x_train, y_train, cv=cv)
        return (model, predictions, score, cv_scores)    
    else:
        return (model, predictions, score)
    

def compute_loss(model, predictions, testX, testY):
    '''
    Assess model accuracy
    '''
    report=metrics.classification_report(testY, predictions)
    matrix=metrics.confusion_matrix(testY, predictions, normalize='true')
    return (report, matrix)

def plot_confusion(confusion_matrix, acc, save_fn=None):
    '''
    Use seaborn to plot confusion matrix
    '''
    plt.figure(figsize=(6,6))
    sns.heatmap(confusion_matrix*100, annot=True, fmt=".1f", linewidths=.5, square = True, 
                robust = True,cmap = 'Blues', cbar_kws={'label': 'Percent accuracy'}, 
               );

    plt.ylabel('True label');
    plt.xlabel('Predicted label');

    plt.xticks(weight="semibold");
    plt.yticks(rotation=0, weight="semibold");

    plt.tick_params(axis='both', which='both', length=0);

    title = str('Overall Accuracy: '+ "{0:.1f}".format(acc*100)+"%")
    plt.title(title, size = 15);

    if save_fn is not None:
        print("Saving as", save_fn.split("/")[-1])
        plt.savefig(save_fn, format='png', facecolor='white', edgecolor='none', dpi=300)
        
def plot_vip(trained_model, p=None):
    '''
    Check out very important pixels --> needs tweaking
    '''
    coef=trained_model.coef_
    coef=coef.reshape(10,28,28)
    clim=max(np.abs(coef.min()), np.abs(coef.max()))

    if p is None:
        p=2.5
    
    title=str(p*2)
    
    fig, axes = plt.subplots(2, 5, 
                             figsize=(8,6),
                             sharex=True, 
                             sharey=True
                            )

    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        cmin, cmax = (np.percentile(coef[i],p), np.percentile(coef[i],100-p))
        im = ax.imshow(np.ma.masked_inside(coef[i], cmin, cmax), 
                       vmin=-clim, vmax=clim, cmap=plt.cm.RdGy)
        ax.set_title(i, fontsize=14)

    cb_ax=fig.add_axes([0.92, 0.1, 0.03, 0.8])
    cbar=fig.colorbar(im, cax=cb_ax)
    
    plt.suptitle("Digit importance - top " + title + "%", fontsize=15, fontweight="bold", y=0.93)
    
def hist_coef(trained_model):
    coef=trained_model.coef_
    fig, axes = plt.subplots(2, 5,
                         figsize=(10,4),
                         sharex=True,
                         sharey=True
                        )

    for i, ax in enumerate(axes.flat):
        ax.hist(coef[i])
        ax.set_title(i, pad=2.5)

    plt.suptitle("Potential for sparsification", fontsize=16, y=1.01);
    
def fandr(in_img, indices):
    '''
    Return input image with only values at specified indices
    '''
    new_img=np.zeros_like(in_img)
    new_img[indices] = in_img[indices]
    return(new_img)