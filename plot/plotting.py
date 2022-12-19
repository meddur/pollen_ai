#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:32:02 2022

@author: mederic
"""
import os 
import pandas as pd
import matplotlib.pyplot as plt
#https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
from scipy.ndimage.filters import gaussian_filter1d

#%%
os.chdir("./plot/")

classes_select = [
                     # "abb_pic_mix",
                    "abies_b",
                    "picea_mix",
                    
                     # "pinus_mix",
                    "pinus_b_mix",
                    "pinus_s",
                    
                    # "tricolp_mix",
                    "acer_mix",
                    "quercus_r",
                    # "acer_r",
                    # "acer_s",
                     'alnus_mix',
                    # "alnus_c",
                    # "alnus_r",
                      "betula_mix",
                        "corylus_c",
                        "eucalyptus",
                        "juni_thuya",
                    # "populus_d",

                    # 'tricolp_mix',
                    #'vesiculate_mix',
                    
        ] 
#%%

######
#Add age data
#CODE R MARCHE AUSSI BIEN

dataframe = pd.read_csv("taxa_sum_1.csv")

bela_age = pd.read_csv('belanger_age_wrong.csv')

depth = dataframe['depth']
depth = list(depth)


age = []

i = 0
for profondeur in bela_age['depth']:
    
    if profondeur in depth and i<808:
        age.append(bela_age['best'][i])
    i = i+1
        
# age.append(9904)
# age.append(9974)
dataframe['age'] = age

dataframe.to_csv("taxa_sum_age.csv", header = True, index = False)


pollen_per = pd.read_csv("taxa_per.csv")


#%%
def plot_with_age(dataframe, y, title):
    plt.figure()
    plt.plot(dataframe['age'], dataframe[y])
    plt.title(title)
    plt.xlabel("age cal bp")
    plt.ylabel("valeur relative (%)")
    
#%%
def plot_one_page(dataframe, y, title, smoothing):
    
    plotte = full_page.add_subplot(4,3,i)
    
    #plotte = axe object
    #full_page = figure object
    
    if smoothing == True: 
        ysmooth = gaussian_filter1d(dataframe[y], sigma = 2)
        plotte.plot(dataframe['age'], ysmooth)
        plotte.set_xticks(range(0, 12000, 2000))
        plotte.set_xticklabels(["0","2","4","6","8","10"])


    else: 
        plotte.plot(dataframe['age'], dataframe[y])
    

    plotte.title.set_text(title)
    # plt.title(title)
    # plt.xlabel("age cal bp")
    # plt.ylabel("valeur relative (%)")

    

#%%

full_page = plt.figure()
i = 1

for taxa in classes_select:
    plot_one_page(pollen_per, taxa, taxa, True)
    plt.tight_layout()
    i = i+1

# plt.tight_layout()
plt.savefig('filename.png', dpi=300)

plt.show()

#####SUBPLOT METHOD: SEE OTHER PYTHON SCRIPT