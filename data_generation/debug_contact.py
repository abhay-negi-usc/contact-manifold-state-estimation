import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv("/home/rp/abhay_ws/contact-manifold-state-estimation/data_generation/pose_sweep_contacts.csv")
# delete 1st row 
df = df.drop(0)
# convert columns to numeric
df['z'] = pd.to_numeric(df['z'])
df['metric'] = pd.to_numeric(df['metric'])
df['contact'] = pd.to_numeric(df['contact'])

for idx, row in df.iterrows():
    if df['contact'][idx] == 1:
        df['metric'][idx] = -df['metric'][idx] # separation positive value, penetration negative value 


plt.figure(figsize=(10,6))
plt.plot(df['z'], df['metric']) 
plt.grid(True)
plt.show() 