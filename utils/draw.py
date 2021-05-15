'''
Author: your name
Date: 2021-05-12 05:51:37
LastEditTime: 2021-05-15 04:34:13
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /kg-mtl/utils/draw.py
'''
from matplotlib import pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size
plt.figure(figsize=(5,3))
neodti=[0.89, 0.95, 0.93]
neodti_std=[0.002, 0.002, 0.003]
mean=[0.92,0.96,0.95]
mean_kg=[0.91,0.96,0.95]
std = [0.022,0.013,0.019]
std_kg=[0.012,0.002,0.003]
env=['ACC','AUC','AUPR']
total_width,n=0.3,3
width=total_width/n
x=np.arange(3)

#rect=plt.bar(left=range(len(mean)),height=mean,align="center")

plt.ylabel(u'Average Metrics',fontsize=12)
x=x-(total_width)/n
# plt.xlabel(u'')
plt.bar(x, height=neodti, width=width, label='NeoDTI',color='palegreen')
plt.bar(x+width, height=mean, width=width, label='KG-MTL#',color='steelblue')
plt.bar(x+2*width,mean_kg,width=width, label='KG-MTL',color='orange')
plt.ylim((0.85,1))
plt.xticks(range(len(env)),env,FontSize=12)
plt.legend(fontsize=12)


k=8
for i in range(len(env)):

    z = [x[i]+width,x[i]+width]

    w = [mean[i]-std[i]/k,mean[i]+std[i]/k]
    z_ = [x[i]+2*width,x[i]+2*width]
    
    w_ = [mean_kg[i]-std_kg[i]/k,mean_kg[i]+std_kg[i]/k]
    z_neodti = [x[i],x[i]]
    
    w_neodti = [neodti[i]-neodti_std[i]/k,neodti[i]+neodti_std[i]/k]
    plt.plot(z_neodti,w_neodti, color='black')
    plt.plot(z,w,color='black')
    plt.plot(z_,w_,color='black')
plt.tight_layout()
plt.savefig('utils/fig.png')