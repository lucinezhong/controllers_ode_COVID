import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools as it
import Input
plt.rc('axes', axisbelow=True)

all_state_names = {'ak': 'Alaska', 'al': 'Alabama', 'ar': 'Arkansas',
                   'as': 'American Samoa',  'az': 'Arizona',
                   'ca': 'California', 'co': 'Colorado',
                   'ct': 'Connecticut', 'dc': 'District of Columbia',
                   'de': 'Delaware', 'fl': 'Florida', 'ga': 'Georgia',
                   'gu': 'Guam', 'hi': 'Hawaii', 'ia': 'Iowa',
                   'id': 'Idaho', 'il': 'Illinois', 'in': 'Indiana',
                   'ks': 'Kansas', 'ky': 'Kentucky', 'la': 'Louisiana',
                   'ma': 'Massachusetts', 'md': 'Maryland', 'me': 'Maine',
                   'mi': 'Michigan', 'mn': 'Minnesota', 'mo': 'Missouri',
                   'mp': 'Northern Mariana Islands', 'ms': 'Mississippi',
                   'mt': 'Montana', 'nc': 'North Carolina',
                   'nd': 'North Dakota', 'ne': 'Nebraska',
                   'nh': 'New Hampshire', 'nj': 'New Jersey',
                   'nm': 'New Mexico', 'nv': 'Nevada', 'ny': 'New York',
                   'oh': 'Ohio', 'ok': 'Oklahoma', 'or': 'Oregon',
                   'pa': 'Pennsylvania', 'pr': 'Puerto Rico',
                   'ri': 'Rhode Island', 'sc': 'South Carolina',
                   'sd': 'South Dakota', 'tn': 'Tennessee', 'tx': 'Texas',
                   'ut': 'Utah', 'va': 'Virginia',
                   'vi': 'Virgin Islands', 'vt': 'Vermont',
                   'wa': 'Washington', 'wi': 'Wisconsin',
                   'wv': 'West Virginia', 'wy': 'Wyoming'}

all_state_ids = {'01': 'al', '02': 'ak', '04': 'az', '05': 'ar',
                 '06': 'ca', '08': 'co', '09': 'ct', '10': 'de',
                 '11': 'dc', '12': 'fl', '13': 'ga', '15': 'hi',
                 '16': 'id', '17': 'il', '18': 'in', '19': 'ia',
                 '20': 'ks', '21': 'ky', '22': 'la', '23': 'me',
                 '24': 'md', '25': 'ma', '26': 'mi', '27': 'mn',
                 '28': 'ms', '29': 'mo', '30': 'mt', '31': 'ne',
                 '32': 'nv', '33': 'nh', '34': 'nj', '35': 'nm',
                 '36': 'ny', '37': 'nc', '38': 'nd', '39': 'oh',
                 '40': 'ok', '41': 'or', '42': 'pa', '44': 'ri',
                 '45': 'sc', '46': 'sd', '47': 'tn', '48': 'tx',
                 '49': 'ut', '50': 'vt', '51': 'va', '53': 'wa',
                 '54': 'wv', '55': 'wi', '56': 'wy', "60": 'as',
                 "66": 'gu', "72": 'pr', '78': 'vi', '69': 'mp'}

all_ids_state = {j:i for i,j in all_state_ids.items()}

state_posx_name = {'ak': (0, 0), 'me': (0, 10),'gu': (7, 0), 'vi': (7, 9), 'pr': (7, 8), 'mp': (7, 1),
              'vt': (1, 9), 'nh': (1, 10),'wa': (2, 0), 'id': (2, 1), 'mt': (2, 2), 'nd': (2, 3), 'mn': (2, 4),
              'il': (2, 5), 'wi': (2, 6), 'mi': (2, 7), 'ny': (2, 8), 'ri': (2, 9), 'ma': (2, 10),'or': (3, 0),
              'nv': (3, 1), 'wy': (3, 2), 'sd': (3, 3), 'ia': (3, 4), 'in': (3, 5), 'oh': (3, 6), 'pa': (3, 7),
              'nj': (3, 8), 'ct': (3, 9), 'ca': (4, 0), 'ut': (4, 1), 'co': (4, 2), 'ne': (4, 3), 'mo': (4, 4),
              'ky': (4, 5), 'wv': (4, 6), 'va': (4, 7), 'md': (4, 8), 'de': (4, 9), 'az': (5, 1), 'nm': (5, 2),
              'ks': (5, 3), 'ar': (5, 4), 'tn': (5, 5), 'nc': (5, 6), 'sc': (5, 7), 'dc': (5, 8), 'ok': (6, 3),
              'la': (6, 4), 'ms': (6, 5), 'al': (6, 6), 'ga': (6, 7), 'hi': (6, 0), 'tx': (7, 3), 'fl': (7, 7)}

state_posx=dict()
for key,value in state_posx_name.items():
    state_posx[all_ids_state[key]]=value

month_dict = {'01':'January','02':'February','03':'March','04':'April','05':'May','06':'June',
              '07':'July','08':'August','09':'September','10':'October','11':'November','12':'December'}


w = 2.95
h = 2.25
ncols = 11
nrows = 8
tups = list(it.product(range(nrows), range(ncols)))

orignal_data,distance_NPI_rate=Input.input_pde()
pop = pd.read_csv('/Volumes/SeagateDrive/US-mobility/Population/co-population2010.csv', encoding='ISO-8859-1')
predict_data=np.loadtxt('analysis-files/predict_data_model_1.txt')
fr = open("analysis-files/shortest-distance.json", 'r+')
distance = eval(fr.read())  # 读取的str转换为字典
fr.close()
stateCounty=dict()
for index,row in pop.iterrows():
    if row['state'] in stateCounty.keys():
        stateCounty[row['state'] ].append(row['FIPS'])
    else:
        stateCounty[row['state']]=[row['FIPS']]
source=53061
dx=0.2
dt=0.1
fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*w,nrows*h), dpi=200, sharex=True)
plt.subplots_adjust(wspace=0.05,hspace=0.05)


for st, po in state_posx.items():
    stlab = all_state_names[all_state_ids[st]]
    if int(st) in stateCounty.keys():
        for counties in stateCounty[int(st)]:
            if str(counties) in distance[str(source)].keys():
                print(st, counties)
                d=distance[str(source)][str(counties)]
                print(d)
                xvals=list([ float(x) for x in orignal_data[str(d)].keys()])
                yvals=list([ float(y) for y in orignal_data[str(d)].values()])
                d=int(d/dx)
                xvals_pred=[int(x/dt) for x in xvals]
                yvals_pred=[predict_data[d][x] for x in xvals_pred]
                print(yvals)
                print(yvals_pred)
                ax[po].plot(xvals, yvals, color='#8dd3c7', linewidth=2.5,alpha=0.5)
                ax[po].plot(xvals, yvals_pred, color='#bebada', linewidth=2.5,alpha=0.5)
    ax[po].set_yscale('log')
    ax[po].set_xscale('log')
    #ylim_st = ax[po].get_ylim()
    #xlim_st = ax[po].get_xlim()
    #ax[po].set_ylim(ylim_st[0], ylim_st[1])
    #ax[po].set_xlim(-1, xvals[-1] + 2)
    ax[po].text(0.02, 0.98, stlab, fontsize='x-large', va='top', ha='left',
               color='grey', fontweight='bold', transform=ax[po].transAxes)

ax[(0,3)].text(0.1,0.99,'United States confirmed cases (real VS predict)',
               color='.3', fontsize=32, va='top', ha='left', transform=ax[(0,1)].transAxes)
for tup in tups:
    if tup not in state_posx.values():
        ax[tup].set_axis_off()
    else:
        ax[tup].set_xticks([])
        ax[tup].set_yticks([])
plt.savefig('analysis-results/state_subplot_casecounts_trends.png', dpi=425, bbox_inches='tight')
