import sys
sys.path.append('Project')
from input_library import *
import Input

path_external='/Volumes/SeagateDrive 1/US-mobility/'
def process(path_file):
    MSA_all = pandas.read_csv(path_external+'covid19-intervention-data/MSA_summary_indicators.csv')
    dict_temp = {}
    for msa in pd.unique(MSA_all['MSA_code']):
        Sd = MSA_all[MSA_all['MSA_code'] == int(msa)]['S(t)'].values[-1]
        Id = MSA_all[MSA_all['MSA_code'] == int(msa)]['I(t)'].values[-1]
        Rd = MSA_all[MSA_all['MSA_code'] == int(msa)]['R(t)'].values[-1]
        Dd = MSA_all[MSA_all['MSA_code'] == int(msa)]['D(t)'].values[-1]
        sum_pop = MSA_all[MSA_all['MSA_code'] == int(msa)]['S(t)'].values[0]
        dict_temp[msa] = [Sd, Id, Rd, Dd]

    frames=[]
    for days in [90,180,270,360]:
        df_temp=pd.read_csv(path_file+'best_case_linear-vaccination-'+str(days)+'.csv')
        I_temp_sum=[dict_temp[msa][1]+dict_temp[msa][2]+dict_temp[msa][3] for msa in df_temp['MSA_code']]
        D_temp = [dict_temp[msa][3] for msa in df_temp['MSA_code']]
        R_temp= [dict_temp[msa][2] for msa in df_temp['MSA_code']]
        df_temp['accumulated I'] = (df_temp['accumulated I']+df_temp['accumulated R'] + df_temp['accumulated D']) - I_temp_sum
        df_temp['accumulated R'] = df_temp['accumulated R'] - R_temp
        df_temp['accumulated D'] = df_temp['accumulated D']-D_temp
        df_temp['vaccination days']=[days for i in range(len(df_temp))]
        frames.append(df_temp)

    result_linear = pd.concat(frames)

    result_linear.to_csv(path_file+'best_case_sigmoid-linear-all.csv')

    frames = []
    for days in [120, 180, 270, 360]:
        df_temp = pd.read_csv(path_file+'best_case_sigmoid-vaccination-'+str(days)+'.csv')
        I_temp_sum = [ dict_temp[msa][1]+dict_temp[msa][2] + dict_temp[msa][3] for msa in df_temp['MSA_code']]
        D_temp = [dict_temp[msa][3] for msa in df_temp['MSA_code']]
        R_temp = [dict_temp[msa][2] for msa in df_temp['MSA_code']]
        df_temp['accumulated I'] = (df_temp['accumulated I']+ df_temp['accumulated R'] + df_temp['accumulated D']) - I_temp_sum
        df_temp['accumulated R'] = df_temp['accumulated R'] - R_temp
        df_temp['accumulated D'] = df_temp['accumulated D'] - D_temp
        df_temp['vaccination days'] = [days for i in range(len(df_temp))]
        frames.append(df_temp)
    result_sigmoid = pd.concat(frames)
    result_sigmoid.to_csv(path_file+'best_case_sigmoid-vaccination-all.csv')
    return dict_temp,result_linear,result_sigmoid

def boxplot_days(data,path_file,type):
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(14, 3.5))
    data = data[['vaccination days','vaccination','days']]
    data['days']=[i+1 for i in data['days']]
    print(data.columns)
    data_temp=data[data["vaccination days"]==120]
    sns.boxplot(ax=ax1,x="vaccination",y="days", data=data_temp, color="#9ecae1", showfliers = False,showmeans=True)
    data_temp = data[data["vaccination days"] == 180]
    sns.boxplot(ax=ax2, x="vaccination", y="days", data=data_temp, color="#9ecae1", showfliers=False, showmeans=True)
    data_temp = data[data["vaccination days"] == 270]
    sns.boxplot(ax=ax3, x="vaccination", y="days", data=data_temp,color="#9ecae1", showfliers=False, showmeans=True)
    data_temp = data[data["vaccination days"] == 360]
    sns.boxplot(ax=ax4, x="vaccination", y="days", data=data_temp, color="#9ecae1", showfliers=False, showmeans=True)
    #sns.boxplot(x="vaccination",y="days", hue="period",  data=df_long)
    #sns.despine(offset=10, trim=True)
    ax1.set_xlabel("")
    ax2.set_xlabel("")
    ax3.set_xlabel("")
    ax4.set_xlabel("")
    ax1.set_ylabel("Elimination day")
    ax2.set_ylabel("")
    ax3.set_ylabel("")
    ax4.set_ylabel("")
    for ax in [ax1,ax2,ax3,ax4]:
        ax.set_ylim(50,1001)
        ax.set_yscale('log')
    plt.tight_layout()
    fig.savefig(path_file+"boxplot_days_"+type+".png", dpi=600)

def boxplots_deaths(data,path_file,type):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14, 3.5))
    data = data[['vaccination days', 'vaccination', 'accumulated D']]
    print(data.columns)
    data_temp = data[data["vaccination days"] == 120]
    sns.boxplot(ax=ax1, x="vaccination", y="accumulated D", data=data_temp, color="#9ecae1", showfliers=False,
                showmeans=True)
    data_temp = data[data["vaccination days"] == 180]
    sns.boxplot(ax=ax2, x="vaccination", y="accumulated D", data=data_temp, color="#9ecae1", showfliers=False,
                showmeans=True)
    data_temp = data[data["vaccination days"] == 270]
    sns.boxplot(ax=ax3, x="vaccination", y="accumulated D", data=data_temp, color="#9ecae1", showfliers=False,
                showmeans=True)
    data_temp = data[data["vaccination days"] == 360]
    sns.boxplot(ax=ax4, x="vaccination", y="accumulated D", data=data_temp, color="#9ecae1", showfliers=False,
                showmeans=True)
    # sns.boxplot(x="vaccination",y="days", hue="period",  data=df_long)
    # sns.despine(offset=10, trim=True)
    ax1.set_xlabel("")
    ax2.set_xlabel("")
    ax3.set_xlabel("")
    ax4.set_xlabel("")
    ax1.set_ylabel("Additional death")
    ax2.set_ylabel("")
    ax3.set_ylabel("")
    ax4.set_ylabel("")
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_ylim(np.min(data['accumulated D'].values), np.max(data['accumulated D'].values))
        ax.set_yscale('log')
        ax.set_ylim(10,20000)
    #plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.tight_layout()
    fig.savefig(path_file+"boxplot_detah_"+type+".png", dpi=600)

def boxplots_infection(data,path_file,type):
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(14, 3.5))
    data = data[['vaccination days','vaccination','accumulated I']]
    print(data.columns)
    data_temp = data[data["vaccination days"] == 120]
    sns.boxplot(ax=ax1, x="vaccination", y="accumulated I", data=data_temp, color="#9ecae1", showfliers=False, showmeans=True)
    data_temp = data[data["vaccination days"] == 180]
    sns.boxplot(ax=ax2, x="vaccination", y="accumulated I", data=data_temp, color="#9ecae1", showfliers=False, showmeans=True)
    data_temp = data[data["vaccination days"] == 270]
    sns.boxplot(ax=ax3, x="vaccination", y="accumulated I", data=data_temp, color="#9ecae1", showfliers=False, showmeans=True)
    data_temp = data[data["vaccination days"] == 360]
    sns.boxplot(ax=ax4, x="vaccination", y="accumulated I", data=data_temp, color="#9ecae1", showfliers=False, showmeans=True)
    # sns.boxplot(x="vaccination",y="days", hue="period",  data=df_long)
    # sns.despine(offset=10, trim=True)
    ax1.set_xlabel("")
    ax2.set_xlabel("")
    ax3.set_xlabel("")
    ax4.set_xlabel("")
    ax1.set_ylabel("Additional infection")
    ax2.set_ylabel("")
    ax3.set_ylabel("")
    ax4.set_ylabel("")
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_ylim(np.min(data['accumulated I'].values), np.max(data['accumulated I'].values))
        ax.set_yscale('log')
        ax.set_ylim(200, 1000000)
    plt.tight_layout()
    fig.savefig(path_file+"boxplot_infection_"+type+".png", dpi=600)

def newly_infection(result_linear,path_file,type):
    fig, ((ax1,ax2,ax3), (ax4,ax5,ax6),(ax7,ax8,ax9))= plt.subplots(3, 3, figsize=(12, 10))
    for vacciantion, ax in zip([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]):
        print(vacciantion)
        ax.text(0.3, 0.9, 'vaccination coverage='+str(vacciantion), transform=ax.transAxes)
        for msa in pd.unique(result_sigmoid['MSA_code']):
            df_infection=pd.read_csv(path_external+'temp/' + str(msa) +"best_case_sigmoid"+str(vacciantion)+".csv")
            ax.plot(df_infection['date'],np.asarray([i if i>0 else 0 for i in df_infection['newI'].values]),color='blue',linewidth=1,alpha=0.5)
        #ax.set_yscale('log')
        ax.set_xlim(0,500)
        ax.set_xlabel("days since Jan-12-2020")

    ax1.set_ylabel("New infected cases")
    ax4.set_ylabel("New infected cases")
    ax7.set_ylabel("New infected cases")
    plt.savefig(path_file+"newI_to_elimnations.png", dpi=600)
    plt.close()

if __name__ == '__main__':
    path_file = '/Users/luzhong/Documents/LuZHONGResearch/20200720COVID-Controllers/results_scenarios/'
    dict_temp, result_linear, result_sigmoid=process(path_file)

    '''
    print('linear boxplot')
    boxplot_days(result_linear,path_file,'linear')
    boxplots_deaths(result_linear,path_file,'linear')
    boxplots_infection(result_linear,path_file,'linear')
     '''
    #########sigmoid
    print('sigmoid boxplot')
    boxplot_days(result_sigmoid,path_file,'sigmoid')
    boxplots_deaths(result_sigmoid,path_file,'sigmoid')
    boxplots_infection(result_sigmoid,path_file,'sigmoid')

    newly_infection(result_sigmoid,path_file,'sigmoid')
