import sys
sys.path.append('/Users/luzhong/Documents/pythonCode/PDE-COVID')
from input_library import *
from Input import *

plot_start_date = datetime.datetime.strptime('4/01/20', '%m/%d/%y')
def moving_average(list_example, n) :
    new_list=[]
    for i in range(len(list_example)):
        if i<=n:
            new_list.append(np.mean([list_example[j] for j in range(0,i+1) if list_example[j]>0]))
        else:
            new_list.append(np.mean([list_example[j] for j in range(i-n, i + 1) if list_example[j]>0]))

    return new_list

def MSA_R_eff(MSA_all,type_file_ratio):
    interval=30
    datelist = []
    timelist=[]
    output_list=[]
    for date in np.unique(MSA_all['time']):
        d = datetime.datetime.strptime(str(date), '%Y-%m-%d')
        if d>=plot_start_date:
            output_list.append(date)
            timelist.append(date)
            d = d.strftime('%b-%d,%Y')
            datelist.append(d[0:6])
    MSA_all=MSA_all[MSA_all['time'].isin(output_list)]
    for y_select in ['beta(t)','gamma(t)','delta(t)','R_eff(t)']:
        MSA_all[y_select]=moving_average(MSA_all[y_select].values,7)
    color_dict={'beta(t)': '#d7191c','gamma(t)':'#e66101','delta(t)':'#0571b0','R_eff(t)':'#bc80bd'}
    for y_select,namestr in zip(['beta(t)','gamma(t)','delta(t)','R_eff(t)'],['beta(t)','delta(t)','gamma(t)','R_eff(t)']):
        data_df = MSA_all[[y_select,'time','MSA_code']]
        data_df=data_df[data_df['time'].isin(timelist)]
        fig, ax = plt.subplots(figsize=(6, 4))
        for msa in pd.unique(MSA_all['MSA_code']):
            data_df_temp = data_df[data_df['MSA_code'] == msa]
            if msa in [35620,26420]:
                alpha_value=0.5
                size=5
                ax.plot([i for i in range(len(data_df_temp["time"]))], data_df_temp[y_select], linewidth=1,
                           color=color_dict[y_select], alpha=alpha_value)

            else:
                alpha_value = 0.1
                size = 1
            ax.scatter([i for i in range(len(data_df_temp["time"]))],data_df_temp[y_select],s=size,facecolor=color_dict[y_select],linewidths=0,alpha=alpha_value)
        if y_select=='R_eff(t)':
            ax.axhline(y=1,linewidth=1, color='black')
        #sns.stripplot(x="time", y=y_select, jitter=True, split=True, linewidth=0.1, alpha=0.1, data=data_df,size=3,palette="Blues")

        #sns.boxplot(x="time", y=y_select, data=data_df, showfliers=False, showmeans=False,palette="Blues")
        print((np.arange(0, len(datelist), interval)))
        print([datelist[i * interval] for i in range(0, len(np.arange(0, len(datelist), interval)))],)
        ax.set_xticks(np.arange(0, len(datelist), interval))
        ax.set_xticklabels([datelist[i * interval] for i in range(0, len(np.arange(0, len(datelist), interval)))],
            fontsize=6, rotation=45)

        if y_select=='beta(t)':
            plt.ylim(0.01,1.1)
            plt.ylabel('Infection rate '+r'$\beta_0+U_{\beta}(t)$')
        if y_select=='gamma(t)':
            plt.ylim(0.005,1)
            plt.ylabel('Recovery rate '+r'$\gamma_0+U_{\gamma}(t)$')
        if y_select=='delta(t)':
            plt.ylim(0.00001,1)
            plt.ylabel('Death rate '+r'$\delta_0+U_{\delta}(t)$')
        if y_select=='R_eff(t)':
            plt.ylim(0.8,12)
            plt.ylabel(r'$R_{eff}(t)$')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig("results_trajectory_fitting/reported/MSA_all_"+y_select+"_"+type_file_ratio+".png", dpi=600)
        plt.close()
    MSA_all = MSA_all[MSA_all['time'].isin(timelist)]

    fig, ax = plt.subplots(figsize=(6, 4))
    MSA_all['ratio_infection_recovery']=[ float(i/j) if float(i/j)< 10 else 10 for i,j in zip(MSA_all['beta(t)'].values,MSA_all['gamma(t)'].values)]
    mean_list=[]
    std_list=[]
    for datex in pd.unique(MSA_all['time']):
        mean_list.append(np.mean(MSA_all[MSA_all["time"]==datex]["ratio_infection_recovery"]))
        std_list.append(np.std(MSA_all[MSA_all["time"] == datex]["ratio_infection_recovery"]))
        print('recovery',datex,mean_list[-1],std_list[-1])

    ax.plot([i for i in range(len(mean_list))], mean_list)
    all_mean=np.mean(mean_list)
    ax.text(0.8, 0.5, str(round(all_mean,3)), transform=ax.transAxes)
    ax.set_xticks(np.arange(0, len(datelist), interval))
    ax.set_xticklabels([datelist[i * interval] for i in range(0, len(np.arange(0, len(datelist), interval)))],
                       fontsize=6, rotation=45)

    ax.fill_between([i for i in range(len(mean_list))],np.asarray(mean_list)-np.asarray(std_list),np.asarray(mean_list)+np.asarray(std_list),alpha=0.5)
    ax.set_ylim(0,6)
    plt.ylabel(r'$(\beta_0+U_{\beta}(t))/(\gamma_0+U_{\gamma}(t))$')
    plt.savefig("results_trajectory_fitting/reported/MSA_all_ratio_infection_recovery.png", dpi=600)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    MSA_all['ratio_infection_death'] = [float(i / j) if float(i / j) < 200 else 200 for i, j in
                                           zip(MSA_all['beta(t)'].values, MSA_all['delta(t)'].values)]
    mean_list = []
    std_list = []
    for datex in pd.unique(MSA_all['time']):
        mean_list.append(np.mean(MSA_all[MSA_all["time"] == datex]["ratio_infection_death"]))
        std_list.append(np.std(MSA_all[MSA_all["time"] == datex]["ratio_infection_death"]))
        print('death',datex, mean_list[-1], std_list[-1])

    ax.plot([i for i in range(len(mean_list))], mean_list)
    all_mean = np.mean(mean_list)
    #ax.text(0.8, 0.7, str(round(all_mean, 3)), transform=ax.transAxes)
    ax.set_xticks(np.arange(0, len(datelist), interval))
    ax.set_xticklabels([datelist[i * interval] for i in range(0, len(np.arange(0, len(datelist), interval)))],
                       fontsize=6, rotation=45)

    ax.fill_between([i for i in range(len(mean_list))], np.asarray(mean_list) - np.asarray(std_list),
                    np.asarray(mean_list) + np.asarray(std_list), alpha=0.5)
    #ax.set_ylim(0, 200)
    plt.ylabel(r'$(\beta_0+U_{\beta}(t))/(\delta+U_{\delta}(t))$')
    plt.savefig("results_trajectory_fitting/reported/MSA_all_ratio_infection_death.png", dpi=600)
    plt.close()


def under_reproting_check(df_reported,df_sero):
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(12, 3.5))
    output_list = []
    for date in np.unique(df_reported['time']):
        d = datetime.datetime.strptime(str(date), '%Y-%m-%d')
        if d >= datetime.datetime.strptime('2021-02-14', '%Y-%m-%d') and d <= datetime.datetime.strptime('2021-02-20',                                                                                          '%Y-%m-%d'):
            output_list.append(date)

    df_reported =df_reported[df_reported['time'].isin(output_list)]
    df_sero = df_sero[df_sero['time'].isin(output_list)]
    df_reported[df_reported['R_eff(t)'] < 0] = 0
    df_sero[df_sero['R_eff(t)']<0]=0
    for msa in pd.unique(df_reported['MSA_code']):
        df_reported_temp = df_reported[df_reported['MSA_code'] == msa]
        df_sero_temp = df_sero[df_sero['MSA_code'] == msa]
        if len(df_reported_temp)==len(df_sero_temp):
            ax1.scatter(df_reported_temp['beta(t)'], df_sero_temp['beta(t)'], color='#377eb8', alpha=0.2,s=10)
            ax2.scatter(df_reported_temp['gamma(t)'], df_sero_temp['gamma(t)'], color='#377eb8', alpha=0.2, s=10)
            ax3.scatter(df_reported_temp['delta(t)'], df_sero_temp['delta(t)'], color='#377eb8', alpha=0.2, s=10)
    ax1.set_ylabel(r'$U_{\beta}$'+' From seroprevalence data')
    ax1.set_xlabel(r'$U_{\beta}$'+' From reported data')
    ax2.set_ylabel(r'$U_{\gamma}$' + ' From seroprevalence data')
    ax2.set_xlabel(r'$U_{\gamma}$' + ' From reported data')
    ax3.set_ylabel(r'$U_{\delta}$' + ' From seroprevalence data')
    ax3.set_xlabel(r'$U_{\delta}$' + ' From reported data')

    ax1.plot([i/100 for i in range(0,100)],[i/100 for i in range(0,100)],linewidth=0.5,color='grey')
    ax1.set_xlim(0,0.1)
    ax1.set_ylim(0,0.1)
    ax2.plot([i / 100  for i in range(0, 100)], [i / 100 for i in range(0, 100)],linewidth=0.5,color='grey')
    ax2.set_xlim(0, 0.02)
    ax2.set_ylim(0, 0.02)
    ax3.plot([i / 100 for i in range(0, 100)], [i / 100 for i in range(0, 100)],linewidth=0.5,color='grey')
    ax3.set_xlim(0, 0.0015)
    ax3.set_ylim(0, 0.0015)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.tight_layout()
    plt.savefig('results_trajectory_fitting/Sero/underreproting_checking_R_eff.png', dpi=600)
    plt.close()

def MSA_compariosn_fitting(MSA_true,MSA_fitted,Msa_statistics,region_dict_type):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    region_dict_type = {'Northeast': 'X', "Midwest": '>', 'South': 'D', 'West': "o"}
    legend_elements = [Line2D([0], [0], marker='X', linewidth=0,markeredgecolor='#d7191c', label='Northeast',
                              markerfacecolor='none', markersize=5),
                       Line2D([0], [0], marker='>', linewidth=0,markeredgecolor='#d7191c', label='Midwest',
                              markerfacecolor='none', markersize=5),

                       Line2D([0], [0], marker='D', linewidth=0,markeredgecolor='#d7191c', label='South',
                              markerfacecolor='none', markersize=5),
                       Line2D([0], [0], marker='o', linewidth=0,markeredgecolor='#d7191c', label='West',
                              markerfacecolor='none', markersize=5),
                       ]


    fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(9, 3))
    print(pd.unique(MSA_fitted['MSA_code']))
    timelist=pd.unique(MSA_fitted['time'])
    maslist = pd.unique(MSA_fitted['MSA_code'])
    #timelist=['2020-12-01']
    MSA_true=MSA_true[MSA_true['time'].isin(timelist)]
    MSA_true = MSA_true[MSA_true['MSA_code'].isin(maslist)]
    MSA_fitted = MSA_fitted[MSA_fitted['time'].isin(timelist)]
    MSA_fitted = MSA_fitted[MSA_fitted['MSA_code'].isin(maslist)]
    color_dict = {'beta(t)': '#d7191c', 'gamma(t)': '#e66101', 'delta(t)': '#0571b0', 'R_eff(t)': '#5e3c99'}
    print(MSA_true)
    print(MSA_fitted)
    for msa in pd.unique(MSA_true['MSA_code']):
        if msa not in [16220]:
            print(msa,Msa_statistics['region'][msa][0])
            print(MSA_true[MSA_true['MSA_code']==msa]['Id'].values[-1],MSA_fitted[MSA_fitted['MSA_code']==msa]['I'].values[-1])
            region_temp=Msa_statistics['region'][msa][0]
            ax1.scatter(MSA_true[MSA_true['MSA_code']==msa]['Id'],MSA_fitted[MSA_fitted['MSA_code']==msa]['I'],s=10,edgecolors= '#d7191c',linewidths=0.1,facecolors='none',alpha=0.5,marker=region_dict_type[region_temp])
            ax2.scatter(MSA_true[MSA_true['MSA_code']==msa]['Rd'], MSA_fitted[MSA_fitted['MSA_code']==msa]['R'],s=10,edgecolors= '#e66101',linewidths=0.1,facecolors='none',alpha=0.5,marker=region_dict_type[region_temp])
            ax3.scatter(MSA_true[MSA_true['MSA_code']==msa]['Dd'], MSA_fitted[MSA_fitted['MSA_code']==msa]['D'],s=10,edgecolors= '#0571b0',linewidths=0.1,facecolors='none',alpha=0.5,marker=region_dict_type[region_temp])
    '''
    ax1.plot([i for i in range(int(MSA_true['Id'].max()))],[i for i in range(int(MSA_true['Id'].max()))],linewidth=0.5,color='grey')
    ax1.plot([i for i in range(int(MSA_true['Rd'].max()))], [i for i in range(int(MSA_true['Rd'].max()))], linewidth=0.5,
             color='grey')
    ax1.plot([i for i in range(int(MSA_true['Dd'].max()))], [i for i in range(int(MSA_true['Dd'].max()))], linewidth=0.5,
             color='grey')
    '''

    ax1.legend(handles=legend_elements, loc=2)
    ax1.set_ylabel('Fitted infected cases')
    ax1.set_xlabel('Real infected cases')
    ax2.set_xlabel('Real recovered cases')
    ax3.set_xlabel('Real dead cases')
    ax1.set_xlim(1,1000000)
    ax2.set_xlim(1, 1000000)
    ax3.set_xlim(1, 1000000)
    ax1.set_ylim(1, 1000000)
    ax2.set_ylim(1, 1000000)
    ax3.set_ylim(1, 1000000)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    plt.tight_layout()
    plt.savefig('results_trajectory_fitting/reported/MSA_compariosn_fitting.png', dpi=600)
    plt.close()


if __name__ == '__main__':
    case ="R_eff_reported"
    case="R_eff_sero"
    Msa_statistics=input_MSA_statistics()
    county_statistics=input_Ctry_statistics()
    regions=pd.read_excel(path_external+'MSAs/MSA_info.xlsx',sheet_name='State-regions')
    regions_dict=dict(zip(regions['code'],regions['Region']))
    Msa_statistics['region']=dict()
    for msa,ctry_list in Msa_statistics['county'].items():
        region_list=[]
        for ctry in ctry_list:
            if ctry in county_statistics['state'].keys():
                state_temp=county_statistics['state'][ctry]
                region_list.append(regions_dict[state_temp])
        Msa_statistics['region'][msa]=pd.unique(region_list)
    region_dict_type={'Northeast':'X',"Midwest":'>','South':'D','West':"o"}

    if case=='R_eff_reported':
        ########reproted cases
        MSA_all=pd.read_csv('results_trajectory_fitting/reported/MSAs-ODE-tracking-parameters.csv')
        MSA_R_eff(MSA_all,'')
        MSA_true = pd.read_csv('dataset/MSA_S_I_R_D.csv')
        MSA_fitted = pd.read_csv('results_trajectory_fitting/reported/MSAs-ODE-tracking-two.csv')
        #MSA_compariosn_fitting(MSA_true,MSA_fitted,Msa_statistics,region_dict_type)

    if case == 'R_eff_sero':
        ########Seroprevalence cases
        MSA_all_reported=pd.read_csv('results_trajectory_fitting/reported/MSAs-ODE-tracking-parameters.csv')
        MSA_all_sero = pd.read_csv('results_trajectory_fitting/Sero/MSAs-ODE-tracking-parameters.csv')
        under_reproting_check(MSA_all_reported,MSA_all_sero)
