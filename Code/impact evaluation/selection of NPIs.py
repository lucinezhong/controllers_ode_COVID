import sys
sys.path.append('/Users/luzhong/Documents/pythonCode/PDE-COVID')
from input_library import *
import Input

emergency_date=datetime.datetime.strptime('2020-03-13', '%Y-%m-%d')
start_date=datetime.datetime.strptime('2020-04-01', '%Y-%m-%d')
end_date = datetime.datetime.strptime('2021-02-20', '%Y-%m-%d')
training_date=datetime.datetime.strptime('2020-10-01', '%Y-%m-%d')
#training_date= datetime.datetime.strptime('2021-02-20', '%Y-%m-%d')


def output_scatter(Data_all,namestr_list,namestr_color_dict,namestr,path_results):
    from matplotlib.lines import Line2D
    datelist = []
    for date in pd.unique(Data_all['date']):
        d=datetime.datetime.strptime(date, '%Y-%m-%d')
        d =d.strftime('%b-%d,%Y')
        datelist.append(d[0:6])


    label_list=['Stay-at-home','School-closure','Quarantined','Work-from-home','Face-mask','Testing','Washing-hand','Avoid-crowd']
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    count=0
    for col,output_str in zip(namestr_list,namestr_list):
        Data_all_temp = Data_all.groupby(['date']).agg({col: ['mean', 'std']}).reset_index()
        if count<4:
            linestyle='-'
        else:
            linestyle=':'
        ax.plot([i for i in range(len(Data_all_temp[col]['mean']))],Data_all_temp[col]['mean'], linestyle=linestyle,c=namestr_color_dict[col],label=label_list[count])
        interval_0=[i-j for i,j in zip(Data_all_temp[col]['mean'],Data_all_temp[col]['std'])]
        interval_1 = [i+j for i, j in zip(Data_all_temp[col]['mean'], Data_all_temp[col]['std'])]
        #print(output_str,'std',np.mean(Data_all_temp[col]['std']))
        #print(Data_all_temp[col]['mean'])
        #ax.fill_between([i for i in range(len(interval_0))], interval_0, interval_1, color=namestr_color_dict[col], alpha=.1)
        count+=1
    ax.set_ylim(0,1)
    ax.set_ylabel(output_str, fontsize=7)
    ax.set_xticks(np.arange(0, len(datelist), 30))
    ax.set_xticklabels([datelist[i * 30] for i in range(0, len(np.arange(0, len(datelist), 30)))], fontsize=8,
                       rotation=45)
    ax.legend( loc=1,fontsize=10)
    #ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(path_results+namestr+"_intervention_data.png",
                dpi=600)
    plt.close()

def combination_results(MSA_all,namestr_list):
    Result_df = pd.DataFrame(columns=['number of NPIs','accuracy_beta','accuracy_gamma','accuracy_delta','NPIs'])
    count=0
    MSA_all['date'] = [datetime.datetime.strptime(str(datex), '%Y-%m-%d') for datex in MSA_all['date']]
    MSA_all = MSA_all.replace([np.inf, -np.inf, np.nan], 0)
    delay_d=1
    index_x = [start_date + datetime.timedelta(days=i) for i in range((training_date - start_date).days) if
               i < (training_date - start_date).days - delay_d]
    index_y = [start_date + datetime.timedelta(days=i) for i in range((training_date - start_date).days) if
               i > delay_d - 1]

    index_x_all=[start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days) if
               i < (end_date - start_date).days - delay_d]
    index_y_all=[start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days) if
               i > delay_d - 1]
    print(len(index_x),len(index_y_all))
    for i in range(1,len(namestr_list)):
        comb_results=list(itertools.combinations(namestr_list, i))
        for temp_list in comb_results:

            MSA_temp=copy.deepcopy(MSA_all)
            temp_listx=list(temp_list)
            for NPI_str in [x for x in namestr_list if x not in temp_listx]:
                MSA_temp[NPI_str]=[0 for i in range(len(MSA_temp))]

            accuarcy_beta=0;accuarcy_gamma=0;accuarcy_delta=0
            N=len(np.unique(MSA_temp['MSA_code']))
            for msa in np.unique(MSA_temp['MSA_code']):
                MSA_each = MSA_temp[MSA_temp['MSA_code'] == msa]
                for y_select, y_select_redu in zip(['beta(t)', 'gamma(t)', 'delta(t)'], ['beta_0', 'gamma_0', 'delta_0']):
                    X = MSA_each[MSA_each['date'].isin(index_x_all)][namestr_list]
                    Y = list(MSA_each[MSA_each['date'].isin(index_y_all)][y_select])
                    Y_temp = list(MSA_each[MSA_each['date'].isin(index_y_all)][y_select_redu])
                    Y = [i - j for i, j in zip(Y, Y_temp)]
                    if y_select == 'beta(t)':
                        popt, pcov = curve_fit(func_beta, X.values, Y, maxfev=100000)
                        predict_value = func_beta(MSA_each[namestr_list].values, *(popt))
                        ture_value=[i - Y_temp[0] for i in MSA_each[y_select].values]
                        accuarcy_beta+=np.mean([abs(x-y) for x,y in zip(predict_value,ture_value)][len(index_y):len(index_y_all)])
                    if y_select == 'gamma(t)':
                        popt, pcov = curve_fit(func_gamma, X.values, Y, maxfev=100000)
                        predict_value = list(func_gamma(MSA_each[namestr_list].values, *(popt)))
                        ture_value=[i - Y_temp[0] for i in MSA_each[y_select].values]
                        accuarcy_gamma+= np.mean([abs(x - y) for x, y in zip(predict_value, ture_value)][len(index_y):len(index_y_all)])

                    if y_select == 'delta(t)':
                        popt, pcov = curve_fit(func_delta, X.values, Y, maxfev=100000)
                        predict_value = list(func_delta(MSA_each[namestr_list].values, *(popt)))
                        ture_value = [i - Y_temp[0] for i in MSA_each[y_select].values]
                        accuarcy_delta+= np.mean([abs(x - y) for x, y in zip(predict_value, ture_value)][len(index_y):len(index_y_all)])
            print(temp_listx,accuarcy_beta)
            Result_df.loc[count]=[len(temp_listx),accuarcy_beta,accuarcy_gamma,accuarcy_delta,temp_listx]
            count+=1

    Result_df.to_csv(path_files + 'selection_of_NPIs.csv')


def func_beta(x, a1, a2, a3,a4,b1,b2,b3,b4):
    return (1-a1 * x[:, 0]-a2 * x[:, 1]-a3 * x[:, 2]-a4 * x[:, 3] ) *(1-b1 * x[:, 4]-b2* x[:, 5]-b3 * x[:, 6]-b4* x[:, 7] )-1

def func_delta(x, a1, a2, a3,a4,b1,b2,b3,b4):
    return (1-a1 * x[:, 0]-a2 * x[:, 1]-a3 * x[:, 2]-a4 * x[:, 3] ) *(1-b1 * x[:, 4]-b2* x[:, 5]-b3 * x[:, 6]-b4* x[:, 7] )-1

def func_gamma(x, a1, a2, a3,a4,b1,b2,b3,b4):
    return (1-a1 * x[:, 0]-a2 * x[:, 1]-a3 * x[:, 2]-a4 * x[:, 3] ) *(1-b1 * x[:, 4]-b2* x[:, 5]-b3 * x[:, 6]-b4* x[:, 7] )-1


def combination_plot(path_files):
    Result_df=pd.read_csv(path_files + 'selection_of_NPIs.csv')
    plt.figure(figsize=(5, 4))
    g = sns.boxplot(x="number of NPIs", y="accuracy_beta", data=Result_df,showfliers = False,boxprops=dict(alpha=.3))
    sns.stripplot(x="number of NPIs", y="accuracy_beta", data=Result_df,jitter=0.2,size=3)
    g.set(yscale="log")
    plt.xlabel("Number of NPIs")
    plt.ylabel("Mean absolute error for "+r"$U_{\beta}$")
    plt.savefig(path_files + "NPIs_vs_accuracy_beta.png", dpi=2000)

    plt.figure(figsize=(5, 4))
    g = sns.boxplot(x="number of NPIs", y="accuracy_gamma", data=Result_df, showfliers=False, boxprops=dict(alpha=.3))
    sns.stripplot(x="number of NPIs", y="accuracy_gamma", data=Result_df, jitter=0.2, size=3)
    g.set(yscale="log")
    plt.xlabel("Number of NPIs")
    plt.ylabel("Mean absolute error for " + r"$U_{\gamma}$")
    plt.savefig(path_files + "NPIs_vs_accuracy_gamma.png", dpi=2000)

    plt.figure(figsize=(5, 4))
    g = sns.boxplot(x="number of NPIs", y="accuracy_delta", data=Result_df, showfliers=False, boxprops=dict(alpha=.3))
    sns.stripplot(x="number of NPIs", y="accuracy_delta", data=Result_df, jitter=0.2, size=3)
    g.set(yscale="log")
    plt.xlabel("Number of NPIs")
    plt.ylabel("Mean absolute error for " + r"$U_{\delta}$")
    plt.savefig(path_files + "NPIs_vs_accuracy_delta.png", dpi=2000)


if __name__ == '__main__':
    MSA_statistics = Input.input_MSA_statistics()
    Ctry_statistics = Input.input_Ctry_statistics()
    State_statistics = Input.input_State_statistics()

    path_files = 'results_interventions/selection/'
    MSA_all = pandas.read_csv(path_external + 'covid19-intervention-data/MSA_summary_indicatorsreported.csv')
    namestr_list_output = ['ratio of excessive time at home','ratio of people support school closure',
                           'ratio of people willing to be quarantined','ratio of popeple willing work from home',
                           'ratio of people wearing face masks','ratio of people taking testing',
                           'rati of people washing hand frequently', 'ratio of people avoiding crowd']

    namestr_color_dict = dict(zip(namestr_list_output, ['#355F8C', '#CB7132', '#447E36','#66c2a5','#fc8d62','#8da0cb','#e78ac3','#e5c494']))

    #output_scatter(MSA_all, namestr_list_output, namestr_color_dict, 'MSA', path_files)

    #combination_results#(MSA_all, namestr_list_output)
    ####permutation
    ####learned coefficient results erors
    combination_plot(path_files)