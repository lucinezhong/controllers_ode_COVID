import sys
sys.path.append('/Users/luzhong/Documents/pythonCode/PDE-COVID')
from input_library import *
import Input
from synth import *

global emergency_date
global start_date
global end_date
global training_date
emergency_date=datetime.datetime.strptime('2020-03-13', '%Y-%m-%d')
start_date=datetime.datetime.strptime('2020-04-01', '%Y-%m-%d')
end_date = datetime.datetime.strptime('2021-02-20', '%Y-%m-%d')
#training_date=datetime.datetime.strptime('2020-10-01', '%Y-%m-%d')
training_date= datetime.datetime.strptime('2021-02-20', '%Y-%m-%d')

###for seroprevlanece
#start_date=datetime.datetime.strptime('2020-04-12', '%Y-%m-%d')


def SIRD(t, y):
    S = y[0]
    I = y[1]
    R = y[2]
    D=  y[3]
    # print(y[0],y[1],y[2])
    u= y[4]
    beta = y[5]
    gamma = y[6]
    delta = y[7]
    return ([-beta * S * I+u-u*S, beta * S * I - gamma * I-delta*I-u*I, gamma * I-u*R, delta*I,0, 0,0,0])



def func_beta(x, a, b, c):
    return (a * x[:, 0] + 1) * (b * x[:, 1] +c * x[:, 2]+1)-1
    #return a * x[:, 0] + b * x[:, 1] + c * x[:, 2]
    #return (a * x[:, 0]+b * x[:, 1]+1) * (a * x[:, 0]+b * x[:, 1]+1)-1+c * x[:, 2]

def func_delta(x, a,b, c):
    return (a * x[:, 0] + 1) * (b * x[:, 1] +c * x[:, 2]+1)-1
    #return a * x[:, 0] + b * x[:, 1] + c * x[:, 2]
    #return (a * x[:, 0]+b * x[:, 1]+1) * (a * x[:, 0]+b * x[:, 1]+1)-1+c * x[:, 2]



def func_gamma(x, a,b, c):
    return (a * x[:, 0] + 1) * (b * x[:, 1] +c * x[:, 2]+1)-1
    #return a * x[:, 0] + b * x[:, 1] + c * x[:, 2]
    #return (a * x[:, 0]+b * x[:, 1]+1) * (a * x[:, 0]+b * x[:, 1]+1)-1+c * x[:, 2]

def moving_average(list_example, n):

    new_list = []

    for i in range(len(list_example)):
        if i <= n:
            new_list.append(np.mean([list_example[j] for j in range(0, i + 1)]))
        else:
            new_list.append(np.mean([list_example[j] for j in range(i - n, i + 1)]))

    return new_list

    return list_example


def reset_time_format(df):
    time = [datetime.datetime.strptime(str(datex), '%m/%d/%Y') for datex in df['date']]
    df['date'] = time
    return df


def summary_data_MSA(MSA_statistics,path_files,result_file_str):
    moving_interval=7
    if result_file_str=='reported':
        infection_df = pd.read_csv('Dataset/MSA_S_I_R_D.csv')
        parameter_df = pd.read_csv('results_trajectory_fitting/reported/MSAs-ODE-tracking-parameters.csv')
    if result_file_str=='sero':
        infection_df = pd.read_csv('Dataset/MSA_S_I_R_D_SERO.csv')
        parameter_df = pd.read_csv('results_trajectory_fitting/Sero/MSAs-ODE-tracking-parameters.csv')

    timelist_all = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in pd.unique(parameter_df['time'])]
    timelist=[date for date in timelist_all if date>=start_date and date<=end_date]
    parameter_df['time'] = [datetime.datetime.strptime(str(datex)[0:10], '%Y-%m-%d') for datex in parameter_df['time']]
    infection_df['time'] = [datetime.datetime.strptime(str(datex)[0:10], '%Y-%m-%d') for datex in infection_df['time']]


    home_dwell_df = pd.read_csv(path_external+'covid19-intervention-data/MSA_home_dwell_time.csv')
    face_mask_df = pd.read_csv(path_external+'covid19-intervention-data/MSA_facemask.csv')
    testing_hopsital_df = pd.read_csv(path_external+'covid19-intervention-data/MSA_testing_hospitalisation.csv')
    work_df=pd.read_csv(path_external+'covid19-intervention-data/MSA_working_from_home.csv')
    avoid_crowd_df=pd.read_csv(path_external+'covid19-intervention-data/MSA_avoid_crowding.csv')
    quarantine_df=pd.read_csv(path_external+'covid19-intervention-data/MSA_quarantine.csv')
    wash_hand_df=pd.read_csv(path_external+'covid19-intervention-data/MSA_frequent_wash_hand.csv')
    school_df=pd.read_csv(path_external+'covid19-intervention-data/MSA_school_closure.csv')

    mean_facemask_others=defaultdict()
    for date_x in pd.unique(face_mask_df['date']):
        date_new=datetime.datetime.strptime(date_x, '%m/%d/%Y')
        if date_new not in mean_facemask_others.keys():
            mean_facemask_others[date_new]=dict()
        mean_facemask_others[date_new]['facemask']=np.mean(face_mask_df[face_mask_df['date']==date_x]['facemask'].values)
        mean_facemask_others[date_new]['working_from_home']=np.mean(work_df[work_df['date']==date_x]['working_from_home'].values)
        mean_facemask_others[date_new]['avoid_crowding']=np.mean(avoid_crowd_df[avoid_crowd_df['date']==date_x]['avoid_crowding'].values)
        mean_facemask_others[date_new]['quarantine']=np.mean(quarantine_df[quarantine_df['date']==date_x]['quarantine'].values)
        mean_facemask_others[date_new]['frequent_wash_hand']=np.mean(wash_hand_df[wash_hand_df['date']==date_x]['frequent_wash_hand'].values)
        mean_facemask_others[date_new]['school_closure']=np.mean(school_df[school_df['date']==date_x]['school_closure'].values)
    mean_home=dict()
    for date_x in pd.unique(home_dwell_df['date']):
        date_new=datetime.datetime.strptime(date_x, '%m/%d/%Y')
        mean_home[date_new]=np.mean(home_dwell_df[home_dwell_df['date']==date_x]['median_home_dwell_time'].values)

    namestr_list = ['testing', 'median_home_dwell_time','facemask','working_from_home','avoid_crowding','quarantine','frequent_wash_hand','school_closure']
    namestr_list_output = ['ratio of people taking testing', 'ratio of excessive time at home','ratio of people wearing face masks',
                           'ratio of popeple willing work from home', 'ratio of people avoiding crowd','ratio of people willing to be quarantined',
                           'rati of people washing hand frequently','ratio of people support school closure']
    namestr = dict(zip(namestr_list, namestr_list_output))



    MSA_all = pd.DataFrame(columns=[])
    for msa in np.unique(parameter_df['MSA_code']):
        # print(msa,MSA_statistics['name'][msa])
        MSA_each = pd.DataFrame(columns=[])
        ########## msa df======================================================

        infection_df_temp = infection_df[infection_df['MSA_code'] == msa]
        infection_df_temp = infection_df_temp[infection_df_temp['time'].isin(timelist)]
        parameter_df_temp=parameter_df[parameter_df['MSA_code'] == msa]
        parameter_df_temp=parameter_df_temp[parameter_df_temp['time'].isin(timelist)]
        testing_hopsital_df_temp = testing_hopsital_df[testing_hopsital_df['MSA_code'] == msa]
        home_dwell_df_temp = home_dwell_df[home_dwell_df['MSA_code'] == msa]
        face_mask_df_temp = face_mask_df[face_mask_df['MSA'] == msa]
        work_df_temp=work_df[work_df['MSA'] == msa]
        avoid_crowd_df_temp=avoid_crowd_df[avoid_crowd_df['MSA']==msa]
        quarantine_df_temp=quarantine_df[quarantine_df['MSA']==msa]
        wash_hand_df_temp=wash_hand_df[wash_hand_df['MSA']==msa]
        school_df_temp=school_df[school_df['MSA']==msa]

        I_temp_dict=dict(zip(infection_df_temp['time'],infection_df_temp['Id']))
        for df_temp_str, col_str in zip([testing_hopsital_df_temp,home_dwell_df_temp, face_mask_df_temp,work_df_temp,avoid_crowd_df_temp,quarantine_df_temp,wash_hand_df_temp,school_df_temp],namestr_list):
            #print(col_str)
            df_temp = copy.deepcopy(df_temp_str)
            df_temp = reset_time_format(df_temp)
            df_temp = df_temp.fillna(0)
            dict_temp = dict(zip(df_temp['date'], df_temp[col_str]))
            if col_str=='median_home_dwell_time':
                pre_intervention=np.mean(list(dict_temp.values())[0:30])
                temp_list=[dict_temp[t]-pre_intervention if t in dict_temp.keys() else 0 for t in timelist]
                temp_list=[i if i>0 else 0 for i in temp_list]
                temp_list=[i/0.2 if i<=0.2 else 1 for i in temp_list]
                if np.sum(temp_list) == 0:
                    pre_intervention = np.mean(list(mean_home.values())[0:30])
                    temp_list = [mean_home[t] - pre_intervention if t in mean_home.keys() else 0 for t in timelist]
                    temp_list = [i if i > 0 else 0 for i in temp_list]
                    temp_list = [i / 0.2 if i <= 0.2 else 1 for i in temp_list]
                    MSA_each[namestr[col_str]] = moving_average(temp_list, moving_interval)
                else:
                    MSA_each[namestr[col_str]] = moving_average(temp_list, moving_interval)

            if col_str=='testing':
                temp_list = [dict_temp[t]/(MSA_statistics['pop'][msa] + 1) if t in dict_temp.keys() else 0 for t in timelist]
                MSA_each[namestr[col_str]] = moving_average([ i if i<=1 else 1 for i in temp_list], moving_interval)
            if col_str in ['facemask','working_from_home','avoid_crowding','quarantine','frequent_wash_hand','school_closure']:
                temp_list = [dict_temp[t] if t in dict_temp.keys() else 0 for t in timelist]
                if np.sum(temp_list)==0:
                    temp_list = [mean_facemask_others[t][col_str] if t in mean_facemask_others.keys() else 0 for t in timelist]
                    MSA_each[namestr[col_str]] = moving_average(temp_list, moving_interval)
                else:
                    MSA_each[namestr[col_str]] = moving_average(temp_list,moving_interval)
            if np.sum(temp_list)==0:
                print(msa,col_str,np.sum(temp_list))
        if result_file_str=='reported':
            beta_0 = np.mean([i for i in parameter_df[parameter_df['MSA_code'] == 35620]['beta(t)'].values if i > 0 ][0:10])
            gamma_0 = np.mean([i for i in parameter_df[parameter_df['MSA_code'] == 35620]['gamma(t)'].values if i > 0 ][0:10])
            delta_0 = np.mean([i for i in parameter_df[parameter_df['MSA_code'] == 35620]['delta(t)'].values if i > 0 ][0:10])
        if result_file_str=='sero':
            beta_0=0.609036855;gamma_0=0.18617936;delta_0=0.009876217

        for y_select_redu, value_temp in zip(['beta_0', 'gamma_0', 'delta_0'], [beta_0, gamma_0, delta_0]):
            MSA_each[y_select_redu] = [value_temp for t in timelist]
        #print(msa,parameter_df_temp)
        MSA_each['beta(t)'] = moving_average(parameter_df_temp['beta(t)'].values,moving_interval)
        MSA_each['gamma(t)'] = moving_average(parameter_df_temp['gamma(t)'].values,moving_interval)
        MSA_each['delta(t)'] = moving_average(parameter_df_temp['delta(t)'].values,moving_interval)
        MSA_each['R_eff(t)'] = moving_average(parameter_df_temp['R_eff(t)'].values,moving_interval)

        ###################features------------------------------------------------
        MSA_each['MSA_code'] = [msa for i in range(len(timelist))]
        MSA_each['date'] = timelist
        MSA_each['I(t)'] = [i for i in infection_df_temp['Id']]
        MSA_each['R(t)'] = [i for i in infection_df_temp['Rd']]
        MSA_each['D(t)'] = [i for i in infection_df_temp['Dd']]
        MSA_each['S(t)'] = [i for i in infection_df_temp['Sd']]
        MSA_each.to_csv(path_external+'covid19-intervention-data/MSA_each_files/' + MSA_statistics['name'][
                msa] + '_summary_indicators'+result_file_str+'.csv')
        MSA_all = MSA_all.append(MSA_each)
    print(np.max(MSA_all['ratio of people taking testing']))
    MSA_all['ratio of people taking testing'] = MSA_all['ratio of people taking testing'] / np.max(MSA_all['ratio of people taking testing'])

    MSA_all.to_csv(path_external+'covid19-intervention-data/MSA_summary_indicators'+result_file_str+'.csv')

def output_scatter(Data_all,namestr_list,namestr_color_dict,namestr,path_results):
    from matplotlib.lines import Line2D
    datelist = []
    for date in pd.unique(Data_all['date']):
        d=datetime.datetime.strptime(date, '%Y-%m-%d')
        d =d.strftime('%b-%d,%Y')
        datelist.append(d[0:6])

    label_list = ['Stay-at-home', 'Face-mask', 'Testing']
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    count=0
    for col,output_str in zip(namestr_list,namestr_list):
        Data_all_temp = Data_all.groupby(['date']).agg({col: ['mean', 'std']}).reset_index()
        ax.plot([i for i in range(len(Data_all_temp[col]['mean']))],Data_all_temp[col]['mean'], c=namestr_color_dict[col],linestyle='-',label=label_list[count])
        interval_0=[i-j for i,j in zip(Data_all_temp[col]['mean'],Data_all_temp[col]['std'])]
        interval_1 = [i+j for i, j in zip(Data_all_temp[col]['mean'], Data_all_temp[col]['std'])]
        print(output_str,'std',np.mean(Data_all_temp[col]['std']))
        print(Data_all_temp[col]['mean'])
        ax.fill_between([i for i in range(len(interval_0))], interval_0, interval_1, color=namestr_color_dict[col], alpha=.1)
        count+=1
    ax.set_ylim(0,1)
    ax.set_ylabel(output_str, fontsize=10)
    ax.set_xticks(np.arange(0, len(datelist), 30))
    ax.set_xticklabels([datelist[i * 30] for i in range(0, len(np.arange(0, len(datelist), 30)))], fontsize=8,
                       rotation=45)
    ax.legend( loc=1)
    #ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(path_results+namestr+"_intervention_data.png",
                dpi=600)
    plt.close()

    for msa in [35620,26420,16980,41940]:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 5))
        count=0
        for col, ax, output_str in zip(namestr_list, [ax1, ax2, ax3], namestr_list):
            df_temp = Data_all[Data_all['MSA_code'] == msa]
            ax.plot([i for i in range(len(df_temp[col]))], df_temp[col], c=namestr_color_dict[col], linestyle='-',label=label_list[count])
            count+=1
        ax1.set_xticks(np.arange(0, len(datelist), 30))
        ax1.set_xticklabels(['' for i in range(0, len(np.arange(0, len(datelist), 30)))], fontsize=8,
                           rotation=45)
        ax2.set_xticks(np.arange(0, len(datelist), 30))
        ax2.set_xticklabels([''for i in range(0, len(np.arange(0, len(datelist), 30)))], fontsize=8,
                           rotation=45)
        ax3.set_xticks(np.arange(0, len(datelist), 30))
        ax3.set_xticklabels([datelist[i * 30] for i in range(0, len(np.arange(0, len(datelist), 30)))], fontsize=8,
                           rotation=45)
        ax1.set_ylim(0, 1.05)
        ax2.set_ylim(0, 1.05)
        ax3.set_ylim(0, 1.05)
        #ax1.legend(loc=2)
        #ax2.legend(loc=2)
        #ax3.legend(loc=2)
        plt.tight_layout()
        plt.savefig(path_results + namestr + "_intervention_data"+str(msa)+".png",
                    dpi=600)
        plt.close()


def impact_evaluation_MSA_DID(MSA_statistics,path_files,namestr_list,color_dict,delay_d,type_file_str):
    ###################difference in difference regression
    MSA_all = pandas.read_csv(path_external+'covid19-intervention-data/MSA_summary_indicators'+type_file_str+'.csv')
    MSA_all['date'] = [datetime.datetime.strptime(str(datex), '%Y-%m-%d') for datex in MSA_all['date']]

    MSA_all=MSA_all.replace([np.inf, -np.inf,np.nan], 0)
    ##############################for non intervention and intervention----------------------------
    index_x = [start_date + datetime.timedelta(days=i) for i in range((training_date - start_date).days) if
               i < (training_date - start_date).days - delay_d]
    index_y = [start_date + datetime.timedelta(days=i) for i in range((training_date- start_date).days) if
               i > delay_d - 1]
    print(len(index_x),len(index_y))
    ########non_intervention
    MSA_coefficient_beta = pd.DataFrame(columns=['MSA_code']+namestr_list)
    count_beta = 0
    MSA_coefficient_delta= pd.DataFrame(columns=['MSA_code']+namestr_list)
    count_delta = 0
    MSA_coefficient_gamma= pd.DataFrame(columns=['MSA_code'] + namestr_list)
    count_gamma = 0
    MSA_predict_beta = pd.DataFrame(columns=['type']+list(pd.unique(MSA_all['date'])))
    MSA_predict_delta = pd.DataFrame(columns=['type'] + list(pd.unique(MSA_all['date'])))
    MSA_predict_gamma= pd.DataFrame(columns=['type'] + list(pd.unique(MSA_all['date'])))
    predict_count=0
    predict_countx=0
    predict_countxx = 0
    for msa in np.unique(MSA_all['MSA_code']):
        MSA_each = MSA_all[MSA_all['MSA_code'] == msa]
        for y_select,y_select_redu in zip(['beta(t)', 'gamma(t)','delta(t)'],['beta_0','gamma_0', 'delta_0']):
            X = MSA_each[MSA_each['date'].isin(index_x)][namestr_list]
            Y = list(MSA_each[MSA_each['date'].isin(index_y)][y_select])
            Y_temp = list(MSA_each[MSA_each['date'].isin(index_y)][y_select_redu])
            Y=[i-j for i,j in zip(Y,Y_temp)]
            if y_select=='beta(t)':
                popt, pcov = curve_fit(func_beta, X.values, Y, maxfev=100000,bounds=([-5,-5,-2],[0,0,2]), p0=([-0.1,-0.1,0.1]))
                sigma_ab =  np.sqrt(np.diagonal(pcov))
                bound_upper = func_beta(MSA_each[namestr_list].values, *(popt + sigma_ab))
                bound_lower = func_beta(MSA_each[namestr_list].values, *(popt - sigma_ab))
                predict_value = func_beta(MSA_each[namestr_list].values, *(popt))
                MSA_predict_beta.loc[predict_count]=[str(msa)+'bound_upper']+list(bound_upper)
                predict_count+=1
                MSA_predict_beta.loc[predict_count] = [str(msa)+'bound_lower'] + list(bound_lower)
                predict_count += 1
                MSA_predict_beta.loc[predict_count] = [str(msa)+'predict_value'] + list(predict_value)
                predict_count += 1
                MSA_predict_beta.loc[predict_count] = [str(msa) + 'true_value'] + [i-Y_temp[0] for i in MSA_each[y_select].values]
                predict_count += 1
                coefficient_list = popt
                if np.sum(coefficient_list) != 0:
                    MSA_coefficient_beta.loc[count_beta] = [msa]+ list(coefficient_list)
                count_beta += 1
            if y_select=='gamma(t)':
                popt, pcov = curve_fit(func_gamma, X.values, Y, maxfev=100000,bounds=([-5,-5,-2],[0,0,2]), p0=([-0.1,-0.1,0.1]))
                sigma_ab =  np.sqrt(np.diagonal(pcov))
                bound_upper = func_gamma(MSA_each[namestr_list].values, *(popt + sigma_ab))
                bound_lower = func_gamma(MSA_each[namestr_list].values, *(popt - sigma_ab))
                predict_value = func_gamma(MSA_each[namestr_list].values, *(popt))
                MSA_predict_gamma.loc[predict_countxx]=[str(msa)+'bound_upper']+list(bound_upper)
                predict_countxx+=1
                MSA_predict_gamma.loc[predict_countxx] = [str(msa)+'bound_lower'] + list(bound_lower)
                predict_countxx += 1
                MSA_predict_gamma.loc[predict_countxx] = [str(msa)+'predict_value'] + list(predict_value)
                predict_countxx += 1
                MSA_predict_gamma.loc[predict_countxx] = [str(msa) + 'true_value'] + [i-Y_temp[0] for i in MSA_each[y_select].values]
                predict_countxx += 1
                coefficient_list = popt
                if np.sum(coefficient_list) != 0:
                    MSA_coefficient_gamma.loc[count_beta] = [msa]+ list(coefficient_list)
                count_gamma += 1
            if y_select=='delta(t)':
                popt, pcov = curve_fit(func_delta, X.values, Y, maxfev=100000,bounds=([-0.1,-0.1,-0.1],[-0.001,-0.001,0.1]),p0=([-0.1,-0.1,0.1]))

                sigma_ab = np.sqrt(np.diagonal(pcov))
                bound_upper = func_delta(MSA_each[namestr_list].values, *(popt + sigma_ab))
                bound_lower = func_delta(MSA_each[namestr_list].values, *(popt - sigma_ab))
                predict_value = func_delta(MSA_each[namestr_list].values, *(popt))
                MSA_predict_delta.loc[predict_countx]=[str(msa)+'bound_upper']+list(bound_upper)
                predict_countx+=1
                MSA_predict_delta.loc[predict_countx] = [str(msa)+'bound_lower'] + list(bound_lower)
                predict_countx += 1
                MSA_predict_delta.loc[predict_countx] = [str(msa)+'predict_value'] + list(predict_value)
                predict_countx += 1
                MSA_predict_delta.loc[predict_countx] = [str(msa) + 'true_value'] + [i-Y_temp[0] for i in MSA_each[y_select].values]
                predict_countx += 1

                coefficient_list = popt
                if np.sum(coefficient_list) != 0:
                    MSA_coefficient_delta.loc[count_delta] = [msa] + list(coefficient_list)
                count_delta += 1
            '''
            model = sm.OLS(Y, X.values).fit()
            ypred = model.predict(X)
            rmse_value = rmse(Y, ypred)
            
            coefficient_list = [list(model.params)[i] for i in range(len(list(model.pvalues)))]
            if np.sum(coefficient_list)!=0:
                MSA_coefficient.loc[count] = [msa, 0] + coefficient_list
            '''


            #print(msa,coefficient_list)
            '''
            if msa in [35620, 31080, 16980, 19100, 26420, 47900, 33100, 37980, 12060, 38060, 14460, 41860, 40140,19820,41940]:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot([i for i in range(len(Y))], [i+Y_temp[0] for i in Y] ,c='black',linestyle=':')
                if y_select == 'beta(t)':
                    ax.scatter([i for i in range(len(Y))], [i+Y_temp[0]  for i in func_beta(X.values, *popt)],s=5 )
                if y_select == 'delta(t)':
                    ax.scatter([i for i in range(len(Y))], [i+Y_temp[0] for i in func_delta(X.values, *popt)],s=5 )
                if y_select == 'gamma(t)':
                    ax.scatter([i for i in range(len(Y))], [i+Y_temp[0] for i in func_gamma(X.values, *popt)],s=5 )
                plt.tight_layout()
                plt.savefig(path_files+"prediction-fitting/MSA_" + str(msa)+y_select + "_for_fitting.png", dpi=600)
                plt.close()
            '''
    MSA_predict_beta.T.to_csv(path_files + 'MSA_predict_true_controller(beta).csv')
    MSA_predict_delta.T.to_csv(path_files + 'MSA_predict_true_controller(delta).csv')
    MSA_predict_gamma.T.to_csv(path_files + 'MSA_predict_true_controller(gamma).csv')
    #########for beta boxplot
    MSA_coefficient_beta = MSA_coefficient_beta.fillna(0)
    MSA_coefficient_delta = MSA_coefficient_delta.fillna(0)
    print('beta', MSA_coefficient_beta.mean())
    print('delta', MSA_coefficient_delta.mean())
    MSA_coefficient_beta.to_csv(path_files + 'MSA_impact_coeffecient_' + 'beta(t)' + '.csv')
    MSA_coefficient_delta.to_csv(path_files + 'MSA_impact_coeffecient_' + 'delta(t)' + '.csv')
    MSA_coefficient_gamma.to_csv(path_files + 'MSA_impact_coeffecient_' + 'gamma(t)' + '.csv')

def boxplot_hori(data,namestr_list,namestr,path_results,color_dict):
    print(color_dict)
    data_df=data.drop(columns=['MSA_code'])
    data_df[0] = -data_df[namestr_list[0]] ###the first two are present in oppistive
    data_df[1] = -data_df[namestr_list[1]]
    data_df[2] = data_df[namestr_list[2]]
    data_df=data_df[[0,1,2]]
    data_df = pd.melt(data_df, value_vars=[0,1,2])
    color_dict_x = dict(zip([0,1,2], ['#4e80c7', '#d49f7b', '#82cc78']))
    print(data_df)
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    sns.violinplot( x='variable', y='value',data=data_df,inner="box",medianprops=dict(color="w", linewidth=1),palette=color_dict_x)#scale="count", inner="box"
    '''
    ax.boxplot(x='variable', y='value',  data=data_df, whis='range', vert=False, positions=np.array([0]),palette=color_dict_x[strx],
                 showcaps=False, widths=0.06, patch_artist=True,
                 boxprops=dict(color=color_dict_x[strx], facecolor=color_dict_x[strx]),
                 whiskerprops=dict(color=color_dict_x[strx], linewidth=1),
                 medianprops=dict(color="w", linewidth=1))
    '''
    ax.axhline(y=0, color='black', linestyle='--',linewidth=0.5)
    if namestr=='beta(t)':
        plt.ylim(-4,4)
    if namestr=='delta(t)':
       plt.ylim(-0.06,0.06)
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    ax.set_xlabel(" ")
    plt.xticks(fontsize=0)
    #plt.tight_layout()
    fig.savefig(path_results+"MSA_coefficient_boxplot_" +"("+namestr+").png",dpi=600)

def boxplot_vertical(data,new_namestr_list,namestr):
    fig = go.Figure()
    for strx in new_namestr_list:
        y_list=[i if abs(i-np.mean(data[strx]))<10 else 0 for i in data[strx]]
        print(namestr,strx,np.mean(y_list))
        fig.add_trace(go.Box(y=y_list,name=strx,boxmean='sd',boxpoints=False ))
    #fig.update_traces(boxpoints='all', jitter=0)
    fig.update_layout(
        boxmode='group',
        template='simple_white',
        xaxis=dict(
            showline=True,
            showgrid=True,
            titlefont=dict(size=15, color='black'),
            tickfont=dict(
                size=15,
                color='black'
            ),
        ),
        yaxis=dict(
            showline=True,
            showgrid=True,
            title_text=namestr,
            titlefont=dict(size=15, color='black'),
            tickfont=dict(
                size=15,
                color='black'
            ),

        ),
        legend=dict(
            x=0.75,
            y=1.15,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=15,
                color="black"
            ),
            bgcolor="rgba(230,236,245,0)",
            borderwidth=0
        ),

    )
    fig.update_layout(showlegend=True)
    # fig.show()
    fig.write_image("analysis-results/MSA_coefficient_boxplot_" + namestr + ".png", width=1000, height=500,
                    scale=10)


def k_means_clustering(MSA_statistics):
    print(MSA_statistics.keys())
    data_df = pd.DataFrame(columns=['age','education','pop','poverty'])
    count=0
    for msa in MSA_statistics['county'].keys():
        data_df.loc[count] = [round(MSA_statistics['65plus'][msa], 3),
                              round(MSA_statistics['education'][msa], 3),np.log10(round(MSA_statistics['pop_estimated'][msa], 3)+1),round(MSA_statistics['poverty'][msa], 3)]
        count += 1
    data_df = data_df.fillna(0)
    centers, labels = find_clusters(data_df.values, 2)
    X=data_df.values
    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes(projection='3d')
    #ax = plt.axes()
    ax.scatter(X[:, 0],X[:, 1],X[:, 2], c=labels,
                s=50, cmap='viridis');
    ax.set_xlabel('age')
    ax.set_ylabel('education')
    ax.set_zlabel('population (log10)')
    ax.axis('tight')
    fig.savefig("analysis-results/MSA_clustering.png",dpi=600)
    count=0
    MSA_statistics['label_cluster']=dict()
    for msa in MSA_statistics['county'].keys():
        if MSA_statistics['pop_estimated'][msa] > 0 and MSA_statistics['hospital bed'][msa] > 0:
            MSA_statistics['label_cluster'][msa]=labels[count]
            count+=1
        else:
            MSA_statistics['label_cluster'][msa]=-1
    f_save = open(path_external+'Population/MSA_demographics.pkl', 'wb')
    pickle.dump(MSA_statistics, f_save)
    f_save.close()



def find_clusters(X, n_clusters, rseed=4):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)

        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels

def validation_infection_dead(MSA_statistics,path_files,namestr_list,delay_d,type_file_str):
    beginingcount=(start_date-start_date).days
    endcount=(end_date-start_date).days
    if type_file_str=='reported':
        df_empty = pd.read_csv('Dataset/MSA_S_I_R_D.csv')
        pre_date='2020-03-31'
    if type_file_str=='sero':
        df_empty = pd.read_csv('Dataset/MSA_S_I_R_D_SERO.csv')
        pre_date='2020-04-12'
    MSA_all = pandas.read_csv(path_external+'covid19-intervention-data/MSA_summary_indicators' + type_file_str + '.csv')
    print(path_files)
    MSA_predict_beta = pandas.read_csv(path_files + 'MSA_predict_true_controller(beta).csv')
    MSA_predict_delta = pandas.read_csv(path_files + 'MSA_predict_true_controller(delta).csv')
    MSA_predict_beta.columns = MSA_predict_beta.iloc[0]
    MSA_predict_beta = MSA_predict_beta.iloc[1:].reset_index(drop=True)
    MSA_predict_delta.columns = MSA_predict_delta.iloc[0]
    MSA_predict_delta = MSA_predict_delta.iloc[1:].reset_index(drop=True)

    MSA_predict_beta = MSA_predict_beta.fillna(0)
    MSA_predict_delta = MSA_predict_delta.fillna(0)
    MSA_S_I_R_D_predict=pd.DataFrame(columns=['MSA_code','date','Id','Rd','Dd','I','R','D','I_low','R_low','D_low','I_high','R_high','D_high','beta_d','beta','delta_d','delta'])
    count=0
    for msa in pd.unique(MSA_all['MSA_code']):
        #if msa in [35620,31080, 16980, 19100, 26420, 47900, 33100, 37980, 12060, 38060, 14460, 41860, 40140, 19820]:
            infection_data=df_empty[df_empty['MSA_code']==msa]
            Sd = infection_data[infection_data['time'] == pre_date]['Sd'].values[0]
            Id = infection_data[infection_data['time'] == pre_date]['Id'].values[0]
            Rd = infection_data[infection_data['time'] == pre_date]['Rd'].values[0]
            Dd = infection_data[infection_data['time'] == pre_date]['Dd'].values[0]
            sum_pop = Sd+Id+Rd+Dd
            print(msa,Sd,Id,Rd,Dd)
            [S_0, S_1, S_2] = [Sd/sum_pop, Sd/sum_pop, Sd/sum_pop]
            [I_0,I_1,I_2]=[Id/sum_pop,Id/sum_pop,Id/sum_pop]
            [R_0,R_1,R_2] = [Rd/sum_pop, Rd/sum_pop, Rd/sum_pop]
            [D_0,D_1,D_2]= [Dd/sum_pop, Dd/sum_pop, Dd/sum_pop]
            u=MSA_statistics['birth_death_rate'][msa]
            beta_0=MSA_all[MSA_all['MSA_code']==int(msa)]['beta_0'].values[0]
            delta_0 = MSA_all[MSA_all['MSA_code'] == int(msa)]['delta_0'].values[0]
            list_beta= MSA_all[MSA_all['MSA_code'] == int(msa)]['beta(t)'][beginingcount:endcount]
            list_gamma = MSA_all[MSA_all['MSA_code'] == int(msa)]['gamma(t)'][beginingcount:endcount]
            list_delta = MSA_all[MSA_all['MSA_code'] == int(msa)]['delta(t)'][beginingcount:endcount]
            predict_u_list_beta = [float(i)+float(beta_0) for i in MSA_predict_beta[str(int(msa)) + 'predict_value']][beginingcount:endcount]
            predict_u_list_beta_lower = [float(i)+float(beta_0) for i in MSA_predict_beta[str(int(msa)) + 'bound_lower']][beginingcount:endcount]
            predict_u_list_beta_upper = [float(i)+float(beta_0) for i in MSA_predict_beta[str(int(msa)) + 'bound_upper']][beginingcount:endcount]
            predict_u_list_delta =[float(i)+float(delta_0) for i in MSA_predict_delta[str(int(msa)) + 'predict_value']][beginingcount:endcount]
            predict_u_list_delta_lower = [float(i)+float(delta_0)  for i in MSA_predict_delta[str(int(msa)) + 'bound_lower']][beginingcount:endcount]
            predict_u_list_delta_upper = [float(i)+float(delta_0) for i in MSA_predict_delta[str(int(msa)) + 'bound_upper']][beginingcount:endcount]
            datelist_temp=MSA_predict_beta['type'][beginingcount:endcount]
            for t,gamma,beta_true,delta_true,beta,beta_low,beta_high, delta, delta_low,delta_high in \
                    zip(datelist_temp,list_gamma,list_beta,list_delta,predict_u_list_beta,predict_u_list_beta_lower,predict_u_list_beta_upper,
                        predict_u_list_delta,predict_u_list_delta_lower,predict_u_list_delta_upper):
                if beta<0 or np.isnan(beta)==True:
                    beta=0.00001
                if delta<0 or np.isnan(delta)==True:
                    delta=0.00001
                if beta_low<0 or np.isnan(beta_low)==True:
                    beta_low=0.00001
                if delta_low<0 or np.isnan(delta_low)==True:
                    delta_low=0.00001
                if beta_high<0 or np.isnan(beta_high)==True:
                    beta_high=0.00001
                if delta_high<0 or np.isnan(delta_high)==True:
                    delta_high=0.00001
                #print(t, u, gamma, beta, delta,beta_low,delta_low,beta_high,delta_high)
                Id = infection_data[infection_data['time'] == str(t)[0:10]]['Id'].values[0]
                Rd = infection_data[infection_data['time'] == str(t)[0:10]]['Rd'].values[0]
                Dd = infection_data[infection_data['time'] == str(t)[0:10]]['Dd'].values[0]

                #####predict
                sol = solve_ivp(SIRD, [0, 1], [S_0, I_0, R_0, D_0] + [u, beta, beta/2.664, delta], t_eval=np.arange(0, 1 + 0.2, 0.2))
                [S_0, I_0, R_0, D_0]  = [sol.y[0][5], sol.y[1][5], sol.y[2][5], sol.y[3][5]]
                #####low
                sol = solve_ivp(SIRD, [0, 1], [S_1, I_1, R_1, D_1] + [u, beta_low, beta_low/2.664, delta_low],t_eval=np.arange(0, 1 + 0.2, 0.2))
                [S_1, I_1, R_1, D_1] = [sol.y[0][5], sol.y[1][5], sol.y[2][5], sol.y[3][5]]
                #####predict
                sol = solve_ivp(SIRD, [0, 1], [S_2, I_2, R_2, D_2] + [u, beta_high, beta_high/2.664, delta_high],t_eval=np.arange(0, 1 + 0.2, 0.2))
                [S_2, I_2, R_2, D_2] = [sol.y[0][5], sol.y[1][5], sol.y[2][5], sol.y[3][5]]
                MSA_S_I_R_D_predict.loc[count]=[msa,str(t)[0:10],Id,Rd,Dd,I_0*sum_pop, R_0*sum_pop, D_0*sum_pop,
                                                I_1*sum_pop, R_1*sum_pop, D_1*sum_pop,I_2*sum_pop, R_2*sum_pop, D_2*sum_pop,beta_true,beta,delta_true,delta]
                #print(count,msa,str(t)[0:10],Id,Rd,Dd,I_0*sum_pop, R_0*sum_pop, D_0*sum_pop,I_1*sum_pop, R_1*sum_pop, D_1*sum_pop,I_2*sum_pop, R_2*sum_pop, D_2*sum_pop)
                count+=1
    MSA_S_I_R_D_predict.to_csv(path_files+'MSA_S_I_R_D_predict.csv')


def plot_predict_true_infection( MSA_statistics,MSA_predict,path_files):
    traning_count = (datetime.datetime.strptime('2020-10-18', '%Y-%m-%d') - end_date).days
    datelist = []
    timelist = []
    for date in np.unique(MSA_predict['date']):
        d = datetime.datetime.strptime(str(date)[0:10], '%Y-%m-%d')
        timelist.append(date)
        d = d.strftime('%b-%d,%Y')
        datelist.append(d[0:6])
    xlist=[i for i in range(len(datelist))]
    y_true=[];y_pred=[];
    y_true1 = [];y_pred1 = [];
    fall_infect, (ax1_all, ax2_all) = plt.subplots(1, 2, figsize=(8, 3))
    fall_dead, (ax3_all,ax4_all) = plt.subplots(1, 2, figsize=(8, 3))
    for ax in [ax1_all, ax2_all,ax3_all,ax4_all]:
        ax.plot([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], linewidth=0.5,alpha=0.5,color = '#1f78b4')
        ax.plot([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], linewidth=0.5,alpha=0.5,color = '#1f78b4')
    for msa in pd.unique(MSA_predict['MSA_code']):
        MSA_predict_temp=MSA_predict[MSA_predict['MSA_code']==msa]
        MSA_predict_temp=MSA_predict_temp[MSA_predict_temp['date'].isin(timelist)]
        True_I=MSA_predict_temp['Id'].values
        True_D=MSA_predict_temp['Dd'].values
        True_R = MSA_predict_temp['Rd'].values
        Predict_I =MSA_predict_temp['I'].values
        Predict_I_low =MSA_predict_temp['I_low'].values
        Predict_I_high=MSA_predict_temp['I_high'].values
        Predict_D =MSA_predict_temp['D'].values
        Predict_D_low =MSA_predict_temp['D_low'].values
        Predict_D_high =MSA_predict_temp['D_high'].values
        Predict_R = MSA_predict_temp['R'].values
        Predict_R_low = MSA_predict_temp['R_low'].values
        Predict_R_high = MSA_predict_temp['R_high'].values
        color_dict={-1:'#1f78b4',-2:'#fdcdac',-3:'#cbd5e8',-4:'#f4cae4',-5:"#e6f5c9",-6:'#fff2ae',-7:'#f1e2cc'}
        if msa in [14460,35620,26420,41940,16980,41180,12020,39460]:
            print(msa, MSA_statistics['name'][msa])
        showlist=[14460,35620,26420,41940,16980,41180,12020,39460]
        labelist={14460:'Boston',35620:'New York',26420:'Houston',41940:"San Jose",16980: "Chicago",41180:'St. Louis',12020: "Athens", 39460:"Punta Gorda"}
        color = '#1f78b4'
        if msa in [35620]:
            color = '#fb8072'
        if msa in [26420]:
            color = '#fdb462'
        size=MSA_statistics['pop'][msa]/100000
        index = (datetime.datetime.strptime('2020-10-20', '%Y-%m-%d')-end_date).days
        x_temp=[np.log10(i+1) for i in  list(Predict_I)[index:index+1]]
        y_temp=[np.log10(i+1) for i in  list(True_I)[index:index+1]]
        ax1_all.scatter(x_temp, y_temp, linewidth=0, alpha=0.5,s=size,facecolor=color)
        if msa in showlist:
            ax1_all.annotate(labelist[msa], (x_temp[0], y_temp[0]),
                             xytext=(-5, 5),
                             textcoords='offset points', ha='right', va='bottom',color=color,
                             bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.5, edgecolor="none"),
                             arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0', lw=0.3), size=5)
        x_temp = [np.log10(i+1) for i in  list(Predict_D)[index:index+1]]
        y_temp =[np.log10(i+1) for i in  list(True_D)[index:index+1]]
        ax3_all.scatter(x_temp,y_temp, linewidth=0, alpha=0.5,s=size,facecolor=color)
        if msa in showlist:
            ax3_all.annotate(labelist[msa], (x_temp[0],y_temp[0]),
                    xytext=(-5, 5),
                    textcoords='offset points', ha='right', va='bottom',color=color,
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.5, edgecolor="none"),
                    arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0', lw=0.3), size=5)
        index = -2
        x_temp = [np.log10(i + 1) for i in list(Predict_I)[index:index + 1]]
        y_temp = [np.log10(i + 1) for i in list(True_I)[index:index + 1]]
        ax2_all.scatter(x_temp[0], y_temp[0], linewidth=0, alpha=0.5, s=size, facecolor=color)
        if msa in showlist:
            ax2_all.annotate(labelist[msa], (x_temp[0], y_temp[0]),
                         xytext=(-5, 5),
                         textcoords='offset points', ha='right', va='bottom',color=color,
                         bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.5, edgecolor="none"),
                         arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0', lw=0.3), size=5)
        x_temp = [np.log10(i + 1) for i in list(Predict_D)[index:index + 1]]
        y_temp = [np.log10(i + 1) for i in list(True_D)[index:index + 1]]
        ax4_all.scatter(x_temp, y_temp, linewidth=0, alpha=0.5, s=size, facecolor=color)
        if msa in showlist:
            ax4_all.annotate(labelist[msa], (x_temp[0], y_temp[0]),
                         xytext=(-5, 5),
                         textcoords='offset points', ha='right', va='bottom',color=color,
                         bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.5, edgecolor="none"),
                         arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0', lw=0.3), size=5)
        if msa in []:
        #if msa in [35620, 31080, 16980, 19100, 26420, 47900, 33100, 37980, 12060, 38060, 14460, 41860, 40140,19820]:
            print(msa,MSA_statistics['name'][int(msa)])
            f, (ax1,ax2,ax3,) = plt.subplots(3, 1, figsize=(6.5, 9))
            ax1.bar(xlist, True_I, width=0.2, color='grey', align='center',
                    label="reported data")
            ax1.plot(xlist, Predict_I, color='#99000d', alpha=0.8, linewidth=0.5,
                     label="predicted data")
            ax1.fill_between(xlist, Predict_I_low, Predict_I_high, color='#fc9272', alpha=0.5)
            ax1.legend(loc=1)
            ax1.text(0, 1.05, MSA_statistics['name'][int(msa)], transform=ax1.transAxes)
            # ax1.text(0, 0.95, r'$R^2=%s$' % (round(r_sq,3)),transform=ax1.transAxes)
            ax1.set_xticks(np.arange(0, len(xlist), 30))
            ax1.set_xticklabels([datelist[i * 30] for i in range(0, len(np.arange(0, len(datelist), 30)))], fontsize=8,
                                rotation=45)
            ax1.set_ylabel('Infected Cases (I(t))', fontsize=10)
            ax1.set_yscale('log')

            ax2.bar(xlist, True_R, width=0.2, color='grey', align='center',
                    label=r'$U_{\gamma}(t)$' + " from reported data")

            ax2.plot(xlist, Predict_R, color='#99000d', alpha=0.8, linewidth=0.5,
                     label=r'$U_{\gamma}(t)$' + " from intervention")
            ax2.fill_between(xlist, Predict_R_low, Predict_R_high, color='#fc9272', alpha=0.5)
            # ax2.text(0, 1.05, MSA_statistics['name'][int(msa)], transform=ax2.transAxes)
            # ax2.text(0, 0.95, r'$R^2=%s$' % (round(r_sq,3)), transform=ax2.transAxes)
            ax2.set_xticks(np.arange(0, len(xlist), 30))
            ax2.set_xticklabels([datelist[i * 30] for i in range(0, len(np.arange(0, len(datelist), 30)))], fontsize=8,
                                rotation=45)
            ax2.set_ylabel('Recovered Cases (R(t))', fontsize=10)
            ax2.set_yscale('log')

            ax3.bar(xlist, True_D, width=0.2, color='grey', align='center',
                    label=r'$U_{\beta}(t)$' + " from reported data")

            ax3.plot(xlist, Predict_D, color='#99000d', alpha=0.8, linewidth=0.5,
                     label=r'$U_{\beta}(t)$' + " from intervention")
            ax3.fill_between(xlist, Predict_D_low, Predict_D_high, color='#fc9272', alpha=0.5)
            #ax2.text(0, 1.05, MSA_statistics['name'][int(msa)], transform=ax2.transAxes)
            # ax2.text(0, 0.95, r'$R^2=%s$' % (round(r_sq,3)), transform=ax2.transAxes)
            ax3.set_xticks(np.arange(0, len(xlist), 30))
            ax3.set_xticklabels([datelist[i * 30] for i in range(0, len(np.arange(0, len(datelist), 30)))], fontsize=8,
                                rotation=45)
            ax3.set_ylabel('Dead Cases (D(t))', fontsize=10)
            ax3.set_yscale('log')
            plt.savefig(path_files + "prediction-infection/predict_true_infection_" + MSA_statistics['name'][int(msa)] + ".png", dpi=2000)
            plt.close()

    for ax in [ax1_all, ax2_all,ax3_all,ax4_all]:
        ax.set_xticks([1, 2, 3,4,5,6])  # choose which x locations to have ticks
        ax.set_xticklabels([r'$10$',r'$10^2$',r'$10^3$',r'$10^4$',r'$10^5$',r'$10^6$'],fontsize=7)  #
        ax.set_yticks([1, 2, 3, 4, 5, 6])  # choose which x locations to have ticks
        ax.set_yticklabels([r'$10$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$'],fontsize=7)  #
        ax.set_xlim(1,6.5)
        ax.set_ylim(1, 6.5)

    fall_infect.savefig(path_files + "prediction-infection/predict_true_infection_all.png", dpi=2000)
    fall_dead.savefig(path_files + "prediction-infection/predict_true_dead_all.png", dpi=2000)

    I_score=[]
    R_score=[]
    D_score=[]
    for date in pd.unique(MSA_predict['date']):
        df_temp=MSA_predict[MSA_predict['date']==date]
        I_true=list(df_temp['Id'].values)
        R_true = list(df_temp['Rd'].values)
        D_true = list(df_temp['Dd'].values)
        I_predict = list(df_temp['I'].values)
        R_predict = list(df_temp['R'].values)
        D_predict = list(df_temp['D'].values)
        slope, intercept, r_value, p_value, std_err = stats.linregress(I_true, I_predict)
        I_score.append(r_value)
        slope, intercept, r_value, p_value, std_err = stats.linregress(R_true, R_predict)
        R_score.append(r_value)
        slope, intercept, r_value, p_value, std_err = stats.linregress(D_true, D_predict)
        D_score.append(r_value)
        print(date,I_score[-1],D_score[-1])
    fall, ax1_all = plt.subplots(1, 1, figsize=(5, 4))
    ax1_all.plot([i for i in range(len(I_score))], I_score,label='infected cases')
    ax1_all.plot([i for i in range(len(I_score))], R_score, label='recovered cases')
    ax1_all.plot([i for i in range(len(D_score))], D_score,label='dead cases')
    ax1_all.set_ylim(0.5, 1.1)
    ax1_all.legend(loc=0)
    ax1_all.set_ylabel("R squared value " + "\n"+'(predicted data VS reported data)')
    ax1_all.set_xticks(np.arange(0, len(xlist), 30))
    ax1_all.set_xticklabels([datelist[i * 30] for i in range(0, len(np.arange(0, len(datelist), 30)))], fontsize=8,
                            rotation=45)
    plt.tight_layout()
    fall.savefig(path_files + "prediction-infection/predict_true_u_all_accuracy.png", dpi=2000)

def validation(MSA_statistics,path_files,namestr_list,delay_d,type_file_str):
    traning_count= (datetime.datetime.strptime('2020-10-18', '%Y-%m-%d')-end_date).days
    MSA_all=pandas.read_csv(path_external+'covid19-intervention-data/MSA_summary_indicators'+type_file_str+'.csv')

    MSA_predict_beta=pandas.read_csv(path_files + 'MSA_predict_true_controller(beta).csv')
    MSA_predict_delta=pandas.read_csv(path_files + 'MSA_predict_true_controller(delta).csv')
    MSA_predict_gamma=pandas.read_csv(path_files + 'MSA_predict_true_controller(gamma).csv')
    MSA_predict_beta.columns = MSA_predict_beta.iloc[0]
    MSA_predict_beta = MSA_predict_beta.iloc[1:].reset_index(drop=True)
    MSA_predict_delta.columns = MSA_predict_delta.iloc[0]
    MSA_predict_delta = MSA_predict_delta.iloc[1:].reset_index(drop=True)
    MSA_predict_gamma.columns = MSA_predict_gamma.iloc[0]
    MSA_predict_gamma = MSA_predict_gamma.iloc[1:].reset_index(drop=True)

    datelist = []
    timelist = []
    for date in np.unique(MSA_predict_beta['type']):
        d = datetime.datetime.strptime(str(date)[0:10], '%Y-%m-%d')
        if d >= start_date and d <= end_date:
            timelist.append(date)
            d = d.strftime('%b-%d,%Y')
            datelist.append(d[0:6])

    y_true=[];y_true1=[];
    y_pred=[];y_pred1=[];
    xx=[];yy=[];
    fall, (ax1_all, ax2_all) = plt.subplots(1, 2, figsize=(12, 4))
    for msa in pd.unique(MSA_all['MSA_code']):
        beta_0 = MSA_all[MSA_all['MSA_code'] == int(msa)]['beta_0'].values[0]
        delta_0 = MSA_all[MSA_all['MSA_code'] == int(msa)]['delta_0'].values[0]

        true_u_list_beta =[float(i) for i in  MSA_predict_beta[str(int(msa)) + 'true_value']]
        predict_u_list_beta =[float(i) for i in  MSA_predict_beta[str(int(msa)) + 'predict_value']]
        predict_u_list_beta_lower = [float(i) for i in MSA_predict_beta[str(int(msa)) + 'bound_lower']]
        predict_u_list_beta_upper =[float(i) for i in  MSA_predict_beta[str(int(msa)) + 'bound_upper']]

        beginingcount = 0
        endingcount = (end_date - start_date).days
        predict_u_list_beta = predict_u_list_beta[beginingcount:endingcount- delay_d]
        predict_u_list_beta_lower = predict_u_list_beta_lower[beginingcount:endingcount- delay_d]
        predict_u_list_beta_upper =predict_u_list_beta_upper[beginingcount:endingcount- delay_d]

        true_u_list_beta = true_u_list_beta[beginingcount+ delay_d:endingcount]

        xlist = [i for i in range(len(true_u_list_beta))]

        ##################for delta
        true_u_list_delta =[float(i) for i in   MSA_predict_delta[str(int(msa)) + 'true_value']]
        predict_u_list_delta =[float(i) for i in   MSA_predict_delta[str(int(msa)) + 'predict_value']]
        predict_u_list_delta_lower = [float(i) for i in  MSA_predict_delta[str(int(msa)) + 'bound_lower']]
        predict_u_list_delta_upper = [float(i) for i in  MSA_predict_delta[str(int(msa)) + 'bound_upper']]

        predict_u_list_delta = predict_u_list_delta[beginingcount:endingcount- delay_d]
        predict_u_list_delta_lower = predict_u_list_delta_lower[beginingcount:endingcount- delay_d]
        predict_u_list_delta_upper = predict_u_list_delta_upper[beginingcount:endingcount- delay_d]
        true_u_list_delta = true_u_list_delta[beginingcount+delay_d:endingcount ]

        index=-2
        color = '#1f78b4'
        if msa in [26420,35620] :
            color = 'orange'
        size = MSA_statistics['pop'][msa] / 100000
        x_temp = [np.log10(i + 1) for i in list(predict_u_list_beta)[index:-1]]
        y_temp = [np.log10(i + 1) for i in list(true_u_list_beta)[index:-1]]
        ax1_all.scatter(x_temp, y_temp, linewidth=0, alpha=0.5, s=size, facecolor=color)
        x_temp = [np.log10(i + 1) for i in list(predict_u_list_delta)[index:-1]]
        y_temp = [np.log10(i + 1) for i in list(true_u_list_delta)[index:-1]]
        ax2_all.scatter(x_temp, y_temp, linewidth=0, alpha=0.5, s=size, facecolor=color)

        xx.append(list(predict_u_list_delta)[-1])
        yy.append(list(true_u_list_delta)[-1])
        #if msa in []:
        if msa in [35620, 31080, 16980, 19100, 26420, 47900, 33100, 37980, 12060, 38060, 14460, 41860, 40140,19820]:
            f, (ax1,ax2) = plt.subplots(2, 1, figsize=(6.5, 6))
            ##################for beta

            def neg_tick(x, pos):
                return '%.1f' % (-x if x else 0)  # avoid negative zero (-0.0) labels

            from matplotlib.ticker import FuncFormatter
            ax1.bar(xlist, true_u_list_beta, width=0.2,color='grey',align='center', label=r'$U_{\beta}(t)$')

            ax1.plot(xlist, predict_u_list_beta, color='#99000d', alpha=0.8, linewidth=0.5,label=r'$\hat{U}_{\beta}(t)$')
            ax1.fill_between(xlist, predict_u_list_beta_lower, predict_u_list_beta_upper,color='#fc9272',alpha=0.5)
            ax1.axvspan(len(predict_u_list_beta)+traning_count, len(predict_u_list_beta), alpha=0.2, color='#efedf5')
            ax1.text(0, 1.05, MSA_statistics['name'][int(msa)], transform=ax1.transAxes)
            ax1.legend(loc=1)
            #ax1.text(0, 0.95, r'$R^2=%s$' % (round(r_sq,3)),transform=ax1.transAxes)
            ax1.set_xticks(np.arange(0, len(xlist), 30))
            ax1.set_xticklabels([" " for i in range(0, len(np.arange(0, len(datelist), 30)))], fontsize=8,rotation=45)
            ax1.set_ylabel(r'$\hat{U}_{\beta}(t)$', fontsize=10)
            ax1.set_ylim(np.min(true_u_list_beta)-0.2,np.max(true_u_list_beta))
            ax1.invert_yaxis()
            print('finish1')

            ax2.bar(xlist, true_u_list_delta, width=0.2, color='grey', align='center',label=r'$U_{\delta}(t)$')

            ax2.plot(xlist, predict_u_list_delta, color='#99000d', alpha=0.8, linewidth=0.5,
                     label= r'$\hat{U}_{\delta}(t)$')
            ax2.fill_between(xlist, predict_u_list_delta_lower, predict_u_list_delta_upper, color='#fc9272', alpha=0.5)
            ax2.axvspan(len(predict_u_list_delta) + traning_count, len(predict_u_list_delta), alpha=0.2, color='#efedf5')
            #ax2.text(0, 1.05, MSA_statistics['name'][int(msa)], transform=ax2.transAxes)
            ax2.legend(loc=1)
            #ax2.text(0, 0.95, r'$R^2=%s$' % (round(r_sq,3)), transform=ax2.transAxes)
            ax2.set_xticks(np.arange(0, len(xlist), 30))
            ax2.set_xticklabels([datelist[i * 30] for i in range(0, len(np.arange(0, len(datelist), 30)))], fontsize=8,
                                rotation=45)
            ax2.set_ylabel(r'$U_{\delta}(t)$', fontsize=10)
            ax2.set_ylim(np.min(true_u_list_delta)-0.015, np.max(true_u_list_delta))
            ax2.invert_yaxis()
            print('finish2')
            plt.tight_layout()
            plt.savefig(path_files+"prediction/predict_true_u_"+MSA_statistics['name'][int(msa)]+".png", dpi=2000)
            plt.close()

    ax1_all.set_xlim(-0.8,0)
    ax1_all.set_ylim(-0.8, 0)
    ax1_all.set_ylabel(r'$U_{\beta}(t)$' +" from reported data")
    ax1_all.set_xlabel(r'$U_{\beta}(t)$'+" from interventions")
    ax2_all.set_xlim(-0.05, 0)
    ax2_all.set_ylim(-0.05, 0)
    ax2_all.set_ylabel(r'$U_{\delta}(t)$'+" from reported data")
    ax2_all.set_xlabel(r'$U_{\delta}(t)$'+" from interventions")
    fall.savefig(path_files + "prediction/predict_true_u_all.png", dpi=2000)

    beta_score = []
    gamma_score = []
    delta_score = []
    from sklearn.metrics import r2_score
    gamma_from_delta=[]
    for index,row in MSA_predict_beta.iterrows():
        list_temp=list(row)[1:len(row)]
        list_temp=[float(i) for i in list_temp]
        gamma_from_delta.append(list_temp)
        beta_predict=list_temp[2:len(list_temp)][::4]
        beta_true = list_temp[3:len(list_temp)][::4]
        slope, intercept, r_value, p_value, std_err = stats.linregress(beta_predict, beta_true)
        beta_score.append(r_value)
    for index, row in MSA_predict_gamma.iterrows():
        list_temp = list(row)[1:len(row)]
        list_temp = [float(i) for i in list_temp]
        list_tempx=gamma_from_delta[index]
        gamma_predict = list_tempx[2:len(list_temp)][::4]
        gamma_true = list_temp[3:len(list_temp)][::4]
        slope, intercept, r_value, p_value, std_err = stats.linregress(gamma_predict, gamma_true)
        gamma_score.append(r_value)

    for index, row in MSA_predict_delta.iterrows():
        list_temp = list(row)[1:len(row)]
        list_temp = [float(i) for i in list_temp]
        delta_predict = list_temp[2:len(list_temp)][::4]
        delta_true = list_temp[3:len(list_temp)][::4]
        slope, intercept, r_value, p_value, std_err = stats.linregress(delta_predict, delta_true)
        delta_score.append(r_value)
    fall, (ax1_all, ax2_all, ax3_all) = plt.subplots(1, 3, figsize=(16, 4))
    ax1_all.plot([i for i in range(len(beta_score))], beta_score)
    ax2_all.plot([i for i in range(len(gamma_true))], gamma_true)
    ax3_all.plot([i for i in range(len(delta_score))], delta_score)
    ax1_all.set_ylim(0.5, 1)
    ax1_all.set_ylabel("R squared value " + "\n" + '(estimated '+r'$U_{\beta(t)}$' +'VS analytical ' +r'$U_{\beta(t)}$' +')')
    ax1_all.set_xticks(np.arange(0, len(xlist), 30))
    ax1_all.set_xticklabels([datelist[i * 30] for i in range(0, len(np.arange(0, len(datelist), 30)))], fontsize=8,
                            rotation=45)

    ax2_all.set_ylim(0.5, 1)
    ax2_all.set_xticks(np.arange(0, len(xlist), 30))
    ax2_all.set_xticklabels([datelist[i * 30] for i in range(0, len(np.arange(0, len(datelist), 30)))], fontsize=8,
                            rotation=45)
    ax2_all.set_ylabel("R squared value " + "\n" + '(estimated '+r'$U_{\gamma(t)}$' +'VS analytical ' +r'$U_{\gamma(t)}$' +')')

    ax3_all.set_ylim(0.5, 1)
    ax3_all.set_xticks(np.arange(0, len(xlist), 30))
    ax3_all.set_xticklabels([datelist[i * 30] for i in range(0, len(np.arange(0, len(datelist), 30)))], fontsize=8,
                            rotation=45)
    ax3_all.set_ylabel("R squared value " + "\n" + '(estimated '+r'$U_delta(t)$' +'VS analytical ' +r'$U_delta(t)$' +')')
    plt.tight_layout()
    fall.savefig(path_files + "prediction/predict_true_u_all_accuracy.png", dpi=2000)


def testing_clustering(df_temp,feature,path_files):
    X = df_temp[feature]
    Y=df_temp['label_beta']
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    '''
    from sklearn.linear_model import LinearRegression
    logreg = LinearRegression()#max_iter=1000
    clf=logreg.fit(X_train, y_train)
    print(logreg.coef_, logreg.intercept_)
    plt.scatter(logreg.predict(X_test),y_test.values)
    plt.show()
   
    print("Accuracy on traning:", metrics.accuracy_score(logreg.predict(X_train), y_train.values))
    
    print("Accuracy:", metrics.accuracy_score(logreg.predict(X_test), y_test.values))
    print("Precision:", metrics.precision_score(logreg.predict(X_test), y_test.values))
    print("Recall:", metrics.recall_score(logreg.predict(X_test), y_test.values))
    pred_y=list(logreg.predict(X_test))
    for i in range(len(pred_y)):
        print(list(pred_y)[i],list(y_test)[i])
    
    y_pred_proba = logreg.predict(X_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=" auc=" + str(round(auc,4)))
    plt.legend(loc=4)
    plt.show()

    '''
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(criterion="entropy",max_depth=4)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(scaler.fit_transform(X))
    #for i in range(len(y_pred)):
        #print(y_pred[i],list(Y)[i],df_temp['testing_std'].values[i],df_temp['testing'].values[i],df_temp['65plus'].values[i])

    print("Accuracy on traning:", metrics.accuracy_score(clf.predict(X_train), y_train))
    print("Accuracy on prediting:",metrics.accuracy_score(clf.predict(X_test),y_test))
    print("Accuracy on ALL:", metrics.accuracy_score(y_pred, list(Y)))
    from sklearn.tree import export_graphviz
    from sklearn.externals.six import StringIO
    from IPython.display import Image
    import pydotplus

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature, class_names=['-1', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('cluster_test_tree.png')
    Image(graph.create_png())

def testing_clustering_plot(df_temp,feature,path_files):
    color_list=[ 'red' if i==-1 else 'blue' for i in df_temp['label_beta']]

    for strx in feature:
        df_temp[strx]=[np.log10(i) if i>0 else -2 for i in df_temp[strx]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df_temp['infection'], df_temp['65plus'], df_temp['poverty'], c=color_list, s=10, alpha=0.5)
    ax.set_xlabel('infection')
    ax.set_ylabel('65plus')
    ax.set_zlabel('poverty')
    plt.savefig(path_files + "plot" + 'infection' + "and" + '65plus' + "and" + 'poverty' + ".png",dpi=600)
    plt.close()


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(df_temp['infection'], df_temp['65plus'], c=color_list, s=10,alpha=0.3)
    ax.set_xlabel('infection')
    ax.set_ylabel('65plus')
    ax.axvline(np.log10(0.01), color='k', linestyle='solid')
    ax.axhline(np.log10(0.117), color='k', linestyle='solid')
    plt.savefig(path_files + "plot" + 'infection' + "and" + '65plus' +".png",dpi=600)
    plt.close()

def testing_test():
    from pandas.plotting import scatter_matrix
    path_files = 'results_interventions/reported/'
    MSA_all = pandas.read_csv(path_external+'covid19-intervention-data/MSA_summary_indicators.csv')
    df_X = pd.DataFrame(
        columns=['MSA_code', 'stay_at_home', 'stay_at_home_std', 'facemask', 'facemask_std', 'testing', 'testing_std',
                 'infection', 'infection_std', 'beta(t)', 'beta(t)_std', '65plus', 'poverty', 'eduction',
                 'pop_estimated'])
    count = 0
    for msa in pd.unique(MSA_all['MSA_code']):

        df_temp = MSA_all[MSA_all['MSA_code'] == msa]
        stay_at_home = np.mean(df_temp['ratio of excessive time at home'].values)
        stay_at_home_varianc = (np.var(df_temp['ratio of excessive time at home'].values))
        facemask = (np.mean(df_temp['ratio of people wearing face masks'].values))
        facemask_variance = (np.std(df_temp['ratio of people wearing face masks'].values))
        testing = (np.mean(df_temp['ratio of people taking testing'].values))
        testing_varaince = (np.var(df_temp['ratio of people taking testing'].values))
        infection = np.mean((df_temp['I(t)'].values + df_temp['R(t)'].values + df_temp['D(t)'].values) / (
                    df_temp['I(t)'].values + df_temp['R(t)'].values + df_temp['D(t)'].values + df_temp['S(t)'].values))
        infection_varaince = np.var((df_temp['I(t)'].values + df_temp['R(t)'].values + df_temp['D(t)'].values) / (
                    df_temp['I(t)'].values + df_temp['R(t)'].values + df_temp['D(t)'].values + df_temp['S(t)'].values))
        beta = (np.mean(df_temp['beta(t)'].values))
        beta_varaince = (np.var(df_temp['beta(t)'].values))
        list_temp = []
        for strx in ['65plus', 'poverty', 'education', 'pop_estimated']:
            list_temp.append(MSA_statistics[strx][msa])
        df_X.loc[count] = [msa, stay_at_home, stay_at_home_varianc, facemask, facemask_variance, testing,
                           testing_varaince, infection, infection_varaince, beta, beta_varaince] + list_temp
        count += 1
    df_X.to_csv(path_files + "testing_clustering_df.csv")
    namestr_list_output = ['ratio of excessive time at home', 'ratio of people wearing face masks',
                           'ratio of people taking testing']
    MSA_coefficient_beta = pd.read_csv(path_files + 'MSA_impact_coeffecient_' + 'beta(t)' + '.csv')
    MSA_coefficient_delta = pd.read_csv(path_files + 'MSA_impact_coeffecient_' + 'delta(t)' + '.csv')
    testing_label_beta = dict(zip(MSA_coefficient_beta['MSA_code'], MSA_coefficient_beta[namestr_list_output[2]]))
    testing_label_delta = dict(zip(MSA_coefficient_delta['MSA_code'], MSA_coefficient_delta[namestr_list_output[2]]))
    X = df_X[['stay_at_home', 'stay_at_home_std', 'facemask', 'facemask_std', 'testing', 'testing_std', 'infection',
              'infection_std']]
    label_temp = []
    label_temp2 = []
    for i in pd.unique(MSA_all['MSA_code']):
        if testing_label_beta[i] < 0:
            label_temp.append(-1)
        if testing_label_beta[i] >= 0:
            label_temp.append(1)
        if testing_label_delta[i] < 0:
            label_temp2.append(-1)
        if testing_label_delta[i] >= 0:
            label_temp2.append(1)
    df_X['label_beta'] = label_temp
    df_X['label_delta'] = label_temp2
    df_X['coefficient'] = [testing_label_beta[i] for i in pd.unique(MSA_all['MSA_code'])]
    print('same for beta and delta:',
          len([1 for i, j in zip(label_temp, label_temp2) if i == j]) / len(df_X['MSA_code']))

    feature_all = ['stay_at_home', 'stay_at_home_std', 'facemask', 'facemask_std', 'testing', 'testing_std',
                   'beta(t)', 'beta(t)_std', 'infection', '65plus', 'poverty', 'eduction']

    '''
    fig, ax = plt.subplots(figsize=(50, 40))
    cmap = cm.get_cmap('RdBu')
    df_temp=copy.deepcopy(df_X)
    for strx in feature_all:
        df_temp[strx] = [np.log10(i) if i > 0 else -2 for i in df_temp[strx]]
    sns.pairplot(df_temp[feature_all+['label_beta']], vars=feature_all, hue="label_beta",plot_kws=dict( alpha=0.3))
    plt.tight_layout()
    plt.savefig(path_files+'scatter_matrix.png')
    plt.close()
    '''

    # feature = [ 'stay_at_home_std','testing_std','65plus','poverty']
    # feature = [ 'testing', '65plus', 'infection','pop_estimated','beta(t)','beta(t)_std'] ####decision tree
    testing_clustering(df_X, feature_all, path_files)

    # print(np.median(df_X['infection']))
    # print(np.median(df_X['65plus']))
    print('postive ratio:', len([i for i in label_temp if i == 1]) / len(df_X['MSA_code']))
    hist = 0
    pre_count = 0
    for msa in pd.unique(df_X['MSA_code']):
        df_temp = df_X[df_X['MSA_code'] == msa]
        # if  ( df_temp['poverty'].values>0.12 and df_temp['65plus'].values<0.117 ):#and df_temp['poverty'].values>0.12:
        if (df_temp['infection'].values > 0.01 and df_temp['65plus'].values < 0.117 and df_temp[
            'poverty'].values > 0.12):  #
            if df_temp['label_beta'].values[0] == -1:
                # print(msa,-1, df_temp['label'].values[0])
                pre_count += 1
        else:
            if df_temp['label_beta'].values[0] == 1:
                # print(msa, 1, df_temp['label'].values[0])
                pre_count += 1

    print('predict ratio:', pre_count / len(df_X['MSA_code']))
    path_files_output = 'results_interventions/reported/features/'
    testing_clustering_plot(df_X, feature_all, path_files_output)


def compare_coefficient(df_reported_beta,df_sero_beta,df_reported_delta,df_sero_delta,namestr_list_output):
    fig, ((ax1, ax2, ax3),(ax4,ax5,ax6)) = plt.subplots(2, 3, figsize=(12, 6))
    for ax, NPI_str in zip([ax1,ax2,ax3],namestr_list_output):
        ax.scatter(df_reported_beta[NPI_str], df_sero_beta[NPI_str],color='#377eb8', alpha=0.2, s=10,label='for '+r"$U_{\beta}$")
    for ax, NPI_str in zip([ax4, ax5, ax6], namestr_list_output):
        ax.scatter(df_reported_delta[NPI_str], df_sero_delta[NPI_str], color='orange', alpha=0.2, s=10, label='for '+r"$U_{\delta}$")
    ax1.set_ylabel(r"$w_{\beta}^s$"+' From seroprevalence data')
    ax1.set_xlabel(r"$w_{\beta}^s$"+ ' From reported data')
    ax2.set_ylabel(r"$w_{\beta}^f$" + ' From seroprevalence data')
    ax2.set_xlabel(r"$w_{\beta}^f$" + ' From reported data')
    ax3.set_ylabel(r"$w_{\beta}^t$" + ' From seroprevalence data')
    ax3.set_xlabel(r"$w_{\beta}^t$" + ' From reported data')
    ax4.set_ylabel(r"$w_{\delta}^s$" + ' From seroprevalence data')
    ax4.set_xlabel(r"$w_{\delta}^s$" + ' From reported data')
    ax5.set_ylabel(r"$w_{\delta}^f$" + ' From seroprevalence data')
    ax5.set_xlabel(r"$w_{\delta}^g$" + ' From reported data')
    ax6.set_ylabel(r"$w_{\delta}^t$" + ' From seroprevalence data')
    ax6.set_xlabel(r"$w_{\delta}^t$" + ' From reported data')

    templist=[i-10 for i in range(0, 10)]+[i for i in range(0, 10)]
    ax1.plot(templist, templist, linewidth=0.5, color='grey')
    ax1.set_ylim(-1,0)
    ax1.set_xlim(-1,0)
    ax2.plot(templist, templist, linewidth=0.5, color='grey')
    ax2.set_xlim(-3,0)
    ax2.set_ylim(-3,0)
    ax3.plot(templist, templist, linewidth=0.5, color='grey')
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 2)
    ax4.plot(templist, templist, linewidth=0.5, color='grey')
    ax4.set_xlim(-0.01, 0)
    ax4.set_ylim(-0.01, 0)
    ax5.plot(templist, templist, linewidth=0.5, color='grey')
    ax5.set_xlim(-0.05, 0)
    ax5.set_ylim(-0.05, 0)
    ax6.plot(templist, templist, linewidth=0.5, color='grey')
    ax6.set_xlim(-0.1, 0.1)
    ax6.set_ylim(-0.1, 0.1)

    #plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    #plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.tight_layout()
    plt.savefig('results_interventions/Sero/underreproting_checking_effects_NPIs.png', dpi=600)
    plt.close()

if __name__ == '__main__':
    MSA_statistics = Input.input_MSA_statistics()
    Ctry_statistics = Input.input_Ctry_statistics()
    State_statistics = Input.input_State_statistics()

    case='MSA_reported'

    #case='MSA_sero'

    #case='MSA_reported_incomplete_NPIs'


    if case == 'MSA_reported':
        path_files = 'results_interventions/reported/'

        MSA_List = list(MSA_statistics['county'].keys())
        summary_data_MSA(MSA_statistics, path_files, 'reported')
        ##############DID--------------------------------
        '''
        MSA_all = pandas.read_csv(path_external+'covid19-intervention-data/MSA_summary_indicatorsreported.csv')
        namestr_list_output = [ 'ratio of excessive time at home','ratio of people wearing face masks','ratio of people taking testing']
        namestr_color_dict=dict(zip(namestr_list_output,['#355F8C','#CB7132','#447E36']))

        output_scatter(MSA_all, namestr_list_output,namestr_color_dict, 'MSA',path_files)

        delay_d = 1
        #impact_evaluation_MSA_DID(MSA_statistics, path_files,namestr_list_output,namestr_color_dict,delay_d, 'reported')
        MSA_coefficient_beta=pd.read_csv(path_files + 'MSA_impact_coeffecient_' + 'beta(t)' + '.csv')
        MSA_coefficient_delta=pd.read_csv(path_files + 'MSA_impact_coeffecient_' + 'delta(t)' + '.csv')

        boxplot_hori(MSA_coefficient_beta, namestr_list_output, 'beta(t)', path_files, namestr_color_dict)
        boxplot_hori(MSA_coefficient_delta, namestr_list_output, 'delta(t)', path_files, namestr_color_dict)

        ##############validation--------------------------------
        #validation(MSA_statistics,path_files,namestr_list_output,delay_d,'reported')

        #validation_infection_dead(MSA_statistics,path_files,namestr_list_output,delay_d,'reported')

        MSA_S_I_R_D_predict=pd.read_csv(path_files+'MSA_S_I_R_D_predict.csv')
        plot_predict_true_infection(MSA_statistics,MSA_S_I_R_D_predict,path_files)
        '''
    if case == 'MSA_sero':
        path_files = 'results_interventions/Sero/'
        MSA_List = list(MSA_statistics['county'].keys())
        #summary_data_MSA(MSA_statistics, path_files, 'sero')

        ##############DID--------------------------------
        MSA_all = pandas.read_csv(path_external+'covid19-intervention-data/MSA_summary_indicatorssero.csv')
        namestr_list_output = ['ratio of excessive time at home', 'ratio of people wearing face masks',
                               'ratio of people taking testing']
        namestr_color_dict = dict(zip(namestr_list_output, ['#355F8C', '#CB7132', '#447E36']))

        delay_d = 1
        #impact_evaluation_MSA_DID(MSA_statistics, path_files,namestr_list_output,namestr_color_dict,delay_d, 'sero',training_date)
        MSA_coefficient_beta = pd.read_csv(path_files + 'MSA_impact_coeffecient_' + 'beta(t)' + '.csv')
        MSA_coefficient_delta = pd.read_csv(path_files + 'MSA_impact_coeffecient_' + 'delta(t)' + '.csv')
        #boxplot_hori(MSA_coefficient_beta, namestr_list_output, 'beta(t)', path_files, namestr_color_dict)
        #boxplot_hori(MSA_coefficient_delta, namestr_list_output, 'delta(t)', path_files, namestr_color_dict)

        ##############validation--------------------------------
        #validation(MSA_statistics,path_files,namestr_list_output,delay_d,'sero')
        #validation_infection_dead(MSA_statistics,path_files,namestr_list_output,delay_d,'sero')
        MSA_S_I_R_D_predict=pd.read_csv(path_files+'MSA_S_I_R_D_predict.csv')
        plot_predict_true_infection(MSA_statistics,MSA_S_I_R_D_predict,path_files)

        #########comprecoefficient-------------------
        MSA_coefficient_beta_reported = pd.read_csv('results_interventions/reported/MSA_impact_coeffecient_' + 'beta(t)' + '.csv')
        MSA_coefficient_delta_reported = pd.read_csv('results_interventions/reported/MSA_impact_coeffecient_' + 'delta(t)' + '.csv')

        compare_coefficient(MSA_coefficient_beta_reported,MSA_coefficient_beta,MSA_coefficient_delta_reported,MSA_coefficient_delta,namestr_list_output)









