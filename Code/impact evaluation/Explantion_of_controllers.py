from input_library import *
import Input

def moving_average(list_example, n) :
    new_list=[]
    for i in range(len(list_example)):
        if i<=n:
            new_list.append(np.mean([list_example[j] for j in range(0,i+1)]))
        else:
            new_list.append(np.mean([list_example[j] for j in range(i-n, i + 1)]))

    return new_list

def reset_time_format(df):
    time=[datetime.datetime.strptime(str(datex), '%m/%d/%Y') for datex in df['date']]
    df['date']=time
    return df

def summary_data_MSA(MSA_statistics, MSA_list):
    end_date = datetime.datetime.strptime('2020-10-28', '%Y-%m-%d')
    infection_df = pd.read_csv('analysis-files/MSA_S_I_R_D.csv')
    infection_df['time'] = [datetime.datetime.strptime(str(datex)[0:10], '%Y-%m-%d') for datex in infection_df['time']]

    public_health_df = pd.read_csv('/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/MSA_public_health.csv')
    COVID_spending_df = pd.read_csv('/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/MSA_COVID_spending.csv')
    intervention_policy_df = pd.read_csv(
        '/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/MSA_intervention_policies.csv')
    non_home_dwell_df = pd.read_csv(
        '/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/MSA_non_home_dwell_time.csv')
    face_mask_df = pd.read_csv('/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/MSA_facemask.csv')
    tracing_quarantine_df = pd.read_csv(
        '/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/MSA_tracing_quarantine.csv')

    namestr_list = ['ventilator', 'testing capactity', 'hospitalized', 'cumulative spending', 'mean', 'median_home_dwell_time',
                    'facemask', 'contact tracing', 'quanrantine']
    namestr_list_output = ['ventilator', 'testing capacity', 'hospitalized', 'cumu_fiscal spending',
                           'intervention policy', 'stay home',
                           'cumu_facemask search', 'cumu_tracing', 'cumu_quarantine']
    namestr = dict(zip(namestr_list, namestr_list_output))


    MSA_all = pd.DataFrame(columns=[])
    for msa in MSA_list:
        #print(msa,MSA_statistics['name'][msa])
        MSA_each = pd.DataFrame(columns=[])
        age_ratio = MSA_statistics['65plus'][msa]
        poverty = MSA_statistics['poverty'][msa]
        education =MSA_statistics['education'][msa]
        beds = MSA_statistics['hospital bed']
        f_save = open('analysis-files/two-controllers/infections(ODE)_controller_temporal' + MSA_statistics['name'][msa] + '.pkl', 'rb')
        para_dict = pickle.load(f_save)
        f_save.close()

        timelist = [date for date in para_dict['beta'].keys() if date <= end_date]
        infection_df = infection_df[infection_df['time'].isin(timelist)]

        ########## msa df======================================================

        infection_df_temp=infection_df[infection_df['MSA_code']==msa]
        public_health_df_temp = public_health_df[public_health_df['MSA_code'] == msa]
        COVID_spending_df_temp = COVID_spending_df[COVID_spending_df['MSA_code'] == msa]
        intervention_policy_df_temp = intervention_policy_df[intervention_policy_df['MSA_code'] == msa]
        non_home_dwell_df_temp = non_home_dwell_df[non_home_dwell_df['MSA_code'] == msa]
        face_mask_df_temp = face_mask_df[face_mask_df['MSA'] == msa]
        tracing_quarantine_df_temp= tracing_quarantine_df[tracing_quarantine_df['MSA'] == msa]

        if msa == 40140:
            face_mask_df_temp = face_mask_df[face_mask_df['MSA'] == 31080]
            tracing_quarantine_df_temp = tracing_quarantine_df[tracing_quarantine_df['MSA'] == 31080]

        print(non_home_dwell_df_temp)
        for df_temp, col_str in zip(
                [public_health_df_temp, public_health_df_temp, public_health_df_temp, COVID_spending_df_temp, intervention_policy_df_temp,
                 non_home_dwell_df_temp, face_mask_df_temp, tracing_quarantine_df_temp, tracing_quarantine_df_temp], namestr_list):
            df_temp = copy.deepcopy(df_temp)
            df_temp = reset_time_format(df_temp)
            print(msa, MSA_statistics['name'][msa],col_str)
            df_temp = df_temp.fillna(0)
            # df_temp[col_str] = (df_temp[col_str] - df_temp[col_str].min()) / (df_temp[col_str].max() - df_temp[col_str].min())
            dict_temp = dict(zip(df_temp['date'], df_temp[col_str]))
            #print(list(df_temp[col_str]))
            if col_str in['facemask','contact tracing','quanrantine']:
                new_list_temp = dict()
                sum_temp = 0
                count_key=0
                for key, value in dict_temp.items():
                    sum_temp += value*math.exp(-0.01*count_key)
                    count_key+=1
                    for i in range(7):
                        new_list_temp[key + datetime.timedelta(days=i)] = sum_temp
                dict_temp = new_list_temp

            if col_str == 'spending':
                new_list_temp = dict()
                sum_temp = 0
                for t in timelist:
                    for key, value in dict_temp.items():
                        if key == t:
                            sum_temp += value
                    new_list_temp[t] = sum_temp
                dict_temp = new_list_temp
            if col_str == 'median_non_home_dwell_time':
                for key, value in dict_temp.items():
                    dict_temp[key] = value
            #print([(dict_temp[t]) if t in dict_temp.keys() else 0 for t in timelist])
            maxvalue = np.max(list(dict_temp.values()));
            minvalue = np.min(list(dict_temp.values()))
            if maxvalue==0:
                maxvalue=1
            MSA_each[namestr[col_str]] = [(dict_temp[t]) / (maxvalue) if t in dict_temp.keys() else 0 for t in timelist]
            if col_str == 'median_non_home_dwell_time':
                MSA_each[namestr[col_str]] = [
                    (dict_temp[t] - minvalue) / (maxvalue - minvalue) if t in dict_temp.keys() else 0 for t in timelist]
            if col_str == 'contact tracing' or col_str == 'quanrantine':
                MSA_each[namestr[col_str]] = [(dict_temp[t]) * 0.01666 / (maxvalue) if t in dict_temp.keys() else 0 for t
                                            in timelist]
            # print([(dict_temp[t] - minvalue) / (maxvalue - minvalue) if t in dict_temp.keys() else 0 for t in timelist_earlier])
        MSA_each['MSA_code'] = [msa for i in range(len(timelist))]
        MSA_each['date'] = timelist
        MSA_each['age'] = [age_ratio for i in range(len(timelist))]
        MSA_each['poverty'] = [poverty for i in range(len(timelist))]
        MSA_each['education'] = [education for i in range(len(timelist))]
        MSA_each['hospital bed'] = [beds for i in range(len(timelist))]
        MSA_each['beta(t)'] = [para_dict['beta'][t] for t in timelist]
        MSA_each['gamma(t)'] = [para_dict['gamma'][t] for t in timelist]
        MSA_each['delta(t)'] = [para_dict['delta'][t] for t in timelist]
        MSA_each['S(t)'] = [i / list(infection_df_temp['Sd'])[0] for i in infection_df_temp['Sd']]
        S_dict = dict(zip(infection_df['time'], MSA_each['S(t)']))
        u = np.mean([v for v in MSA_statistics['birth_death_rate'].values() if np.isnan(v) == False])
        MSA_each['R(t)'] = [(para_dict['beta'][t] * S_dict[t]) / (para_dict['gamma'][t] + para_dict['delta'][t] + u) for t in
                          timelist]
        MSA_each.to_csv('/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/MSA_each_files/'+MSA_statistics['name'][msa]+'_summary_indicators.csv')
        MSA_all = MSA_all.append(MSA_each)


        '''
        data_df = MSA_each[namestr_list_output]
        # for strx in ['beta(t)','gamma(t)','delta(t)','R(t)']:
        # data_df[strx]=np.log10(data_df[strx])
        sns_plot = sns.pairplot(data_df, diag_kind='kde')
        sns_plot.savefig("analysis-results/"+MSA_statistics['name'][msa]+"_variable_correlations_all.png")
        '''
    namestr_list_output = ['ventilator', 'testing capacity', 'stay home', 'cumu_facemask search', 'cumu_tracing',
                           'cumu_quarantine']
    output_scatter(MSA_all, namestr_list_output, 'MSA', namestr_list_output)
    MSA_all.to_csv('/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/MSA_summary_indicators.csv')


def factor_analysis_MSA(MSA_statistics,MSA_list):
    MSA_all = pandas.read_csv('/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/MSA_summary_indicators.csv')
    namestr_list_output = ['testing capacity', 'intervention policy','stay home', 'cumu_facemask search', 'cumu_tracing',
                           'cumu_quarantine']
    for strx in namestr_list_output+['beta(t)','gamma(t)','delta(t)','R(t)']:
        MSA_all[strx]=moving_average(list(MSA_all[strx].values),7)
    MSA_all_new = pd.DataFrame(
        columns=['MSA_code','date', 'Factor1', 'Factor2', 'age', 'poverty', 'education', 'hospital bed', 'beta(t)', 'gamma(t)',
                 'delta(t)', 'R(t)'])
    MSA_factor=defaultdict()
    f=open('analysis-files/MSA_factor_analysis.txt','a')
    count=0
    judge = 0
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    cmap = plt.get_cmap('RdYlGn')
    colors = [cmap(i) for i in np.linspace(0, 1, 6)]

    for msa in MSA_list:
        print(MSA_statistics['name'][msa],msa)
        MSA_df=MSA_all[MSA_all['MSA_code'] == msa]
        data_df=MSA_df[namestr_list_output]
        for col in namestr_list_output:
            if data_df[col].sum(axis=0)==0:
                value_list=[random.uniform(0, 0.0001) for i in data_df[col]]
                data_df[col]=value_list
        print(data_df)
        chi_square_value, p_value = calculate_bartlett_sphericity(data_df)
        print(chi_square_value, p_value)
        kmo_all, kmo_model = calculate_kmo(data_df)
        print(kmo_model)

        fa = FactorAnalyzer()
        fa.analyze(data_df, len(list(data_df.columns)), rotation=None)
        # Check Eigenvalues
        ev, v = fa.get_eigenvalues()
        print(ev)

        fa = FactorAnalyzer()
        fa.analyze(data_df, 2, rotation="varimax")
        loading_df=fa.loadings
        print(loading_df)
        print(msa,MSA_statistics['name'][msa],file=f)
        print(loading_df,file=f)
        print("================================",file=f)
        loading_df=loading_df.to_dict()
        markerlist = ['o', '^', 'v', 's', '*', 'p']
        countx = 0
        MSA_factor[msa]=dict(zip(['intervention policy','stay home','testing capacity','cumu_facemask search','cumu_tracing','cumu_quarantine'],[loading_df['Factor2']['intervention policy'],
                                                                                                   loading_df['Factor2']['stay home'],
                                                                                                   loading_df['Factor1']['testing capacity'],
                                                                                                   loading_df['Factor1']['cumu_facemask search'],
                                                                                                   loading_df['Factor1']['cumu_tracing'],
                                                                                                   loading_df['Factor1']['cumu_quarantine']]))
        for strx in namestr_list_output:
            if judge==0:
                plt.scatter(loading_df['Factor1'][strx], loading_df['Factor2'][strx], s=30, marker=markerlist[countx],
                            c=colors[countx],label=strx)
            else:
                plt.scatter(loading_df['Factor1'][strx], loading_df['Factor2'][strx], s=30, marker=markerlist[countx],c=colors[countx])
            countx += 1
        judge=1
        for index,row in MSA_df.iterrows():
            Factor1=0;Factor2=0
            US_loading=dict(zip(['intervention policy','stay home'],[0.939196,0.962854]))
            for strx in ['intervention policy','stay home']:
                if loading_df['Factor2'][strx]>0:
                    Factor2 += row[strx] * loading_df['Factor2'][strx]
            US_loading = dict(zip(['testing capacity','cumu_facemask search','cumu_tracing','cumu_quarantine'], [0.871596,0.983541,0.950296,0.998262 ]))
            for strx in ['testing capacity','cumu_facemask search','cumu_tracing','cumu_quarantine']:
                if loading_df['Factor1'][strx]>0:
                    Factor1 += row[strx] * loading_df['Factor1'][strx]
            MSA_all_new.loc[index] = [msa,row['date'], Factor1, Factor2] + [row[strx] for strx in ['age', 'poverty', 'education', 'hospital bed',
                                                                        'beta(t)', 'gamma(t)', 'delta(t)', 'R(t)']]
            count+=1
            #print(index, row['date'],Non_health_factor,health_factor)
    MSA_all_new.to_csv('analysis-files/MSA_summary_factors.csv')
    f.close()
    plt.axvline(x=0.6, linewidth=0.2)
    plt.axhline(y=0.6, linewidth=0.2)
    ax.legend(loc=0)
    ax.set_xlabel('Factor1')
    ax.set_ylabel('Factor2')
    fig.savefig('analysis-results/MSA_factor_analysis_results.png', dpi=600)

    f_save = open('analysis-files/MSA_factors.pkl', 'wb')
    pickle.dump(MSA_factor, f_save)
    f_save.close()

def multivariate_regression_MSA(MSA_statistics,MSA_list):

    MSA_all=pandas.read_csv('/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/MSA_summary_indicators.csv')
    MSA_list = pd.unique(MSA_all['MSA_code'])
    ##################for multivariate fitting--------------------------------------------------------
    for y_select, y_select_redu in zip(['beta(t)', 'delta(t)'], ['beta_0', 'delta_0']):
        MSA_all[y_select]=[i-j for i,j in zip (MSA_all[y_select],MSA_all[y_select_redu])]
    namestr_list = [ 'ratio of excessive time at home','ratio of people wearing face masks','ratio of people taking testing']
    namestr_list_short=['S','F','T']
    ##########reconstract the dataframe--------------------------------------------------------
    count = 0
    f=open('analysis-results/multivariate_regression_MSA.txt','w')
    MSA_feature = pandas.DataFrame(columns=['MSA_code','outcome', 'delay days', 'feature', 'coefficient', 'p_value', 'R_squared', 'MSE'])
    for msa in MSA_list:
        if MSA_statistics['pop'][msa]>10000:
            print(msa,MSA_statistics['name'][msa])
            MSA_df=copy.deepcopy(MSA_all)
            MSA_df=MSA_df[MSA_df['MSA_code']==msa]
            MSA_df=MSA_df.replace(np.nan, 0)
            y_linear = MSA_df[['beta(t)', 'gamma(t)', 'delta(t)', 'R(t)']]
            x_linear = MSA_df[namestr_list]
            X = x_linear.values;
            Y = y_linear.values
            poly = PolynomialFeatures(2, include_bias=False)
            X_poly = poly.fit_transform(X)
            X_poly_feature_name = poly.get_feature_names(namestr_list_short)
            df_poly = pd.DataFrame(X_poly, columns=X_poly_feature_name)
            for strx in ['beta(t)', 'gamma(t)', 'delta(t)', 'R(t)']:
                df_poly[strx] = list(y_linear[strx].values)
        #####################result--------------------------------------------------------
            feature_function_dict = defaultdict()
            feature_r2_dict = defaultdict()
            for y_selct in ['beta(t)', 'delta(t)']:
                #print('strange',y_selct)
                feature_function_dict[y_selct] = dict()
                feature_r2_dict[y_selct] = dict()

                Y_train = df_poly[[y_selct]]
                X_train = df_poly.drop(['beta(t)', 'gamma(t)', 'delta(t)','R(t)'], axis=1)

                #print("select features")
                for delay_day in [7]:
                    index_x=[i for i in range(len(X_train)) if i<=len(X_train)-delay_day-1]
                    index_y=[i for i in range(len(X_train)) if i>delay_day-1]
                    X_train=X_train.iloc[index_x]
                    Y_train =Y_train.iloc[index_y]
                    list_of_feature_all,r2_all,rmse_all=feature_selection_MSA(X_train, Y_train, X_poly_feature_name,y_selct)
                    for list_of_feature,r2,rmse in zip(list_of_feature_all,r2_all,rmse_all):

                        x_train_Result= X_train[list_of_feature]
                        poly = LinearRegression(normalize=True,fit_intercept=False)
                        model_poly = poly.fit(x_train_Result, Y_train)
                        y_poly = poly.predict(x_train_Result)
                        RMSE_poly = np.sqrt(np.sum(np.square(y_poly - Y_train)))
                        #print("Root-mean-square error of simple polynomial model:", RMSE_poly)
                        r_sq = r2_score(Y_train, y_poly)
                        #print("Root-squared value:", r_sq,RMSE_poly)
                        # p_value_list = stats.coef_pval(model_poly, x_train_Result, Y_train)
                        p_value_list = []
                        coefficient_list = list(model_poly.coef_)
                        print(msa, y_selct, list_of_feature, r2, rmse, coefficient_list,file=f)
                        #######################display_results_pandas................................
                        #print([y_selct, delay_day, list_of_feature + ['interc'], coefficient_list, p_value_list,r_sq, float(RMSE_poly)])
                        MSA_feature.loc[count] = [msa, y_selct, delay_day, list_of_feature + ['interc'],
                                                  coefficient_list, p_value_list, r_sq, float(RMSE_poly)]
                        count += 1
                #######################after selecting days................................

                #(x_train_Result, Y_train, y_poly, MSA_statistics['name'][msa], y_selct,list_of_feature_select + ['interc'], coefficient_list, r_sq)
    f.close()
    MSA_feature.to_csv('analysis-files/MSA_features.csv')


def output_predict_regression_MSA(X_train,Y_train,Y_predict,namestr,type_str,feature_name, feature_coefficient,r_sq):
    import matplotlib.tri as tri

    x_true, y_true, z_true = np.asarray(X_train[feature_name[0]]), np.asarray(X_train[feature_name[1]]), np.asarray(
        Y_train[type_str])
    x = np.random.uniform(np.min(x_true), np.max(x_true), 100)
    y = np.random.uniform(np.min(x_true), np.max(y_true), 100)
    x, y = np.meshgrid(x, y)
    z = feature_coefficient[0] * x + feature_coefficient[1] * y + feature_coefficient[2]
    print('begin')
    fig = plt.figure(figsize=(7, 5))
    ax = fig.gca(projection='3d')
    #
    # ax.contour3D(X, Y, Z, 50, cmap='binary')
    #ax.plot_surface(x,y,z, rstride=1, cstride=1, alpha=0.2,color='grey')
    ax.scatter(x_true, y_true, z_true, facecolor=(0, 0, 0), s=10, marker='o', edgecolor='k')
    #ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none', alpha=0.1)
    ax.set_xlabel(feature_name[0])
    ax.set_ylabel(feature_name[1])
    ax.set_zlabel(type_str)
    ax.axis('tight')

    ax.text2D(0.05, 0.95, r'$Y = %.3f($' % feature_coefficient[0] + feature_name[0] + r'$)+%.3f($' % feature_coefficient[1] +
            feature_name[1] + r'$)+%.3f$' % feature_coefficient[2],
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, fontsize=10)
    ax.text2D(0.05, 0.9,
            r'$R^2 = %.3f$' % r_sq,
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, fontsize=10)

    fig.savefig("analysis-results/"+namestr+type_str+'_regression_result.png',dpi=600)


def summary_data_US(MSA_statistics):
    end_date=datetime.datetime.strptime('2020-10-28', '%Y-%m-%d')
    infection_df=pd.read_csv('analysis-files/US_S_I_R_D.csv')
    infection_df['time']=[datetime.datetime.strptime(str(datex)[0:10], '%Y-%m-%d') for datex in infection_df['time']]

    public_health_df=pd.read_csv('/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/US_public_health.csv')
    COVID_spending_df = pd.read_csv('/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/US_COVID_spending.csv')
    intervention_policy_df= pd.read_csv('/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/US_intervention_policies.csv')
    non_home_dwell_df= pd.read_csv('/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/US_non_home_dwell_time.csv')
    face_mask_df = pd.read_csv('/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/US_facemask.csv')
    tracing_quarantine_df = pd.read_csv('/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/US_tracing_quarantine.csv')

    f_save = open('analysis-files/two-controllers/infections(ODE)_controller_temporal' + 'US'+ '.pkl', 'rb')
    para_dict = pickle.load(f_save)
    f_save.close()

    age_ratio = np.mean(list(MSA_statistics['65plus'].values()))
    poverty = np.mean(list(MSA_statistics['poverty'].values()))
    education = np.mean([ i for i in MSA_statistics['education'].values() if np.isnan(i)==False])
    beds=np.sum(list(MSA_statistics['hospital bed'].values()))
    namestr_list = ['ventilator', 'testing capactity', 'hospitalized', 'spending', 'mean', 'median_non_home_dwell_time',
                    'facemask','contract tracing', 'quanrantine']
    namestr_list_output = ['ventilator', 'testing capacity', 'hospitalized', 'cumu_fiscal spending', 'intervention policy', 'stay home',
                    'cumu_facemask search','cumu_tracing', 'cumu_quarantine']
    namestr=dict(zip(namestr_list,namestr_list_output))
    US_all = pd.DataFrame(columns=[])

    timelist=[date for date in para_dict['beta'].keys() if date<=end_date]
    for df_temp,col_str in zip([public_health_df,public_health_df,public_health_df,COVID_spending_df,intervention_policy_df,non_home_dwell_df,face_mask_df,tracing_quarantine_df,tracing_quarantine_df],namestr_list):
        #print(col_str)
        df_temp=copy.deepcopy(df_temp)
        df_temp=reset_time_format(df_temp)

        df_temp=df_temp.fillna(0)
        #df_temp[col_str] = (df_temp[col_str] - df_temp[col_str].min()) / (df_temp[col_str].max() - df_temp[col_str].min())
        dict_temp = dict(zip(df_temp['date'], df_temp[col_str]))
        if col_str in ['facemask','contract tracing', 'quanrantine']:
            new_list_temp=dict()
            sum_temp=0
            count_key=0
            for key,value in dict_temp.items():
                sum_temp+=value*math.exp(-0.01*count_key)
                count_key+=1
                for i in range(7):
                    new_list_temp[key+datetime.timedelta(days=i)]=sum_temp
            dict_temp=new_list_temp

        if col_str == 'spending':
            new_list_temp=dict()
            sum_temp = 0
            for t in timelist:
                for key, value in dict_temp.items():
                    if key==t:
                        sum_temp += value
                new_list_temp[t]=sum_temp
            dict_temp=new_list_temp
        if col_str=='median_non_home_dwell_time':
            for key, value in dict_temp.items():
                dict_temp[key]=1-value

        maxvalue=np.max(list(dict_temp.values()));minvalue=np.min(list(dict_temp.values()))
        US_all[namestr[col_str]] = [(dict_temp[t]) / (maxvalue) if t in dict_temp.keys() else 0 for t in timelist]
        if col_str=='median_non_home_dwell_time':
            US_all[namestr[col_str]] = [(dict_temp[t]-minvalue) / (maxvalue-minvalue) if t in dict_temp.keys() else 0 for t in timelist]
        if col_str == 'contract tracing' or col_str =='quanrantine':
            US_all[namestr[col_str]] = [(dict_temp[t])*0.01666/ (maxvalue) if t in dict_temp.keys() else 0 for t in timelist]
        #print([(dict_temp[t] - minvalue) / (maxvalue - minvalue) if t in dict_temp.keys() else 0 for t in timelist_earlier])
    US_all['date']=timelist
    US_all['age'] = [age_ratio for i in range(len(timelist))]
    US_all['poverty'] = [poverty for i in range(len(timelist))]
    US_all['education'] = [education for i in range(len(timelist))]
    US_all['hospital bed'] = [beds for i in range(len(timelist))]
    US_all['beta(t)']=[para_dict['beta'][t] for t in timelist]
    US_all['gamma(t)'] = [para_dict['gamma'][t] for t in timelist]
    US_all['delta(t)'] = [para_dict['delta'][t] for t in timelist]

    infection_df = infection_df[infection_df['time'].isin(timelist)]
    US_all['S(t)'] = [i / list(infection_df['Sd'])[0] for i in infection_df['Sd']]
    S_dict=dict(zip(infection_df['time'],US_all['S(t)']))
    u=np.mean([ v for v in MSA_statistics['birth_death_rate'].values() if np.isnan(v)==False])
    US_all['R(t)'] = [(para_dict['beta'][t]*S_dict[t])/(para_dict['gamma'][t]+para_dict['delta'][t]+u) for t in timelist]
    US_all.to_csv('/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/US_summary_indicators.csv')

   # namestr_list_output = ['ventilator', 'testing capacity',  'stay home', 'cumu_facemask search', 'cumu_tracing', 'cumu_quarantine']
    namestr_list_output = ['testing capacity', 'intervention policy', 'stay home', 'cumu_facemask search',
                           'cumu_tracing','cumu_quarantine']


    data_df = US_all[namestr_list_output]
    #for strx in ['beta(t)','gamma(t)','delta(t)','R(t)']:
        #data_df[strx]=np.log10(data_df[strx])
    sns_plot = sns.pairplot(data_df,diag_kind = 'kde')
    sns_plot.savefig("analysis-results/US_variable_correlations_all.png")
    '''
    print(list(data_df['testing capacity'].values))
    grid = sns.PairGrid(data=data_df,
                        vars=namestr_list_output, size=4)
    grid = grid.map_upper(plt.scatter, color='darkred')
    grid = grid.map_diag(plt.hist, bins=30, color='darkred',
                         edgecolor='k')
    # Map a density plot to the lower triangle
    grid = grid.map_lower(sns.kdeplot, cmap='Reds')
    
    grid.savefig("analysis-results/US_variable_correlations.png")
    '''
    output_scatter(US_all, namestr_list_output, 'US', namestr_list_output)


def factor_analysis_US():
    US_all = pandas.read_csv('/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/US_summary_indicators.csv')

    namestr_list_output = ['testing capacity', 'intervention policy', 'stay home', 'cumu_facemask search', 'cumu_tracing',
     'cumu_quarantine']
    for strx in namestr_list_output+['beta(t)','gamma(t)','delta(t)','R(t)']:
        US_all[strx]=moving_average(list(US_all[strx].values),7)
    data_df=US_all[namestr_list_output]
    print(data_df)
    chi_square_value, p_value = calculate_bartlett_sphericity(data_df)
    print(chi_square_value, p_value)
    kmo_all, kmo_model = calculate_kmo(data_df)
    print(kmo_model)


    fa = FactorAnalyzer()
    fa.analyze(data_df, len(namestr_list_output), rotation=None)
    # Check Eigenvalues
    ev, v = fa.get_eigenvalues()
    print(ev)

    fa = FactorAnalyzer()
    fa.analyze(data_df, 2, rotation="varimax")
    loading_df=fa.loadings
    print(loading_df)

    US_all_new = pd.DataFrame(columns=['date','Factor1','Factor2','age','poverty','education','hospital bed','beta(t)','gamma(t)','delta(t)','R(t)'])
    for index,row in US_all.iterrows():
        Factor2=row['stay home']*loading_df['Factor2']['stay home']+row['intervention policy']*loading_df['Factor2']['intervention policy']
        Factor1 =row['testing capacity']*loading_df['Factor1']['testing capacity']+row['cumu_facemask search']*loading_df['Factor1']['cumu_facemask search']+row['cumu_tracing']*loading_df['Factor1']['cumu_tracing']+row['cumu_quarantine']*loading_df['Factor1']['cumu_quarantine']
        US_all_new.loc[index]=[row['date'],Factor1,Factor2]+[row[strx] for strx in ['age','poverty','education','hospital bed','beta(t)','gamma(t)','delta(t)','R(t)']]
        #print(index, row['date'],Non_health_factor,health_factor)
    '''
    for stx in ['Non-hospital','hospital']:
        maxvalue = np.max(US_all_new[stx])
        US_all_new[stx] = [i / (maxvalue) for i in US_all_new[stx]]
    '''
    US_all_new.to_csv('analysis-files/US_summary_factors.csv')

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    loading_df=loading_df.to_dict()
    markerlist=['o','^','v','s','*','p']
    cmap = plt.get_cmap('RdYlGn')
    colors = [cmap(i) for i in np.linspace(0, 1, 6)]
    count=0
    for strx in namestr_list_output:
        plt.scatter(loading_df['Factor1'][strx], loading_df['Factor2'][strx], s=50, marker=markerlist[count], c=colors[count],label=strx)
        count+=1
    plt.axvline(x=0.6,linewidth=0.2)
    plt.axhline(y=0.6,linewidth=0.2)
    ax.legend(loc=0)
    ax.set_xlabel('Factor1 - Eigenvalues:'+str(round(ev['Original_Eigenvalues'][0],2)))
    ax.set_ylabel('Factor2 - Eigenvalues:'+str(round(ev['Original_Eigenvalues'][1],2)))
    fig.savefig('analysis-results/US_factor_analysis_results.png',dpi=600)




def multivariate_regression_US():
    US_all=pandas.read_csv('/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/US_summary_factors.csv')
    ##################for multivariate fitting--------------------------------------------------------
    #namestr_list =['Factor1','Factor2','age','poverty','education']
    #namestr_list_short = ['F1', 'F2','a','p','e']
    namestr_list=[ 'ratio of time at home','ratio of people wearing face masks','ratio of people taking testing']
    namestr_list_short=['S','F','T']
    ##########reconstract the dataframe--------------------------------------------------------
    y_linear = US_all[['beta(t)', 'gamma(t)', 'delta(t)', 'R(t)']]
    x_linear = US_all[namestr_list]

    X = x_linear.values;
    Y = y_linear.values
    poly = PolynomialFeatures(2, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_poly_feature_name = poly.get_feature_names(namestr_list_short)
    df_poly = pd.DataFrame(X_poly, columns=X_poly_feature_name)
    df_poly[['beta(t)', 'gamma(t)', 'delta(t)', 'R(t)']] = y_linear

    #####################result--------------------------------------------------------
    US_feature = pandas.DataFrame(columns=['outcome','delay days', 'feature',  'coefficient', 'p_value','R_squared', 'MSE'])

    count = 0
    feature_function_dict = defaultdict()
    feature_r2_dict = defaultdict()
    for y_selct in ['R(t)','beta(t)', 'delta(t)' ]:
        days_select = 0
        r2_select = 0
        list_of_feature_select = []
        X_Y_select=[]

        feature_function_dict[y_selct] = dict()
        feature_r2_dict[y_selct] = dict()

        Y_train = df_poly[[y_selct]]
        X_train = df_poly.drop(['beta(t)', 'gamma(t)', 'delta(t)'], axis=1)

        #print("select features")
        if y_selct=='R(t)':
            delay_day_list=[1,2,3,4,5,6,7,8,9,10]
        for delay_day in delay_day_list:
            index_x=[i for i in range(len(X_train)) if i<=len(X_train)-delay_day-1]
            index_y=[i for i in range(len(X_train)) if i>delay_day-1]
            X_train=X_train.iloc[index_x]
            Y_train =Y_train.iloc[index_y]
            list_of_feature=feature_selection_US(X_train, Y_train, X_poly_feature_name,y_selct)
            x_train_Result= X_train[list_of_feature]
            poly = LinearRegression(normalize=True,fit_intercept=False)
            model_poly = poly.fit(x_train_Result, Y_train)
            y_poly = poly.predict(x_train_Result)
            RMSE_poly = np.sqrt(np.sum(np.square(y_poly - Y_train)))
            #print("Root-mean-square error of simple polynomial model:", RMSE_poly)
            r_sq = r2_score(Y_train, y_poly)
            #print("Root-squared value:", r_sq)
            p_value_list=stats.coef_pval(model_poly, x_train_Result, Y_train)

            feature_function_dict[y_selct][delay_day]=list_of_feature
            feature_r2_dict[y_selct][delay_day]=r_sq
            if r2_select <= r_sq:
                r2_select = r_sq
                days_select = delay_day
                list_of_feature_select = list_of_feature
                X_Y_select = [x_train_Result,Y_train]
                print(delay_day,r_sq,list_of_feature_select)
        if y_selct in  ['beta(t)','gamma(t)', 'delta(t)']:
            delay_day_list=[days_select]
        #######################after selecting days................................
        [x_train_Result, Y_train]=X_Y_select
        poly = LinearRegression(normalize=True,fit_intercept=False)
        model_poly = poly.fit(x_train_Result, Y_train)
        y_poly = poly.predict(x_train_Result)
        RMSE_poly = np.sqrt(np.sum(np.square(y_poly - Y_train)))
        print("Root-mean-square error of simple polynomial model:", RMSE_poly)
        r_sq = r2_score(Y_train, y_poly)
        print("Root-squared value:", r_sq)
        p_value_list = stats.coef_pval(model_poly, x_train_Result, Y_train)
        coefficient_list = list(model_poly.coef_[0]) + list(model_poly.intercept_)
        #######################display_results_pandas................................
        print([y_selct, days_select,list_of_feature_select+['interc'], coefficient_list, p_value_list, r_sq, float(RMSE_poly)])
        US_feature.loc[count] = [y_selct, days_select,list_of_feature_select+['interc'], coefficient_list, p_value_list, r_sq, float(RMSE_poly)]
        count+=1

        #print(list(Y_train.values))
        #print(list(y_poly))
        #print([y_selct, list_of_feature+['interc'], coefficient_list, p_value_list, float(str(r_sq)[0:5]), float(RMSE_poly)])
        output_predict_regression_US(x_train_Result,Y_train,y_poly,'US_', y_selct,list_of_feature_select+['interc'],coefficient_list,r_sq)

    US_feature.to_csv('analysis-files/US_features.csv')

def feature_selection_US(x_train,y_train,X_poly_feature_name,y_select):

    list_of_feature = []
    if y_select == 'beta(t)' or y_select == 'R(t)':
        list_of_feature = [X_poly_feature_name[7], X_poly_feature_name[12]]
    if y_select == 'delta(t)':
        list_of_feature = [X_poly_feature_name[0], X_poly_feature_name[5]]
    features=list_of_feature
    '''
    features=[]
    old_r2 = 0
    features = []
    for k in range(1, 3):
        for i in range(1, 3000):
            list_of_index = random.sample([j for j in range(len(X_poly_feature_name))], k)
            list_of_index.sort(reverse=False)
            list_of_feature = [X_poly_feature_name[j] for j in list_of_index]
            x_train_temp = x_train[list_of_feature]

            poly = LinearRegression(normalize=True,fit_intercept=False)
            model_poly = poly.fit(x_train_temp, y_train)
            y_poly = poly.predict(x_train_temp)
            RMSE_poly = np.sqrt(np.sum(np.square(y_poly - y_train)))
            # print("Root-mean-square error of simple polynomial model:", RMSE_poly)
            coefficient_of_dermination =r2_score(y_train, y_poly)
            if coefficient_of_dermination >= old_r2:
                #print(coefficient_of_dermination,list_of_feature)
                old_r2=coefficient_of_dermination
                features=list_of_feature
    '''
    return features

def feature_selection_MSA(x_train,y_train,X_poly_feature_name,y_select):
    features=[]
    r2=[]
    rmse=[]
    for k in range(2, 4):
        for i in range(1, 10000):
            list_of_index = random.sample([j for j in range(len(X_poly_feature_name))], k)
            list_of_index.sort(reverse=False)
            list_of_feature = [X_poly_feature_name[j] for j in list_of_index]
            x_train_temp = x_train[list_of_feature]
            poly = LinearRegression(normalize=True,fit_intercept=False)
            model_poly = poly.fit(x_train_temp, y_train)
            y_poly = poly.predict(x_train_temp)
            RMSE_poly = np.sqrt(np.sum(np.square(y_poly - y_train)))
            # print("Root-mean-square error of simple polynomial model:", RMSE_poly)
            coefficient_of_dermination =r2_score(y_train, y_poly)
            if coefficient_of_dermination >= 0.7:
                if list_of_feature not in features:
                    features.append(list_of_feature)
                    r2.append(coefficient_of_dermination)
                    rmse.append(RMSE_poly)
    return features,r2,rmse


def predband(x, xd, yd, p, func, conf=0.95):
    alpha = 1.0 - conf  # significance
    N = xd.size  # data sample size
    var_n = len(p)  # number of parameters
    # Quantile of Student's t distribution for p=(1-alpha/2)
    import scipy
    q = scipy.stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
    # Stdev of an individual measurement
    se = np.sqrt(1. / (N - var_n) * \
                 np.sum((yd - func(xd, *p)) ** 2))
    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    # Predicted values (best-fit model)
    yp = func(x, *p)
    # Prediction band
    dy = q * se * np.sqrt(1.0 + (1.0 / N) + (sx / sxd))
    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy
    return lpb, upb


def output_predict_regression_US(X_train,Y_train,Y_predict,namestr,type_str,feature_name, feature_coefficient,r_sq):
    import matplotlib.tri as tri
    x_true, y_true, z_true = np.asarray(X_train[feature_name[0]]), np.asarray(X_train[feature_name[1]]), np.asarray(
        Y_train[type_str])
    x = np.random.uniform(np.min(x_true), np.max(x_true), 100)
    y = np.random.uniform(np.min(x_true), np.max(y_true), 100)
    x, y = np.meshgrid(x, y)
    z = feature_coefficient[0] * x + feature_coefficient[1] * y + feature_coefficient[2]
    print('begin')
    fig = plt.figure(figsize=(7, 5))
    ax = fig.gca(projection='3d')
    #
    # ax.contour3D(X, Y, Z, 50, cmap='binary')
    #ax.plot_surface(x,y,z, rstride=1, cstride=1, alpha=0.2,color='grey')
    ax.scatter(x_true, y_true, z_true, facecolor=(0, 0, 0), s=10, marker='o', edgecolor='k')
    #ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none', alpha=0.1)
    ax.set_xlabel(feature_name[0])
    ax.set_ylabel(feature_name[1])
    ax.set_zlabel(type_str)
    ax.axis('tight')

    ax.text2D(0.05, 0.95, r'$Y = %.3f($' % feature_coefficient[0] + feature_name[0] + r'$)+%.3f($' % feature_coefficient[1] +
            feature_name[1] + r'$)+%.3f$' % feature_coefficient[2],
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, fontsize=10)
    ax.text2D(0.05, 0.9,
            r'$R^2 = %.3f$' % r_sq,
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, fontsize=10)

    fig.savefig("analysis-results/"+namestr+type_str+'_regression_result.png',dpi=600)

    '''
    popt, pcov = curve_fit(f, Y_train, Y_predict)
    r2 = r2_score(Y_predict, f(Y_train,1))
    # calculate parameter confidence interval
    a = unc.correlated_values(popt, pcov)

    px = np.linspace(np.min(Y_train), np.max(Y_train), 100)
    py=f(list(px),1)
    lpb, upb = predband(px, np.asarray(Y_train), np.asarray(Y_predict), popt, f, conf=0.95)
    # calculate regression confidence interval
    nom = unp.nominal_values(py)
    std = unp.std_devs(py)
    
    # plot the regression
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.scatter(Y_train, Y_predict, s=3, label='Data')
    ax.plot(px, nom, c='black', label=r'$fitting$',alpha=0.5)

    # uncertainty lines (95% confidence)
    ax.plot(px, nom - 1.96 * std, c='orange', label='95% Confidence Region',alpha=0.5)
    ax.plot(px, nom + 1.96 * std, c='orange',alpha=0.5)
    # prediction band (95% confidence)
    ax.plot(px, lpb, 'k--', label='95% Prediction Band',alpha=0.5)
    ax.plot(px, upb, 'k--',alpha=0.5)
    ax.legend(loc='best')

    ax.text(0, 0.05,r'$Y = %.3f($' % feature_coefficient[0] + feature_name[0] + r'$)+%.3f($' % feature_coefficient[1] +feature_name[1]+r'$)+%.3f$' % feature_coefficient[2],
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, fontsize=10)
    ax.text(0, 0.1,
            r'$R^2 = %.3f$' % r_sq,
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, fontsize=10)

    ax.set_ylabel('predictive (Y)')
    ax.set_xlabel('true value')
    plt.legend(loc='best')
    fig.savefig("analysis-results/"+namestr+type_str+'_regression_result.png',dpi=600)
    '''

def output_scatter(Data_all,namestr_list,namestr,namestr_list_output):
    fig, ((ax1, ax2, ax3,ax4,ax5,ax6),(ax7,ax8,ax9,ax10,ax11,ax12),
          (ax13,ax14,ax15,ax16,ax17,ax18),(ax19,ax20,ax21,ax22,ax23,ax24)) = plt.subplots(4, 6, figsize=(30, 15))
    if namestr=='MSA':
        Data_all_temp=Data_all.groupby(['date']).mean()
        for col,ax,output_str in zip(namestr_list,[ax1, ax2, ax3,ax4,ax5,ax6],namestr_list_output):
            ax.plot([i for i in range(len(Data_all_temp))],Data_all_temp[col], c='blue')
            ax.set_xlabel('date', fontsize=15)
            ax.set_ylabel(output_str, fontsize=15)
    if namestr=='US':
        for col,ax,output_str in zip(namestr_list,[ax1, ax2, ax3,ax4,ax5,ax6],namestr_list_output):
            ax.plot([i for i in range(len(Data_all[col]))],Data_all[col], c='blue')
            ax.set_xlabel('date', fontsize=15)
            ax.set_ylabel(output_str, fontsize=15)

    for col,ax,output_str in zip(namestr_list,[ax7,ax8,ax9,ax10,ax11,ax12],namestr_list_output):
        ax.scatter(Data_all[col], Data_all['beta(t)'], edgecolors=(0, 0, 0), s=50, c='#e41a1c')
        ax.set_xlabel(output_str, fontsize=15)
        ax.set_ylabel(r'$\beta(t)$', fontsize=15)
        #ax.set_yscale('log')
        #ax.set_xscale('log')

    for col,ax,output_str in zip(namestr_list,[ax13,ax14,ax15,ax16,ax17,ax18],namestr_list_output):
        ax.scatter(Data_all[col], Data_all['gamma(t)'], edgecolors=(0, 0, 0), s=50, c='#377eb8')
        ax.set_xlabel(output_str, fontsize=15)
        ax.set_ylabel(r'$\gamma(t)$', fontsize=15)
        #ax.set_yscale('log')
        #ax.set_xscale('log')

    for col,ax,output_str in zip(namestr_list,[ax19,ax20,ax21,ax22,ax23,ax24],namestr_list_output):
        ax.scatter(Data_all[col], Data_all['delta(t)'], edgecolors=(0, 0, 0), s=50, c='#4daf4a')
        ax.set_xlabel(output_str, fontsize=15)
        ax.set_ylabel(r'$\delta(t)$', fontsize=15)
        #ax.set_yscale('log')
        #ax.set_xscale('log')


    plt.tight_layout()
    plt.savefig("analysis-results/"+namestr+"_beta_gamma_correlation.png",
                dpi=600)
    plt.close()

def coefficient_analysis_MSA(MSA_statistics,MSA_List):
    namestr_list_output = ['testing capacity', 'intervention policy', 'stay home', 'cumu_facemask search',
                           'cumu_tracing',
                           'cumu_quarantine']
    f_save = open('analysis-files/MSA_factors.pkl', 'rb')
    MSA_factors = pickle.load(f_save)
    f_save.close()
    print(MSA_factors)
    MSA_feature_df=pd.read_csv('analysis-files/MSA_features.csv')
    MSA_coefficient_df= pd.DataFrame(columns=['msa','type','age','poverty','education','hosptial bed']+['F1','F2','intercept']+namestr_list_output)
    count=0
    for index,row in MSA_feature_df.iterrows():
        msa=row['MSA_code']
        coeffi_factor1=ast.literal_eval(row['coefficient'])[0]
        coeffi_factor2=ast.literal_eval(row['coefficient'])[1]
        intercept = ast.literal_eval(row['coefficient'])[2]
        coefficientlist=[MSA_factors[msa][strx] if  MSA_factors[msa][strx] >0 else 0 for strx in namestr_list_output]
        coefficientlist=[coefficientlist[0]*coeffi_factor1,coefficientlist[1]*coeffi_factor2,coefficientlist[2]*coeffi_factor2,coefficientlist[3]*coeffi_factor1,coefficientlist[4]*coeffi_factor1,coefficientlist[5]*coeffi_factor1]
        MSA_coefficient_df.loc[count]=[row['MSA_code'],row['outcome']]+[MSA_statistics[strx][msa] for strx in ['65plus','poverty','education','hospital bed']]+ast.literal_eval(row['coefficient'])+coefficientlist
        count+=1
    MSA_coefficient_df.to_csv('analysis-files/MSA_coefficient.csv')

    Data_all=MSA_coefficient_df
    for col_type in ['R(t)','beta(t)','gamma(t)']:
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6),
              (ax7, ax8, ax9)) = plt.subplots(3, 3,figsize=(15, 10))
        Data_all_temp=Data_all[Data_all['type']==col_type]
        print(col_type)
        print(Data_all_temp)
        for col, ax in zip(['F1','F2','intercept'], [ax1, ax2, ax3]):
            ax.scatter(Data_all_temp[col], Data_all_temp['age'], c='blue')
            ax.set_xlabel('coefficient_'+col, fontsize=15)
            ax.set_ylabel('age', fontsize=15)
        for col, ax in zip(['F1', 'F2', 'intercept'], [ax4, ax5, ax6]):
            ax.scatter(Data_all_temp[col], Data_all_temp['poverty'], c='blue')
            ax.set_xlabel('coefficient_'+col, fontsize=15)
            ax.set_ylabel('poverty', fontsize=15)
        for col, ax in zip(['F1', 'F2', 'intercept'], [ax7, ax8, ax9]):
            ax.scatter(Data_all_temp[col], Data_all_temp['education'], c='blue')
            ax.set_xlabel('coefficient_'+col, fontsize=15)
            ax.set_ylabel('education', fontsize=15)

        plt.tight_layout()
        plt.savefig("analysis-results/MSA_coefficient_"+col_type+".png",
                    dpi=600)
        plt.close()

if __name__ == '__main__':
    MSA_statistics = Input.input_MSA_statistics()
    Ctry_statistics = Input.input_Ctry_statistics()
    State_statistics = Input.input_State_statistics()

    case = 'MSA'
    if case=='US':
        #factor_analysis_US()
        multivariate_regression_US()

    if case=='MSA':
        #MSA_List = [35620, 31080, 16980, 19100, 26420, 47900, 33100, 37980, 12060, 38060, 14460, 41860, 40140, 19820]
        MSA_List=list(MSA_statistics['name'].keys())
        multivariate_regression_MSA(MSA_statistics,MSA_List)
        #coefficient_analysis_MSA(MSA_statistics,MSA_List)





