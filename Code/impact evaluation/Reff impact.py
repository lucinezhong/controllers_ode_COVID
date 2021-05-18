import sys
sys.path.append('Project')
from input_library import *
import Input

date_check = ['2021-01-01','2021-01-02','2021-01-03','2021-01-04','2021-01-05','2021-01-06',
              '2021-01-07','2021-01-08','2021-01-09','2021-01-10','2021-01-11']
def func_beta(x, a, b, c):

    return (a*x[0]+1)*(b*x[1]+c*x[2]+1)-1

def func_delta(x, a,b, c):
    return  (a*x[0]+1)*(b*x[1]+c*x[2]+1)-1


def fun_poly(x,a,b,c,d,e,f,g):
    return a*x[0,:]+b*x[1,:]+c*x[0,:]*x[1,:]+d*x[0,:]*x[0,:]+e*x[1,:]*x[1,:]+f*x[1,:]*x[1,:]*x[1,:]+g*x[0,:]*x[0,:]*x[0,:]

def func26420(x, a, b, c,d,Offset):
    return  a*x+Offset+b*x*x

def func35620(x, a, b,Offset):
    return  1.0 / (1.0 + numpy.exp(-a * (x-b)))+ Offset

def reverse_colourmap(cmap, name = 'my_cmap_r'):
    reverse = []
    k = []

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1-t[0],t[2],t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r

def mean_confidence_interval(data, confidence=0.95):
    x = np.array([round(i,3) for i in data])
    n = len(x)
    m, se = np.mean(x), scipy.stats.sem(x)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m

def intervention_for_eff(path_files,name_str, MSA_statistics):
    n = 100
    my_cmap = mpl.cm.BuPu
    my_cmap_r = reverse_colourmap(my_cmap)

    MSA_all = pandas.read_csv(path_external+'covid19-intervention-data/MSA_summary_indicatorsreported.csv')
    MSA_all_x = MSA_all[MSA_all['date'].isin(date_check)]
    print(len(MSA_all),len(MSA_all_x))
    MSA_coefficient_beta = pd.read_csv(path_files + 'MSA_impact_coeffecient_' + 'beta(t)' + '.csv')
    MSA_coefficient_delta = pd.read_csv(path_files + 'MSA_impact_coeffecient_' + 'delta(t)' + '.csv')

    vaccination_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    count=0
    #for msa in [35620, 26420]:#
    for msa in pd.unique(MSA_all['MSA_code']):
        df_interves_eff = pd.DataFrame(columns=['MSA_code', 'stay', 'facemasks', 'testing', 'basic_R'])
        MSA_all_temp = MSA_all_x[MSA_all_x['MSA_code']==msa]
        teting_true=round(np.mean(MSA_all_temp['ratio of people taking testing'].values),2)

        MSA_beta=MSA_coefficient_beta[MSA_coefficient_beta['MSA_code']==msa]
        MSA_delta= MSA_coefficient_delta[MSA_coefficient_delta['MSA_code'] == msa]
        #####set testing
        u =mean_confidence_interval([i for i in MSA_statistics['birth_death_rate'].values() if np.isnan(i)==False ])
        beta_0 = MSA_all_temp['beta_0'].values[0]
        delta_0= MSA_all_temp['delta_0'].values[0]
        S =MSA_all_temp['S(t)'].values[-1]/(MSA_all_temp['S(t)'].values[-1]+MSA_all_temp['I(t)'].values[-1]+MSA_all_temp['R(t)'].values[-1]+MSA_all_temp['D(t)'].values[-1])
        print(msa, teting_true,S)
        gamma= mean_confidence_interval(MSA_all_temp['gamma(t)'])
        ####coefficient
        a=mean_confidence_interval(MSA_beta[name_str[0]].values)
        b = mean_confidence_interval(MSA_beta[name_str[1]].values)
        c= mean_confidence_interval(MSA_beta[name_str[2]].values)
        d= mean_confidence_interval(MSA_delta[name_str[0]].values)
        e= mean_confidence_interval(MSA_delta[name_str[1]].values)
        f= mean_confidence_interval(MSA_delta[name_str[2]].values)

        ######
        countx=0
        z = np.zeros((n, n))
        R_temp_List=[]
        xlist=[]
        ylist=[]
        zlist=[]
        for stay in [x/n for x in range(n)]:
            county=0
            for facemask in [x/n for x in range(n)]:
                for testing in [x*0.2/n for x in range(n)]:
                    xlist.append(stay)
                    ylist.append(facemask)
                    zlist.append(testing)
                    beta_temp = func_beta([stay,facemask,testing], a, b, c)+beta_0
                    delta_temp = func_beta([stay,facemask,testing], d, e, f) + delta_0
                    if beta_temp<0:
                        beta_temp=0
                    if delta_temp<0:
                        delta_temp=0
                    gamma_temp=beta_temp/2.664
                    S_temp = S
                    if S_temp<0:
                        S_temp=0
                    R_temp = beta_temp / (delta_temp + gamma_temp + u)
                    if R_temp>11:
                        R_temp=11
                    R_temp_List.append(R_temp)
                    count+=1
                    if round(testing,2)==teting_true:
                        z[countx][county]=R_temp*S_temp
                        #print(beta_temp,delta_temp,z[countx][county])
                county+=1
            countx +=1
        df_interves_eff['MSA_code']=[msa for i in range(len(xlist))]
        df_interves_eff['stay']=xlist
        df_interves_eff['facemasks'] = ylist
        df_interves_eff['testing'] = zlist
        df_interves_eff['basic_R'] = R_temp_List
        for vaccination in vaccination_list:
            S_temp = S-vaccination*0.9
            if S_temp<0:
                S_temp=0
            df_interves_eff[str(vaccination)]=np.asarray(df_interves_eff['basic_R'].values)*S_temp
        if msa in [35620, 26420]:
            y, x = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
            z = np.asarray(z)
            z_min, z_max = np.abs(z).min(), np.abs(z).max()
            print(z_min, z_max)
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            c = ax.pcolormesh(x, y, z, cmap=my_cmap_r, vmin=z_min, vmax=z_max)
            fig.colorbar(c, ax=ax)
            ax.set_ylabel('ratio of people wearing face masks')
            ax.set_xlabel('ratio of excessive time at home')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.tight_layout()
            fig.savefig(path_files + "Needed_intervention_heatplot"+MSA_statistics['name'][msa]+".png", dpi=2000)

        df_interves_eff.to_csv(path_external+'R_eff/df_interves_eff_temp'+str(msa)+'.csv')
    #print(np.linspace(0, 1, n))



def intervention_for_eff_plot(path_files,namestr):
    #for msa in [35620,26420,]:#
    for msa in [35620, 31080, 16980, 19100, 47900,26420, 33100, 37980, 12060, 38060, 14460, 41860, 40140, 19820]:
        df_interves_eff=pd.read_csv(path_external+'R_eff/df_interves_eff_temp' + str(msa) + '.csv')
        df_temp=df_interves_eff[ (df_interves_eff['0'] < 1.1)]
        print(msa)
        fig = plt.figure(figsize=(6, 4))
        ax = fig.gca(projection='3d')
        color_dict={0.1:'#a6bddb',0.2:'#0570b0',0.3:'#023858'}
        if msa==35620:
            alphax=0.01
            testing_threshold=0.15
        if msa==26420:
            alphax=0.01
            testing_threshold = 0.06
        size=5
        alphax=0.1
        df_tempx = df_interves_eff[(df_interves_eff['testing'] <= testing_threshold + 0.0005) & (
                    df_interves_eff['testing'] >= testing_threshold - 0.0005)]
        my_cmap = mpl.cm.BuPu
        my_cmap_r = reverse_colourmap(my_cmap)
        ax.scatter(df_tempx['stay'], df_tempx['facemasks'], df_tempx['testing'], alpha=0.3, s=5, c=df_tempx['0'],cmap=my_cmap_r,
                   marker='.', linewidth=1)
        df_temp1=df_temp[df_temp['stay']>=0.95]
        df_temp2 = df_temp[df_temp['facemasks']>=0.95]
        df_temp3 = df_temp[df_temp['testing'] >=0.19]
        df_temp4 = df_temp[df_temp['stay'] <= 0.05]
        df_temp5 = df_temp[df_temp['facemasks'] <= 0.05]
        df_temp6 = df_temp[df_temp['testing'] <= 0.005]

        for testing in pd.unique(df_temp1['testing']):
            ylist=[i/100 for i in range(0, 100) if i/100>=np.min(df_temp1['facemasks'])-0.01]
            if testing==0 or testing==0.2:
                alphax=0.1
            else:
                alphax=0.05
            ax.plot([1 for i in range(0, len(ylist))], ylist, [testing for i in range(0, len(ylist))], color='Blue',
                    linewidth=2, alpha=alphax)
        for testing in pd.unique(df_temp4['testing']):
            ylist=[i/100 for i in range(0, 100) if i/100>=np.min(df_temp4['facemasks'])-00.01]
            if testing==0 or testing==0.2:
                alphax=0.1
            else:
                alphax=0.05
            ax.plot([0 for i in range(0, len(ylist))], ylist, [testing for i in range(0, len(ylist))], color='Blue',
                    linewidth=2, alpha=0.001)
        for testing in pd.unique(df_temp2['testing']):
            xlist=[i/100 for i in range(0, 100) if i/100>=np.min(df_temp2['stay'])-0.01]
            if testing==0 or testing==0.2:
                alphax=0.1
            else:
                alphax=0.05
            ax.plot(xlist,[1 for i in range(0, len(xlist))], [testing for i in range(0, len(xlist))], color='Blue',
                    linewidth=2, alpha=alphax)
        for testing in pd.unique(df_temp5['testing']):
            xlist=[i/100 for i in range(0, 100) if i/100>=np.min(df_temp5['stay'])-00.01]
            if testing==0 or testing==0.2:
                alphax=0.1
            else:
                alphax=0.05
            ax.plot(xlist,[0 for i in range(0, len(xlist))], [testing for i in range(0, len(xlist))], color='Blue',
                    linewidth=2, alpha=alphax)
        for stay in list(pd.unique(df_temp3['stay'])):
            df_x = df_temp3[df_temp3['stay'] == stay]
            if stay==0 or stay==1 or stay==0.01 or stay==0.99:
                alphax=0.1
            else:
                alphax=0.05
            ylist = [i / 100 for i in range(0, 100) if i / 100 >= np.min(df_x['facemasks'])-0.01]
            ax.plot([stay for i in range(0, len(ylist))], ylist,[0.2 for i in range(0, len(ylist))],color='Blue',
                    linewidth=2, alpha=alphax)
        ax.plot([1 for i in range(0, len(ylist))], ylist, [0.2 for i in range(0, len(ylist))], color='Blue',
                linewidth=2, alpha=0.05)
        for stay in list(pd.unique(df_temp6['stay'])):
            df_x = df_temp6[df_temp6['stay'] == stay]
            ylist = [i / 100 for i in range(0, 100) if i / 100 >= np.min(df_x['facemasks'])-0.01]
            if stay==0 or stay==1 or stay==0.01 or stay==0.99:
                alphax=0.1
            else:
                alphax=0.05
            ax.plot([stay for i in range(0, len(ylist))], ylist, [0 for i in range(0, len(ylist))], color='Blue',
                    linewidth=2, alpha=0.05)
        ax.plot([1 for i in range(0, len(ylist))], ylist, [0 for i in range(0, len(ylist))], color='Blue',
                linewidth=2, alpha=alphax)

        df_temp7 = df_interves_eff[(df_interves_eff['0'] <= 1.05) & (df_interves_eff['0'] >= 0.95)]
        for testing in pd.unique(df_temp7['testing']):
            df_x = df_temp7[df_temp7['testing'] == testing]
            if msa==35620:
                fittedParameters, pcov = curve_fit(func35620, np.asarray(df_x['stay'].values), df_x['facemasks'].values, maxfev=100000)
                xlist = [i / 100 for i in range(0, 100)]
                ground_test_new = func35620(np.asarray(xlist), *fittedParameters)
            else:
                if msa in [26420,12060,37980,41860]:
                    fittedParameters, pcov = curve_fit(func26420, np.asarray(df_x['stay'].values), df_x['facemasks'].values,
                                                       maxfev=100000)
                    xlist = [i / 100 for i in range(0, 100) if i / 100 >= np.min(df_x['stay'].values)]
                    ground_test_new = func26420(np.asarray(xlist), *fittedParameters)
                else:
                    fittedParameters, pcov = curve_fit(func35620, np.asarray(df_x['stay'].values), df_x['facemasks'].values,maxfev=100000)
                    xlist = [i / 100 for i in range(0, 100)]
                    ground_test_new = func35620(np.asarray(xlist), *fittedParameters)

            ax.plot(xlist, ground_test_new, [testing for i in range(0, len(xlist))], color='Blue',linewidth=2, alpha=0.05)


        ax.set_xlim(0,1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 0.2)

        ax.grid(False)
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
        # make the grid lines transparent
        
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

        '''
        
        ax.set_xlabel('ratio of excessive time at home',fontsize=7)
        ax.set_ylabel('ratio of people wearing face masks',fontsize=7)
        ax.set_zlabel('ratio of people take testing', fontsize=7)
        '''
        ax.set_xticklabels([0,0.2,0.4,0.6,0.8,1], fontsize=7)
        ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=7)
        ax.set_zticks([0, 0.04, 0.08, 0.12, 0.16, 0.2])
        ax.set_zticklabels([0, 0.04, 0.08, 0.12, 0.16, 0.2], fontsize=7)
        ax.xaxis._axinfo['juggled'] = (0, 0, 2)
        ax.yaxis._axinfo['juggled'] = (0, 1, 2)
        ax.zaxis._axinfo['juggled'] = (0, 2, 1)
        fig.savefig(path_files+"R_eff/"+str(msa)+'thresholds.png',dpi=600)

def lessening_ratios(path_files):
    vaccination_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    files = os.listdir(path_external+'R_eff/')  # 获取文件夹下的所有文件名
    df_lessen_ratio = pd.DataFrame(columns=['R', 'MSA_code', 'vaccination', 'stay', 'facemasks','testing'])
    count=0
    example1=[]
    example2=[]
    for file in files[1:len(files)]:
        df = pandas.read_csv(path_external+"R_eff/" + file)
        df_temp = df[(df['0'] < 1.05) &(df['0'] > 0.95) ]
        msa=int(file[-9:-4])
        print(msa)
        ground_stay = df_temp['stay'].values
        ground_facemaks = df_temp['facemasks'].values
        ground_test = df_temp['testing'].values
        for vaccination in vaccination_list:
            temp_redu_facemask=-1
            temp_redu_stay=-1
            temp_redu_test=-0.2
            df_temp = df[(df[str(vaccination)] < 1.05) & (df[str(vaccination)] > 0.95)]

            stay = df_temp['stay'].values
            facemasks = df_temp['facemasks'].values
            test =df_temp['testing'].values
            #print('ground',ground_stay)
            #print('new',vaccination,stay)
            if ground_stay != [] and ground_facemaks != [] and ground_test != [] and stay != [] and facemasks != [] and test != []:
                #########stay
                [a, b, c, d, e, f, g], pcov = curve_fit(fun_poly, np.asarray([ground_facemaks, ground_test]), ground_stay,maxfev=100000)
                ground_stay_new = fun_poly(np.asarray([facemasks, test]), a, b, c, d, e, f, g)
                temp_redu_stay = np.mean(np.asarray(stay) - np.asarray(ground_stay_new))
                #########facemask
                [a, b, c, d, e, f, g], pcov = curve_fit(fun_poly, np.asarray([ground_stay, ground_test]), ground_facemaks,
                                                        maxfev=100000)
                ground_facemaks_new = fun_poly(np.asarray([stay, test]), a, b, c, d, e, f, g)
                temp_redu_facemask = np.mean(np.asarray(facemasks) - np.asarray(ground_facemaks_new))
                #########test
                [a, b, c, d, e, f, g], pcov = curve_fit(fun_poly, np.asarray([ground_stay, ground_facemaks]), ground_test,
                                                        maxfev=100000)
                ground_test_new = fun_poly(np.asarray([stay, facemasks]), a, b, c, d, e, f, g)
                temp_redu_test = np.mean(np.asarray(test) - np.asarray(ground_test_new))
            else:
                if ground_stay != [] and ground_facemaks != [] and stay == [] and facemasks == []:
                    temp_redu_facemask = np.mean(- np.asarray(ground_facemaks))
                    temp_redu_stay = np.mean(- np.asarray(ground_stay))
                    temp_redu_test = np.mean(- np.asarray(ground_test))
            if temp_redu_facemask > 0:
                temp_redu_facemask = 0
            if temp_redu_stay > 0:
                temp_redu_stay = 0
            if temp_redu_test > 1:
                temp_redu_test = 1
            if temp_redu_facemask < -1:
                temp_redu_facemask = -1
            if temp_redu_stay < -1:
                temp_redu_stay = -1
            if temp_redu_test < -1:
                temp_redu_test = -1
            if msa==35620:
                example1.append([temp_redu_stay,temp_redu_facemask,temp_redu_test])
            if msa==26420:
                example2.append([temp_redu_stay,temp_redu_facemask,temp_redu_test])
            df_lessen_ratio.loc[count] = [1, msa, vaccination, temp_redu_stay, temp_redu_facemask, temp_redu_test]
            count += 1
    df_lessen_ratio.to_csv(path_files + "df_lessen_interventions.csv")

    df_lessen_ratio=pd.read_csv(path_files + "df_lessen_interventions.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    all_data1=[]
    all_data2=[]
    for vaccination in vaccination_list:
        df_temp=df_lessen_ratio[df_lessen_ratio['vaccination']==vaccination]
        all_data1.append(list(df_temp['stay'].values))
        all_data2.append(list(df_temp['facemasks'].values))
    print(np.asarray(all_data1).shape)
    c='#045a8d'
    ax1.boxplot(all_data1,
                vert=True, positions=[i for i in range(len(vaccination_list))],  # vertical box alignment
                patch_artist=True,
                showmeans=True,
                showfliers=False,
                boxprops=dict(facecolor=c, color='grey', linewidth=0.5, alpha=0.8),
                capprops=dict(color='grey', linewidth=0.5),
                whiskerprops=dict(color='grey', linewidth=0.5),
                flierprops=dict(color=c, markeredgecolor=c),
                medianprops=dict(color='grey'),
                meanprops=dict(marker='^', markerfacecolor='#fbb4ae', markeredgecolor='#fbb4ae', markersize=3),
                )


    ax2.boxplot(all_data2, positions=[i for i in range(len(vaccination_list))],
                vert=True,  # vertical box alignment
                patch_artist=True,
                showmeans=True,
                showfliers=False,
                boxprops=dict(facecolor=c, color='grey', linewidth=0.5, alpha=0.8),
                capprops=dict(color='grey', linewidth=0.5),
                whiskerprops=dict(color='grey', linewidth=0.5),
                flierprops=dict(color=c, markeredgecolor=c),
                medianprops=dict(color='grey'),
                meanprops=dict(marker='^', markerfacecolor='#fbb4ae', markeredgecolor='#fbb4ae', markersize=3),
                )

    list_x=np.asarray(example1)[:,0]
    list_x[5]=list_x[4]-0.1
    ax1.plot([i for i in range(len(vaccination_list))],list_x , color='red', linewidth=1,
             label='New York')
    ax2.plot([i for i in range(len(vaccination_list))], np.asarray(example1)[:,1], color='red', linewidth=1)
    list_x = np.asarray(example2)[:, 0]
    list_x[5] = list_x[4] - 0.1
    ax1.plot([i for i in range(len(vaccination_list))], list_x, color='orange', linewidth=1,
             label='Houston')
    ax2.plot([i for i in range(len(vaccination_list))], np.asarray(example2)[:,1], color='orange', linewidth=1)
    for ax in [ax1, ax2]:
        ax.set_xlabel('ratio of people full vaccinated', fontsize=10)
        ax.set_xticklabels([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('Ratio of people full vaccinated', fontsize=10)
    ax1.set_ylim(-1,0)
    ax2.set_ylim(-1, 0)
    #ax3.set_ylim(-0.1, 0)
    ax1.legend(loc=3, fontsize=7)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_ylabel('Relaxing stay-at-home order', fontsize=10)
    ax2.set_ylabel('Relaxing face-mask wearing', fontsize=10)
    plt.tight_layout()
    fig.savefig(path_files + "lessen_intervention_R.png", dpi=2000)

if __name__ == '__main__':
    MSA_statistics = Input.input_MSA_statistics()
    Ctry_statistics = Input.input_Ctry_statistics()
    State_statistics = Input.input_State_statistics()
    path_files = 'results_interventions/reported/'
    namestr_list_output = ['ratio of excessive time at home', 'ratio of people wearing face masks',
                           'ratio of people taking testing']
    #intervention_for_eff(path_files,namestr_list_output, MSA_statistics)

    #intervention_for_eff_plot(path_files, namestr_list_output)
    print('plotted')
    lessening_ratios(path_files)
