"""Verify the implementation of the diffusion equation."""
from input_library import *
from Input import *
import sys
sys.path.append('/Users/luzhong/Documents/pythonCode/PDE-COVID')
start_date = datetime.datetime.strptime('2/01/20', '%m/%d/%y')
end_date = datetime.datetime.strptime('02/20/21', '%m/%d/%y')
plot_start_date= datetime.datetime.strptime('4/01/20', '%m/%d/%y')

def SIR(t, y):
    S = y[0]
    I = y[1]
    R = y[2]
    # print(y[0],y[1],y[2])
    u= y[3]
    beta = y[4]
    gamma = y[5]
    return ([-beta * S * I+u-u*S, beta * S * I - gamma * I-u*I, gamma * I-u*R, 0, 0,0])


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



def SIRD_delay(initial_state,num_times,u, d, beta, gamma, delta):
    g = lambda t: array(initial_state)
    tt = linspace(0, num_times, num_times)

    def model(Y, t, u,d, beta, gamma, delta):
        S, I, R, D = Y(t)
        S_d, I_d, R_d, D_d = Y(t - d)
        return array([-beta * S * I_d + u - u * S, beta * S * I_d - gamma * I - delta * I - u * I, gamma * I - u * R, delta * I])

    sol = ddeint(model, g, tt, fargs=(u, d, beta, gamma, delta))
    return [sol[:, 0],sol[:, 1],sol[:, 2],sol[:, 3]]

def learn_parameter_MSA(df_empty, MSA_statistics, MSA_list,path_results,path_files,type):
    select_dyas = [0,1,2,3,4,5,6]
    MSA_parameter = dict()

    if type=='reported':
        start_date = datetime.datetime.strptime('04/01/20', '%m/%d/%y')  ##end_date for parameter_learning
    if type=='sero':
        start_date = datetime.datetime.strptime('04/12/20', '%m/%d/%y')  ##end_date for parameter_learning
    end_date = datetime.datetime.strptime('10/01/20', '%m/%d/%y')
    for msa, ctry_list in MSA_statistics['county'].items():
        df_empty_temp=df_empty[df_empty['MSA_code']==int(msa)]
        #print(msa,MSA_statistics['name'][msa])
        if len(df_empty_temp)!=0:
            time_list = [datetime.datetime.strptime(str(i), '%Y-%m-%d') for i in df_empty_temp['time']]
            S_true = dict(zip(time_list, df_empty_temp['Sd']))
            I_true = dict(zip(time_list, df_empty_temp['Id']))
            R_true = dict(zip(time_list, df_empty_temp['Rd']))
            D_true = dict(zip(time_list, df_empty_temp['Dd']))
            u = MSA_statistics['birth_death_rate'][msa]
            #start_date=datetime.datetime.strptime(list(df_empty_temp['start_date'])[0],'%Y-%m-%d')
            #end_date=start_date+datetime.timedelta(days=7)
            sum_pop = MSA_statistics['pop'][msa]
            datelist_temp = [t for t in time_list if t >= start_date and t<=end_date]
            num_times = len(datelist_temp)

            datelist_str = [t.strftime('%b-%d,%Y')[0:6] for t in datelist_temp]
            dataI_percent = [float(value / sum_pop) for key, value in I_true.items() if key in datelist_temp]
            dataR_percent = [float(value / sum_pop) for key, value in R_true.items() if key in datelist_temp]
            dataD_percent = [float(value / sum_pop) for key, value in D_true.items() if key in datelist_temp]

            initial_state = [1 - (I_true[start_date] + R_true[start_date] + D_true[start_date]) / sum_pop,I_true[start_date] / sum_pop, R_true[start_date] / sum_pop, D_true[start_date] / sum_pop]

            def sumsq(p):
                beta, gamma, delta = p

                def SIRD(t, y):
                    S = y[0]
                    I = y[1]
                    R = y[2]
                    D = y[3]
                    return ([-beta * S * I + u - u * S, beta * S * I - gamma * I - delta * I - u * I, gamma * I - u * R,
                             delta * I])

                sol = solve_ivp(SIRD, [0, num_times - 1], initial_state, t_eval=np.arange(0, num_times - 1 + 0.2, 0.2))
                # print(sol.y[1][::5])
                return (sum((sol.y[1][::5] - dataI_percent) ** 2) + sum((sol.y[2][::5] - dataR_percent) ** 2) + sum(
                    (sol.y[3][::5] - dataD_percent) ** 2))

            msol = minimize(sumsq, [0.1, 0.03, 0.003], method='Nelder-Mead')

            beta, gamma, delta = msol.x
            if beta<0:
                beta=0
            if gamma < 0:
                gamma = 0
            if delta<0:
                delta=0
            print(msa, MSA_statistics['name'][msa],u, beta, gamma, delta)
            MSA_parameter[msa] = [u, beta, gamma, delta]
            sol = solve_ivp(SIRD, [0, num_times - 1], initial_state + [u, beta, gamma, delta],
                            t_eval=np.arange(0, num_times - 1 + 0.2, 0.2))
            if msa==35620:
                output_fitting(MSA_statistics['name'][msa], sol, sum_pop, datelist_str, dataI_percent, dataR_percent, dataD_percent, u,beta, gamma, delta,path_results)
    print('finished')
    f_save = open(path_files+'MSA_ode_parameter.pkl', 'wb')
    pickle.dump(MSA_parameter, f_save)
    f_save.close()

def learn_parameter_US(df_empty, MSA_statistics):
    select_dyas=[0,1,2,3,4,5,6]
    US_parameter = dict()
    start_date = datetime.datetime.strptime('04/12/20', '%m/%d/%y')##end_date for parameter_learning
    time_list=[datetime.datetime.strptime(str(i), '%Y-%m-%d') for i in df_empty['time']]
    S_true = dict(zip(time_list, df_empty['Sd']))
    I_true = dict(zip(time_list, df_empty['Id']))
    R_true = dict(zip(time_list, df_empty['Rd']))
    D_true = dict(zip(time_list, df_empty['Dd']))
    u_list = list(MSA_statistics['birth_death_rate'].values())
    u=np.mean([0 if math.isnan(x) else x for x in u_list])
    sum_pop=S_true[datetime.datetime.strptime('01/22/20', '%m/%d/%y')]
    datelist_temp = [t for t in time_list if t>=start_date and (t).weekday() in select_dyas]
    num_times=len(datelist_temp)

    datelist_str = [t.strftime('%b-%d,%Y')[0:6] for t in datelist_temp]
    print(datelist_str)
    dataI_percent = [float(value / sum_pop) for key, value in I_true.items() if key >= start_date and (key).weekday() in select_dyas]
    dataR_percent = [float(value / sum_pop) for key, value in R_true.items() if key >= start_date and (key).weekday() in select_dyas]
    dataD_percent = [float(value / sum_pop) for key, value in D_true.items() if key >= start_date and (key).weekday() in select_dyas]

    initial_state=[1- (I_true[start_date]+R_true[start_date]+D_true[start_date])/ sum_pop, I_true[start_date] / sum_pop, R_true[start_date] / sum_pop,D_true[start_date] / sum_pop]
    print(initial_state,num_times)

    def sumsq(p):
        beta, gamma,delta = p
        def SIRD(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            D = y[3]
            return ([-beta * S * I+u-u*S, beta * S * I - gamma * I-delta*I-u*I, gamma * I-u*R, delta*I])

        sol = solve_ivp(SIRD, [0, num_times - 1], initial_state,t_eval=np.arange(0, num_times - 1 + 0.2, 0.2))
        # print(sol.y[1][::5])
        return (sum((sol.y[1][::5] - dataI_percent) ** 2) + sum((sol.y[2][::5] - dataR_percent) ** 2)+ sum((sol.y[3][::5] - dataD_percent) ** 2))

    msol = minimize(sumsq, [0.1, 0.03,0.003], method='Nelder-Mead')

    beta, gamma, delta = msol.x
    for temp_para in [beta, gamma, delta]:
        if temp_para < 0:
            temp_para = 0
    print(u, beta, gamma, delta)
    US_parameter['US'] = [u, beta, gamma, delta]
    sol = solve_ivp(SIRD, [0, num_times - 1], initial_state+[u, beta, gamma, delta], t_eval=np.arange(0, num_times - 1 + 0.2, 0.2))
    output_fitting('United States', sol, sum_pop, datelist_str, dataI_percent, dataR_percent, dataD_percent, u,beta, gamma,delta)
    f_save = open('results_trajectory_fitting/US_ode_parameter.pkl', 'wb')
    pickle.dump(US_parameter, f_save)
    f_save.close()

def output_fitting(namestr,sol,sum_pop,datelist_str,I_true,R_true,D_true,u,beta,gamma,delta,path_results):
    interval_day=7
    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(111)
    plt.plot(sol.t, [i * sum_pop for i in sol.y[1]],color='#e41a1c')
    plt.plot(sol.t,[i * sum_pop for i in sol.y[2]],color='#377eb8')
    plt.plot(sol.t,[i * sum_pop for i in sol.y[3]],color='#4daf4a')
    plt.plot(np.arange(0, len(datelist_str)),[value*sum_pop for value in I_true],
             "k*:",markersize=2)
    plt.plot(np.arange(0, len(datelist_str)),[value*sum_pop for value in R_true],
             "ks:",markersize=2)
    plt.plot(np.arange(0, len(datelist_str)), [value*sum_pop for value in D_true],
             "k>:",markersize=2)
    plt.legend(["Infected", "Recovered", "Dead", "Real Infected", "Real Recovered","Real Dead"],loc=1)

    ax.text(0, 1.05, namestr, transform=ax.transAxes)
    ax.text(0.1, 0.9, r'$\beta=%s$' % (round(beta, 4)), transform=ax.transAxes)
    ax.text(0.1, 0.85, r'$\gamma=%s$' % (round(gamma, 4)), transform=ax.transAxes)
    ax.text(0.1, 0.8, r'$\delta=%s$' % (round(delta, 4)), transform=ax.transAxes)
    ax.text(0.1, 0.75, r'$\mu=%s$' % (round(u, 4)), transform=ax.transAxes)
    ax.set_xticks(np.arange(0, len(datelist_str), interval_day))
    ax.set_xticklabels([datelist_str[i * interval_day] for i in range(0, len(np.arange(0, len(datelist_str), interval_day)))], fontsize=6,
                       rotation=45)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.savefig(path_results+"parameters/infections(ODE)_fitted" + namestr+ ".png", dpi=600)







def trajectory_controller_two_controller_MSA(df_empty, MSA_statistics, MSA_list,path_results,path_files):
    f_read = open(path_files+'MSA_ode_parameter.pkl', 'rb')
    mas_prameters = pickle.load(f_read)
    f_read.close()
    df_controlled = pd.DataFrame(columns=['MSA_code', 'time', 'S', 'I', 'R','D'])
    df_controlled_parameter = pd.DataFrame(columns=['MSA_code','MSA_name','time', 'beta(t)', 'gamma(t)', 'delta(t)', 'R_eff(t)'])
    count_p=0
    count=0
    for msa, ctry_list in MSA_statistics['county'].items():
        if msa in MSA_list:
            df_empty_temp=df_empty[df_empty['MSA_code']==msa]
            start_date_temp=[]
            if len(list(df_empty_temp['start_date'].values))!=0:
                start_date_temp=datetime.datetime.strptime(df_empty_temp['start_date'].values[0], '%Y-%m-%d')
                if start_date_temp<start_date:
                    start_date_temp=start_date
            if len(df_empty_temp)!=0 and np.max(df_empty_temp['Sd'])>1000 and start_date_temp!=[]:
                print(msa,MSA_statistics['name'][msa],start_date_temp)
                [u0, beta0, gamma0, delta0] = mas_prameters[msa]
                time_list = [datetime.datetime.strptime(str(i), '%Y-%m-%d') for i in df_empty_temp['time']]
                S_true = dict(zip(time_list, df_empty_temp['Sd']))
                I_true = dict(zip(time_list, df_empty_temp['Id']))
                R_true = dict(zip(time_list, df_empty_temp['Rd']))
                D_true = dict(zip(time_list, df_empty_temp['Dd']))
                sum_pop =  MSA_statistics['pop'][msa]
                datelist_temp = [t for t in time_list if t >= start_date and t<=end_date]
                datelist_temp = datelist_temp[0:len(datelist_temp)]
                #print(datelist_temp)
                num_times = len(datelist_temp)
                datelist_str = [t.strftime('%b-%d,%Y')[0:6] for t in datelist_temp]
                #print(u0, beta0, gamma0, delta0,sum_pop)
                dataS_percent = [float(value / sum_pop) for key, value in S_true.items() if key in datelist_temp]
                dataI_percent = [float(value / sum_pop) for key, value in I_true.items() if key in datelist_temp]
                dataR_percent = [float(value / sum_pop) for key, value in R_true.items() if key in datelist_temp]
                dataD_percent = [float(value / sum_pop) for key, value in D_true.items() if key in datelist_temp]

                new_para_dict = defaultdict()
                new_para_dict['gamma'] = dict();
                new_para_dict['beta'] = dict();
                new_para_dict['delta'] = dict();
                new_para_dict['R_eff'] = dict();
                #print([S_0, I_0, R_0, D_0])

                [u_new, beta_new, gamma_new, delta_new] = [u0, beta0, gamma0, delta0]

                S_dict = dict();
                I_dict = dict();
                R_dict = dict();
                D_dict = dict()
                for t in datelist_temp:
                    if t==start_date_temp:

                        [S_new, I_new, R_new, D_new]= [1 - (I_true[start_date_temp] + R_true[start_date_temp] + D_true[start_date_temp]) / sum_pop,
                            I_true[start_date_temp] / sum_pop, R_true[start_date_temp] / sum_pop,D_true[start_date_temp] / sum_pop]
                        print(t, [S_new, I_new, R_new, D_new])
                    if t >= start_date_temp:
                        tx = t + datetime.timedelta(days=1)
                        [S, I, R, D] = [S_new, I_new, R_new, D_new]

                        S_dict[t] = S_new;
                        I_dict[t] = I_new;
                        R_dict[t] = R_new;
                        D_dict[t] = D_new;

                        [u,beta, gamma, delta] = [u_new,beta_new, gamma_new, delta_new]

                        sol = solve_ivp(SIRD, [0, 1], [S, I, R, D] + [u, beta, gamma, delta], t_eval=np.arange(0, 1 + 0.2, 0.2))

                        [S_new, I_new, R_new, D_new] = [sol.y[0][5], sol.y[1][5], sol.y[2][5], sol.y[3][5]]

                        [beta_new, gamma_new, delta_new] = controller_caculation(sum_pop, t, tx, S_true, I_true, R_true, D_true,
                                                                                 S_dict, I_dict, R_dict, D_dict, u0, beta0,
                                                                                 gamma0, delta0)
                        #print(t, I_new*sum_pop, I_true[t],R_new*sum_pop,R_true[t],D_new*sum_pop,D_true[t],S_new*sum_pop,S_true[t])
                        #print(t, round(beta_new, 4), round(gamma_new, 4), round(delta_new, 4))
                        new_para_dict['beta'][t] = beta_new
                        new_para_dict['gamma'][t] = gamma_new
                        new_para_dict['delta'][t] = delta_new
                        new_para_dict['R_eff'][t]=(beta_new*S_true[tx]/sum_pop)/(gamma_new+delta_new+u)
                        df_controlled_parameter.loc[count_p]=[msa,MSA_statistics['name'][msa],t,new_para_dict['beta'][t],new_para_dict['gamma'][t],new_para_dict['delta'][t],new_para_dict['R_eff'][t]  ]
                        count_p+=1
                    else:
                        new_para_dict['beta'][t] = 0
                        new_para_dict['gamma'][t] = 0
                        new_para_dict['delta'][t] = 0
                        new_para_dict['R_eff'][t] = 0
                        df_controlled_parameter.loc[count_p] = [msa, MSA_statistics['name'][msa], t,
                                                                new_para_dict['beta'][t], new_para_dict['gamma'][t],
                                                                new_para_dict['delta'][t], new_para_dict['R_eff'][t]]
                        count_p += 1

                for t in datelist_temp:
                    if t==start_date_temp:
                        [S_new, I_new, R_new, D_new]= [1 - (I_true[start_date_temp] + R_true[start_date_temp] + D_true[start_date_temp]) / sum_pop,
                            I_true[start_date_temp] / sum_pop, R_true[start_date_temp] / sum_pop,D_true[start_date_temp] / sum_pop]
                    if t >= start_date_temp:
                        #print([msa,t, S_new * sum_pop, I_new * sum_pop, R_new * sum_pop, D_new * sum_pop])
                        df_controlled.loc[count] = [msa,t, S_new * sum_pop, I_new * sum_pop, R_new * sum_pop, D_new * sum_pop]
                        count += 1
                        [u, beta, gamma, delta] = [u, new_para_dict['beta'][t], new_para_dict['gamma'][t],
                                                   new_para_dict['delta'][t]]
                        sol = solve_ivp(SIRD, [0, 1], [S_new, I_new, R_new, D_new] + [u, beta, gamma, delta],
                                        t_eval=np.arange(0, 1 + 0.2, 0.2))
                        [S_new, I_new, R_new, D_new] = [sol.y[0][5], sol.y[1][5], sol.y[2][5], sol.y[3][5]]
                    else:
                        df_controlled.loc[count] = [msa, t, 1 * sum_pop, 0,0,0]
                        count += 1
                f_save = open(path_files+'parameters/infections(ODE)_controller_temporal' + MSA_statistics['name'][msa] + '.pkl', 'wb')
                pickle.dump(new_para_dict, f_save)
                f_save.close()

                if msa in [10500,35620, 31080, 16980, 19100, 26420, 47900, 33100, 37980, 12060, 38060, 14460, 41860]:
                    df_x=df_controlled[df_controlled['MSA_code']==msa]
                    output_controllers(df_x,  MSA_statistics['name'][msa] , sum_pop, datelist_str,dataI_percent, dataR_percent,
                                       dataD_percent, u, new_para_dict,path_results)

    df_controlled.to_csv(path_files + 'MSAs-ODE-tracking-two.csv')
    df_controlled_parameter.to_csv(path_files+'MSAs-ODE-tracking-parameters.csv')

def controller_caculation(sum_pop,t,tx,S_d, I_d, R_d,D_d,S, I, R,D,u0, beta0, gamma0,delta0):
    k1 = 0.1
    k2 = 0.1
    k3 = 0.1
    if I[t]!=0:
        temp_gamma=(1/I[t])*((R_d[tx]/sum_pop-R_d[t]/sum_pop)+u0*R[t]-gamma0*I[t]-k2*(R[t]-R_d[t]/sum_pop))+gamma0
        #print((1 / (I[t])), (R_d[tx]/sum_pop-R_d[t]/sum_pop), u0*R[t]-gamma0*I[t]-k2*(R[t]-R_d[t]/sum_pop))
        if temp_gamma<0:
            temp_gamma=0
        temp_delta = (1 / I[t]) * ((D_d[tx]/sum_pop - D_d[t]/sum_pop) -delta0 * I[t] - k3 * (D[t]-D_d[t]/sum_pop))+delta0
        #print((1 / (I[t])), (D_d[tx]/sum_pop - D_d[t]/sum_pop),  -delta0 * I[t] - k3 * (D[t]-D_d[t]/sum_pop))
        if temp_delta < 0:
            temp_delta = 0
        temp_beta=(1/(I[t]*S[t]))*((I_d[tx]/sum_pop-I_d[t]/sum_pop)+u0*I[t]+temp_gamma*I[t]+temp_delta*I[t]-k1*(I[t]-I_d[t]/sum_pop))-beta0+beta0
        if temp_beta < 0:
            temp_beta = 0

        if temp_gamma>1:
            temp_gamma=1
        if temp_delta>1:
            temp_delta=1
        if temp_beta>2:
            temp_beta=2

    else:
        temp_gamma = 0;temp_beta = 0;temp_delta = 0;
    #print(1/(I[t]*S[t]),(I_d[tx]/sum_pop-I_d[t]/sum_pop),u0*I[t]+temp_gamma*I[t]+temp_delta*I[t]-k1*(I[t]-I_d[t]/sum_pop))


    return temp_beta, temp_gamma,temp_delta

def trajectory_controller_two_controller_US(df_empty, MSA_statistics):
    select_dyas=[0,1,2,3,4,5,6]
    start_date = datetime.datetime.strptime('3/10/20', '%m/%d/%y')
    end_date= datetime.datetime.strptime('12/20/20', '%m/%d/%y')
    f_read = open('results_trajectory_fitting/US_ode_parameter.pkl', 'rb')
    mas_prameters = pickle.load(f_read)
    f_read.close()
    df_controlled = pd.DataFrame(columns=[ 'time', 'S', 'I', 'R','D'])
    count = 0
    [u0, beta0, gamma0,delta0] = mas_prameters['US']
    time_list = [datetime.datetime.strptime(str(i), '%Y-%m-%d') for i in df_empty['time']]
    S_true = dict(zip(time_list, df_empty['Sd']))
    I_true = dict(zip(time_list, df_empty['Id']))
    R_true = dict(zip(time_list, df_empty['Rd']))
    D_true = dict(zip(time_list, df_empty['Dd']))
    u_list = list(MSA_statistics['birth_death_rate'].values())
    u = np.mean([0 if math.isnan(x) else x for x in u_list])
    sum_pop = S_true[datetime.datetime.strptime('01/22/20', '%m/%d/%y')]
    datelist_temp = [t for t in time_list if t >= start_date and  t <= end_date]
    datelist_temp=datelist_temp[0:len(datelist_temp)-1]
    #print(datelist_temp)
    num_times = len(datelist_temp)
    datelist_str = [t.strftime('%b-%d,%Y')[0:6] for t in datelist_temp]
    print(datelist_str)

    dataI_percent = [float(value / sum_pop) for key, value in I_true.items() if key in datelist_temp]
    dataR_percent = [float(value / sum_pop) for key, value in R_true.items() if key in datelist_temp]
    dataD_percent = [float(value / sum_pop) for key, value in D_true.items() if key in datelist_temp]


    [S_0, I_0, R_0,D_0] = [1 - (I_true[start_date] + R_true[start_date] + D_true[start_date]) / sum_pop,
                     I_true[start_date] / sum_pop, R_true[start_date] / sum_pop, D_true[start_date] / sum_pop]
    new_para_dict = defaultdict()
    new_para_dict['gamma']=dict();new_para_dict['beta']=dict();new_para_dict['delta']=dict();

    [u_new,beta_new,gamma_new,delta_new]=[u0, beta0, gamma0,delta0]
    [S_new, I_new, R_new, D_new]=[S_0, I_0, R_0,D_0]
    S_dict=dict();I_dict=dict();R_dict=dict();D_dict=dict()
    for t in datelist_temp:
        tx = t + datetime.timedelta(days=1)
        [S,I,R,D]=[S_new, I_new, R_new,D_new]
        #print(t,S_new, I_new, R_new,D_new)
        S_dict[t]=S_new;I_dict[t]=I_new;R_dict[t]=R_new;D_dict[t]=D_new;

        [beta, gamma, delta]=[beta_new,gamma_new,delta_new]
        sol = solve_ivp(SIRD, [0, 1], [S,I,R,D]+[u,beta,gamma,delta],t_eval=np.arange(0, 1 + 0.2, 0.2))

        [S_new, I_new, R_new,D_new] = [sol.y[0][5], sol.y[1][5], sol.y[2][5],sol.y[3][5]]

        [beta_new,gamma_new,delta_new]=controller_caculation(sum_pop,t,tx,S_true, I_true, R_true,D_true,S_dict,I_dict,R_dict,D_dict,u0, beta0, gamma0,delta0)
        #print(t,round(beta_new, 4),round(gamma_new, 4),round(delta_new, 4))
        new_para_dict['beta'][t]=beta_new
        new_para_dict['gamma'][t] = gamma_new
        new_para_dict['delta'][t] = delta_new

    [S_new, I_new, R_new, D_new] = [S_0, I_0, R_0, D_0]
    count=0
    for t in datelist_temp:
        df_controlled.loc[count] = [ t, S_new * sum_pop, I_new * sum_pop, R_new * sum_pop,D_new * sum_pop]
        count += 1
        [u,beta,gamma, delta] = [u,new_para_dict['beta'][t],new_para_dict['gamma'][t],new_para_dict['delta'][t]]
        # print(t, beta, gamma, (I_0 * sum_pop))
        sol = solve_ivp(SIRD, [0, 1], [S_new, I_new, R_new, D_new]+ [u,beta,gamma, delta], t_eval=np.arange(0, 1 + 0.2, 0.2))
        [S_new, I_new, R_new, D_new]=[sol.y[0][5],sol.y[1][5],sol.y[2][5],sol.y[3][5]]

    RMSD_I = np.mean([math.pow(df_controlled['I'].values[i] - list(I_true.values())[i], 2) for i in range(num_times)])
    RMSD_I = math.pow(RMSD_I, 0.5)
    RMSD_R = np.mean([math.pow(df_controlled['R'].values[i] - list(I_true.values())[i], 2) for i in range(num_times)])
    RMSD_R = math.pow(RMSD_R, 0.5)
    RMSD_D = np.mean([math.pow(df_controlled['D'].values[i] - list(I_true.values())[i], 2) for i in range(num_times)])
    RMSD_D = math.pow(RMSD_R, 0.5)
    f_save = open('results_trajectory_fitting/two-controllers/infections(ODE)_controller_temporal' + 'US'+ '.pkl', 'wb')
    pickle.dump(new_para_dict, f_save)
    f_save.close()
    df_controlled.to_csv('results_trajectory_fitting/USA-ODE-tracking-two.csv')
    output_controllers(df_controlled, 'United States', sum_pop, datelist_str, dataI_percent, dataR_percent, dataD_percent, u, new_para_dict, RMSD_I, RMSD_R,
                       RMSD_D)



def output_controllers(df_temp,namestr,sum_pop,datelist_str,I_true,R_true,D_true,u,para_dict,path_results):
    beginingcount=0
    interval_day=30
    xaxis=[i for i in range(len(datelist_str))]
    fig, ax1 = plt.subplots(1, 1,figsize=(6, 4))
    color_dict = {'beta(t)': '#d7191c', 'gamma(t)': '#e66101', 'delta(t)': '#0571b0', 'R_eff(t)': '#5e3c99'}
    ax1.plot(xaxis,df_temp['I'].values,color='#d7191c')
    ax1.plot(xaxis, df_temp['R'].values,color='#e66101')
    ax1.plot(xaxis, df_temp['D'].values,color='#0571b0')
    ax1.plot(xaxis, [value*sum_pop for value in I_true], "o",
             markersize=2, markerfacecolor='#d7191c', markeredgewidth=0, alpha=0.5, color='#d7191c')
    ax1.plot(xaxis, [value*sum_pop for value in R_true], "o",
             markersize=2, markerfacecolor='#e66101', markeredgewidth=0, alpha=0.5, color='#e66101')
    ax1.plot(xaxis, [value * sum_pop for value in D_true], "o",
             markersize=2, markerfacecolor='#0571b0', markeredgewidth=0, alpha=0.5, color='#0571b0')

    ax1.legend(["Fitted Infected Cases", "Fitted Recovered Cases", "Fitted Dead Cases","Reported Infected Cases", "Reported Recovered Cases", "Reported Dead Cases"], loc=2)
    ax1.text(0, 1.05, namestr, transform=ax1.transAxes)
    '''
    ax1.text(0.3, 0.9, 'RMSD (I) = %.2f \n' % round(RMSD_I) + 'RMSD (R) =%.2f \n' % round(RMSD_R)+'RMSD (D) =%.2f' % round(RMSD_D),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax1.transAxes, fontsize=7)
    '''

    ax1.set_ylabel("Cases")
    ax1.set_xticks(np.arange(0, len(datelist_str), interval_day))
    ax1.set_xticklabels([datelist_str[i * interval_day] for i in range(0, len(np.arange(0, len(datelist_str), interval_day)))],fontsize=6,rotation=45)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax1.set_xlim(beginingcount,len(xaxis))
    plt.tight_layout()

    plt.savefig(path_results + "infections(ODE)_Controllers" + namestr + "(fitting results).png",
                dpi=600)
    plt.close()

    fig, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    ax2.plot(xaxis, [value for key, value in para_dict['beta'].items()], ":", color='#e41a1c', alpha=0.5)
    ax2.plot(xaxis, [value for key, value in para_dict['gamma'].items()], ":", color='#377eb8', alpha=0.5)
    ax2.plot(xaxis, [value for key, value in para_dict['delta'].items()], ":", color='#4daf4a', alpha=0.5)
    ax2.plot(xaxis, moving_average([value for key, value in para_dict['beta'].items()], 7), "-", color='#e41a1c',
             alpha=0.5)
    ax2.plot(xaxis, moving_average([value for key, value in para_dict['gamma'].items()], 7), "-", color='#377eb8',
             alpha=0.5)
    ax2.plot(xaxis, moving_average([value for key, value in para_dict['delta'].items()], 7), "-", color='#4daf4a',
             alpha=0.5)

    ax2.legend([r"$\beta(t)$", r"$\gamma(t)$", r"$\delta(t)$"])
    ax2.text(0, 1.05, namestr, transform=ax2.transAxes)
    ax2.set_xticks(np.arange(0, len(datelist_str), interval_day))
    ax2.set_xticklabels([datelist_str[i * interval_day] for i in range(0, len(np.arange(0, len(datelist_str), interval_day)))], fontsize=6,
                        rotation=45)
    ax2.set_yscale('log')
    ax2.set_xlim(beginingcount, len(xaxis))
    ax2.set_ylim((pow(10, -5), pow(10, 0)))
    plt.tight_layout()

    plt.savefig(path_results + "infections(ODE)_Controllers" + namestr + "(parameters).png",
                dpi=600)
    plt.close()

    R_effecitve=[value for key,value in para_dict['R_eff'].items()]
    fig, ax3 = plt.subplots(1, 1, figsize=(6, 4))
    ax3.plot(xaxis, R_effecitve, ":", color='blue', alpha=0.5, linewidth=1)
    ax3.plot(xaxis, moving_average(R_effecitve,7), "-", color='blue', alpha=0.5)

    ax3.legend([r"$R_0(t)$"])
    ax3.text(0, 1.05, namestr, transform=ax3.transAxes)
    ax3.set_xticks(np.arange(0, len(datelist_str), interval_day))
    ax3.set_xticklabels([datelist_str[i * interval_day] for i in range(0, len(np.arange(0, len(datelist_str), interval_day)))], fontsize=6,
                       rotation=45)
    ax3.axhline(y=1, linestyle='-', linewidth=1, color='grey')
    ax3.axhline(y=2, linestyle='-', linewidth=1, color='grey')
    ax3.axhline(y=3, linestyle='-', linewidth=1, color='grey')
    ax3.set_yscale('log')
    ax3.set_ylim((pow(10, -0.5), pow(10, 1.5)))
    ax3.set_xlim(beginingcount, len(xaxis))
    plt.tight_layout()

    plt.savefig(path_results+"infections(ODE)_Controllers" + namestr+ "(R_effective).png",
                dpi=600)
    plt.close()



def moving_average(list_example, n) :
    new_list=[]
    for i in range(len(list_example)):
        if i<=n:
            new_list.append(np.mean([list_example[j] for j in range(0,i+1)]))
        else:
            new_list.append(np.mean([list_example[j] for j in range(i-n, i + 1)]))

    return new_list


if __name__ == '__main__':
    ########input=========================================
    case='MSAs_reported'
    case='MSAs_sero'
    MSA_list = [35620, 31080, 16980, 19100, 26420, 47900, 33100, 37980, 12060, 38060, 14460, 41860, 40140, 19820]
    MSA_statistics = input_MSA_statistics()

    ########begin=============================================
    if case=='MSAs_reported':#e31a1c
        df_empty = pd.read_csv('dataset/MSA_S_I_R_D.csv')
        path_results='results_trajectory_fitting/reported/'
        path_files= 'results_trajectory_fitting/reported/'
        MSA_list = list(MSA_statistics['county'].keys())
        #learn_parameter_MSA(df_empty, MSA_statistics, MSA_list,path_results,path_files,'reported')
        print("trajectory_controller_two_controller_MSA")
        trajectory_controller_two_controller_MSA(df_empty, MSA_statistics,MSA_list,path_results,path_files)

    if case == 'MSAs_sero':  # e31a1c
        df_empty = pd.read_csv('dataset/MSA_S_I_R_D_SERO.csv')
        path_results = 'results_trajectory_fitting/Sero/'
        path_files = 'results_trajectory_fitting/Sero/'
        MSA_list = list(MSA_statistics['county'].keys())
        learn_parameter_MSA(df_empty, MSA_statistics, MSA_list,path_results,path_files,'sero')
        print("trajectory_controller_two_controller_MSA")
        trajectory_controller_two_controller_MSA(df_empty, MSA_statistics, MSA_list, path_results, path_files)











