import sys

sys.path.append('/Users/luzhong/Documents/pythonCode/PDE-COVID')
from input_library import *

path_external = '/Volumes/SeagateDrive 1/US-mobility/'

global date_check
date_check = ['2021-01-01','2021-01-02','2021-01-03','2021-01-04','2021-01-05','2021-01-06',
              '2021-01-07','2021-01-08','2021-01-09','2021-01-10','2021-01-11']
f_save = open('Dataset/MSA_demographics.pkl', 'rb')
MSA_statistics = pickle.load(f_save)
f_save.close()

MSA_all = pandas.read_csv(path_external + 'covid19-intervention-data/MSA_summary_indicators.csv')
MSA_all = MSA_all[MSA_all['date'].isin(date_check)]


def func_beta(x, a, b, c):
    return (a * x[0] + 1) * (b * x[1] + c * x[2] + 1) - 1


def func_delta(x, a, b, c):
    return (a * x[0] + 1) * (b * x[1] + c * x[2] + 1) - 1


def vaccination_sigmoid(n, cap, n_max):
    if cap==0:
        y = [0 for i in range(n)] + [0 for i in range(n_max - n)]
    else:
        if cap==0.9:
            sigma_dict = {120: 6, 180: 5, 270: 3.5, 360: 2.8}
        if cap==0.8:
            sigma_dict = {120: 5.8, 180: 4.6, 270: 3.3, 360: 2.7}
        if cap==0.7:
            sigma_dict = {120: 5.4, 180: 4.3, 270: 3.1, 360: 2.6}
        if cap==0.6:
            sigma_dict = {120: 5.2, 180: 4.0, 270: 2.9, 360: 2.4}
        if cap==0.5:
            sigma_dict = {120: 5.1, 180: 3.9, 270: 2.8, 360: 2.2}
        if cap==0.4:
            sigma_dict = {120: 5.0, 180: 3.5, 270: 2.7, 360: 1.8}
        if cap==0.3:
            sigma_dict = {120: 4.5, 180: 3.0, 270: 2.0, 360: 1.0}
        if cap==0.2:
            sigma_dict = {120: 3.0, 180: 1.5, 270: 0.8, 360: 0.4}
        if cap==0.1:
            sigma_dict = {120: 0.5, 180: 0.1, 270: 0.1, 360: 0.1}

        result = trunced_normal_distribution(0, n, sigma_dict[n])
        y = [result[0]] + [result[i + 1] - result[i] for i in range(n - 1)]
        y = [i * cap for i in y] + [0 for i in range(n_max - n)]
    return y


def vaccination_linear(n, cap, n_max):
    each = cap / n
    y = [each for i in range(n)] + [0 for i in range(n_max - n)]
    return y


def SIRD(t, y):
    S = y[0]
    I = y[1]
    R = y[2]
    D = y[3]
    # print(y[0],y[1],y[2])
    u = y[4]
    beta = y[5]
    gamma = y[6]
    delta = y[7]
    return ([-beta * S * I + u * (S + I + R) - u * S, beta * S * I - gamma * I - delta * I - u * I, gamma * I - u * R,
             delta * I, 0, 0, 0, 0])


def zero_infection_days(list_temp, pop_sum, run_days):
    newlist = [i if i > 0 else 0 for i in list_temp]
    for i in range(0, run_days - 90):
        sum_infection = np.sum(newlist[i:i + 90])
        if sum_infection <= 1:
            return "true", i
            break
    return "false", run_days - 1


def scenario_test(period, run_days, case, vaccination_type, path_files):
    MSA_beta = pd.read_csv('results_interventions/reported/MSA_impact_coeffecient_beta(t).csv')
    MSA_delta = pd.read_csv('results_interventions/reported/MSA_impact_coeffecient_delta(t).csv')

    df_result = pd.DataFrame(
        columns=['vaccination', 'MSA_code', 'zero_new_infection', 'days', 'accumulated I', 'accumulated R',
                 'accumulated D', 'sum_pop'])
    count_x = 0
    for vaccination in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:  #
        print(vaccination)
        if vaccination_type == 'linear':
            vaccine_list = vaccination_linear(period, vaccination, run_days)
        if vaccination_type == 'sigmoid':
            vaccine_list = vaccination_sigmoid(period, vaccination, run_days)
        for msa in MSA_beta['MSA_code']:
            # if msa in [35620,31080, 16980, 19100, 26420, 47900, 33100, 37980, 12060, 38060, 14460, 41860, 40140,19820]:#
            MSA_all_temp = MSA_all[MSA_all['MSA_code'] == int(msa)]
            Sd = MSA_all_temp['S(t)'].values[-1]
            Id = MSA_all_temp['I(t)'].values[-1]
            Rd = MSA_all_temp['R(t)'].values[-1]
            Dd = MSA_all_temp['D(t)'].values[-1]
            sum_pop = (Sd + Id + Rd + Dd)
            u = MSA_statistics['birth_death_rate'][msa]
            beta_0 = MSA_all_temp['beta_0'].values[0]
            delta_0 = MSA_all_temp['delta_0'].values[0]
            gamma = np.mean(MSA_all_temp['gamma(t)'].values)
            beta = np.mean(MSA_all_temp['beta(t)'].values)
            delta = np.mean(MSA_all_temp['delta(t)'].values)
            a = MSA_beta[MSA_beta['MSA_code'] == msa]['ratio of excessive time at home'].values
            b = MSA_beta[MSA_beta['MSA_code'] == msa]['ratio of people wearing face masks'].values
            c = MSA_beta[MSA_beta['MSA_code'] == msa]['ratio of people taking testing'].values
            d = MSA_delta[MSA_delta['MSA_code'] == msa]['ratio of excessive time at home'].values
            e = MSA_delta[MSA_delta['MSA_code'] == msa]['ratio of people wearing face masks'].values
            f = MSA_delta[MSA_delta['MSA_code'] == msa]['ratio of people taking testing'].values
            stay_at_home = np.mean(MSA_all_temp['ratio of excessive time at home'].values)
            facemask = np.mean(MSA_all_temp['ratio of people wearing face masks'].values)
            testing = np.mean(MSA_all_temp['ratio of people taking testing'].values)
            if beta < 0:
                beta = 0
            if delta < 0:
                delta = 0
            print(msa, Sd, Id, Rd, Dd, stay_at_home, facemask, testing, beta, gamma, delta)
            [S_current, I_current, R_current, D_current] = [Sd / sum_pop, Id / sum_pop, Rd / sum_pop, Dd / sum_pop]
            df_infection = pd.DataFrame(
                columns=['date', 'S', 'I', 'R', 'D', 'newI', 'beta', 'gamma', 'delta', 'vaccination'])
            count = 0
            for vaccinex in vaccine_list:
                # print('vaccinex',vaccinex)
                [S_current_old, I_current_old, R_current_old, D_current_old] = [S_current, I_current, R_current,
                                                                                D_current]
                I_old_sum = I_current_old + R_current_old + D_current_old
                if (S_current - vaccinex) > 0:
                    S_current = S_current - vaccinex * 0.9
                else:
                    S_current = 0
                sol = solve_ivp(SIRD, [0, 1], [S_current, I_current, R_current, D_current] + [u, beta, gamma, delta],
                                t_eval=np.arange(0, 1 + 0.2, 0.2))
                [S_current, I_current, R_current, D_current] = [sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], sol.y[3][-1]]
                I_sum = I_current + R_current + D_current
                df_infection.loc[count] = [count] + [S_current * sum_pop, I_current * sum_pop, R_current * sum_pop,
                                                     D_current * sum_pop, (I_sum - I_old_sum) * sum_pop, beta, gamma,
                                                     delta, vaccinex]
                count += 1
            df_infection.to_csv(path_external + 'temp/' + str(msa) + case + str(vaccination) + ".csv")
            list_tempx = [x for x in (df_infection['newI'])]
            judge, day = zero_infection_days(list_tempx, sum_pop, run_days)
            if judge == 'true':
                df_result.loc[count_x] = [vaccination, msa, judge, day, df_infection['I'].values[day],
                                          df_infection['R'].values[day], df_infection['D'].values[day], sum_pop]
                print('true',
                      [vaccination, msa, judge, day, df_infection['I'].values[day], df_infection['R'].values[day],
                       df_infection['D'].values[day], sum_pop])
            else:
                print('false',
                      [vaccination, msa, judge, day, df_infection['I'].values[day], df_infection['R'].values[day],
                       df_infection['D'].values[day], sum_pop])
                df_result.loc[count_x] = [vaccination, msa, judge, day, df_infection['I'].values[day],
                                          df_infection['R'].values[day], df_infection['D'].values[day], sum_pop]
            count_x += 1
    df_result.to_csv(path_files + case + '-vaccination-' + str(period) + '.csv')


def plot_senario(df_result, case, path_files, period_day):
    # df_result['accumulated I']=[i+j+k for i,j,k in zip(df_result['accumulated I'],df_result['accumualted R'],df_result['accumualted D'])]
    # df_result['accumulated D'] = [i * df_result['sum_pop'].values[0] for i in df_result['accumulated D']]
    dict_temp = {}
    for msa in pd.unique(MSA_all['MSA_code']):
        Sd = MSA_all[MSA_all['MSA_code'] == int(msa)]['S(t)'].values[-1]
        Id = MSA_all[MSA_all['MSA_code'] == int(msa)]['I(t)'].values[-1]
        Rd = MSA_all[MSA_all['MSA_code'] == int(msa)]['R(t)'].values[-1]
        Dd = MSA_all[MSA_all['MSA_code'] == int(msa)]['D(t)'].values[-1]
        sum_pop = MSA_all[MSA_all['MSA_code'] == int(msa)]['S(t)'].values[0]
        dict_temp[msa] = [Sd, Id, Rd, Dd]
    for index, row in df_result.iterrows():
        msa_temp = row['MSA_code']
        print(msa_temp)
        df_result['accumulated I'][index] = (df_result['accumulated I'][index] + df_result['accumualted R'][index] +
                                             df_result['accumualted D'][index]) \
                                            - row['vaccination'] * sum_pop - (
                                                        dict_temp[msa_temp][1] + dict_temp[msa_temp][2] +
                                                        dict_temp[msa_temp][3])
        df_result['accumualted R'][index] = df_result['accumualted R'][index] - dict_temp[msa_temp][2]
        df_result['accumualted D'][index] = df_result['accumualted D'][index] - dict_temp[msa_temp][3]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    sns.stripplot(x='vaccination', y='days', jitter=True, split=True, linewidth=0.1, alpha=0.1, data=df_result, size=3,
                  palette="Blues", ax=ax1)

    sns.boxplot(x='vaccination', y='days', data=df_result, showfliers=False, showmeans=False, palette="Blues", ax=ax1)
    ax1.set_ylabel("days needed for zero infection")
    sns.stripplot(x='vaccination', y='accumulated I', jitter=True, split=True, linewidth=0.1, alpha=0.1, data=df_result,
                  size=3,
                  palette="Blues", ax=ax2)

    sns.boxplot(x='vaccination', y='accumulated I', data=df_result, showfliers=False, showmeans=False, palette="Blues",
                ax=ax2)
    ax2.set_ylabel("additional infected cases")
    sns.stripplot(x='vaccination', y='accumualted D', jitter=True, split=True, linewidth=0.1, alpha=0.1, data=df_result,
                  size=3,
                  palette="Blues", ax=ax3)

    sns.boxplot(x='vaccination', y='accumualted D', data=df_result, showfliers=False, showmeans=False, palette="Blues",
                ax=ax3)
    ax3.set_ylabel("additional dead cases")
    plt.tight_layout()
    fig.savefig(path_files + case + str(period_day) + ".png", dip=600)
    plot_senario_each_msa(df_result, case)


def plot_senario_each_msa(df_result, case):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    for msa in df_result['MSA_code']:
        df_temp = df_result[df_result['MSA_code'] == msa]
        df_temp = df_temp.sort_values(by=['vaccination'])
        ax1.plot(df_temp['vaccination'], df_temp['days'], color='#9ecae1', alpha=0.2, linewidth=1)
        ax2.plot(df_temp['vaccination'], df_temp['accumulated I'], color='#9ecae1', alpha=0.2, linewidth=1)
        ax3.plot(df_temp['vaccination'], df_temp['accumualted D'], color='#9ecae1', alpha=0.2, linewidth=1)
    ax1.set_ylabel("days needed for zero infection")
    ax2.set_ylabel("accumulated infected cases")
    ax3.set_ylabel("accumulated dead cases")
    plt.tight_layout()
    fig.savefig('analysis-results/scenarios/' + case + "(each_MSA).png", dip=600)


def adoption_figure_linear(n, path_file):
    ###take 0.7 as an example
    list_temp = vaccination_linear(n, 0.7, 270)
    list_temp = [np.sum(list_temp[0:i]) for i in range(len(list_temp))]
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    # ax.plot([i for i in range(n)],list_k,color='blue')
    ax.axvline(180, color='black', linewidth=1)
    ax.plot([i for i in range(270)], list_temp, color='#a50f15', linestyle='-')
    ax.set_xticks([0, 90, 180, 270])
    ax.set_xticklabels([' ', '', 'x months', ''])
    ax.set_yticks([0, 0.7])
    ax.set_yticklabels([' ', 'maximum ratio of' + '\n' + ' people full vaccinated'], rotation=90, fontsize=8)
    plt.tight_layout()
    fig.savefig(path_file + "adotpionfigure_linear.png", dpi=600)


def trunced_normal_distribution(lower_bar, upper_bar, sigma_range):
    mu = int((upper_bar - lower_bar) / 2)
    sigma = (upper_bar - lower_bar) / sigma_range
    s = [1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)) for bins in
         range(lower_bar, upper_bar)]
    s1 = np.cumsum([1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)) for bins in
                    range(lower_bar, upper_bar)])
    result = np.cumsum([i for i in s])
    result = [i / np.max(result) for i in result]
    return result


def adoption_real_data():
    from sklearn.metrics import mean_squared_error
    as_of_days = 36
    vaccine_df = pd.read_csv(path_external + "covid19-intervention-data/vaccination/20200221US_vaccination_SUM.csv")
    list_df = [i / 328200000 for i in vaccine_df['people_fully_vaccinated'].values]

    fig = plt.figure(figsize=(4, 2.5))
    ax = fig.add_subplot(111)
    cap=0.9
    color_dict = {120: "#c6dbef", 180: '#4292c6', 270: '#08519c', 360: '#08306b'}
    if cap==0.9:
        sigma_dict = {120: 6, 180: 5, 270: 3.5, 360: 2.8}
    if cap==0.8:
        sigma_dict = {120: 5.8, 180: 4.6, 270: 3.3, 360: 2.7}
    if cap==0.7:
        sigma_dict = {120: 5.4, 180: 4.3, 270: 3.1, 360: 2.6}
    if cap==0.6:
        sigma_dict = {120: 5.2, 180: 4.0, 270: 2.9, 360: 2.4}
    if cap==0.5:
        sigma_dict = {120: 5.1, 180: 3.9, 270: 2.8, 360: 2.2}
    if cap==0.4:
        sigma_dict = {120: 5.0, 180: 3.5, 270: 2.7, 360: 1.8}
    if cap==0.3:
        sigma_dict = {120: 4.5, 180: 3.0, 270: 2.0, 360: 1.0}
    if cap==0.2:
        sigma_dict = {120: 3.0, 180: 1.5, 270: 0.8, 360: 0.4}
    if cap==0.1:
        sigma_dict = {120: 0.5, 180: 0.1, 270: 0.1, 360: 0.1}
    for day in [120, 180, 270, 360]:

        results = trunced_normal_distribution(0, day, sigma_dict[day])
        print(results)
        results = [i * cap for i in results]
        error = mean_squared_error(results[0:as_of_days], list_df, squared=False)
        print(day, error)

        results = list(results) + [results[-1] for i in range(480 - day)]
        ax.plot([i for i in range(0, len(results))], results, color=color_dict[day], linewidth=1,label=r"$H(t)$" +" reach saturation in "+r'$T_s=$'+ str(day)+" days" )
        '''
        plt.plot([i for i in range(0, as_of_days)], results[0:as_of_days], color='red')
        plt.plot([i for i in range(0, as_of_days)], list_df, color='blue')
        plt.show()
        '''

    ax.plot([i for i in range(0, as_of_days)], list_df, color='#ff7f00', linewidth=1,
            label="Reported "+ r"$H(t)$")
    # plt.show()
    ax.legend(loc=0, fontsize=5)
    ax.set_xticks([0, 90, 180, 270, 360, 450])
    ax.set_xticklabels([0, 90, 180, 270, 360, 450])
    ax.set_yticks([0, 0.1, 0.9])
    ax.set_yticklabels([' 0', '0.1', 'Saturation'], rotation=45,fontsize=8)
    ax.set_ylabel("Ratio of people full vaccinated "+ r"$H(t)$",fontsize=8)
    ax.set_xlabel("t (days)",fontsize=8)
    plt.tight_layout()
    fig.savefig(path_file + "adotpionfigure_sigmoid_all.png", dpi=600)





if __name__ == '__main__':
    f_save = open('Dataset/MSA_demographics.pkl', 'rb')
    MSA_statistics = pickle.load(f_save)
    f_save.close()
    path_file = '/Users/luzhong/Documents/LuZHONGResearch/20200720COVID-Controllers/results_scenarios/'
    ###we have several intervention, stay at home, testing, facemask, vaccination
    #case='best_case_linear' ### immidiately
    case = 'best_case_sigmoid'
    case='adoption_figure'
    run_days = 1000

    if case == 'best_case_linear':
        for finish_period in [120, 180, 270, 360]:
            vaccination_type = 'linear'
            scenario_test(finish_period, run_days, case, vaccination_type,
                          path_file)  ###output: time needed for 1,0.1,0.01, 0,
            df_result = pd.read_csv(path_file + case + '-vaccination-' + str(finish_period) + '.csv')

    if case == 'best_case_sigmoid':
        for finish_period in [120, 180, 270, 360]:
            vaccination_type = 'sigmoid'
            scenario_test(finish_period, run_days, case, vaccination_type,
                          path_file)  ###output: time needed for 1,0.1,0.01, 0,
            df_result = pd.read_csv(path_file + case + '-vaccination-' + str(finish_period) + '.csv')

    if case == 'adoption_figure':
        adoption_real_data()
        # adoption_figure_linear(finish_period,path_file)
