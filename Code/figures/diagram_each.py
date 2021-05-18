import sys
sys.path.append('/Users/luzhong/Documents/pythonCode/PDE-COVID')
from input_library import *
import Input

def moving_average(list_example, n) :
    new_list=[]
    for i in range(len(list_example)):
        if i<=n:
            new_list.append(np.mean([list_example[j] for j in range(0,i+1) if list_example[j]>0]))
        else:
            new_list.append(np.mean([list_example[j] for j in range(i-n, i + 1) if list_example[j]>0]))

    return new_list

def output_scatter(Data_all,namestr_list,namestr_color_dict,namestr,path_results):
    from matplotlib.lines import Line2D
    datelist = []
    for date in pd.unique(Data_all['date']):
        d=datetime.datetime.strptime(date, '%Y-%m-%d')
        d =d.strftime('%b-%d,%Y')
        datelist.append(d[0:6])

    label_list = ['Stay-at-home order', 'Face-mask wearing', 'Testing']

    count=0
    for col,output_str in zip(namestr_list,namestr_list):
        fig, ax = plt.subplots(1, 1, figsize=(3, 2))
        Data_all_temp = Data_all.groupby(['date']).agg({col: ['mean', 'std']}).reset_index()
        ax.plot([i for i in range(len(Data_all_temp[col]['mean']))],Data_all_temp[col]['mean'], c=namestr_color_dict[col],linestyle='-',label=label_list[count])
        count+=1
        ax.set_ylim(0,1)
        ax.set_ylabel(output_str, fontsize=10)
        ax.set_xticks(np.arange(0, len(datelist), 30))
        ax.set_xticklabels([datelist[i * 30] for i in range(0, len(np.arange(0, len(datelist), 30)))], fontsize=8,
                           rotation=45)
        plt.tight_layout()
        plt.savefig(path_results+namestr+"_intervention_data"+col+".png",
                    dpi=600)
        plt.close()



def MSA_R_eff(MSA_all,path_files):
    interval=30
    datelist = []
    timelist=[]
    output_list=[]
    for date in np.unique(MSA_all['date']):
        d = datetime.datetime.strptime(str(date), '%Y-%m-%d')
        if d>= datetime.datetime.strptime('4/01/20', '%m/%d/%y'):
            output_list.append(date)
            timelist.append(date)
            d = d.strftime('%b-%d,%Y')
            datelist.append(d[0:6])
    MSA_all=MSA_all[MSA_all['date'].isin(output_list)]

    for y_select,y_0 in zip(['beta(t)','gamma(t)','delta(t)','R_eff(t)'],['beta_0','gamma_0','delta_0']):
        MSA_all[y_select]=MSA_all[y_select]-MSA_all[y_0].values[0]
        print(y_select,MSA_all[y_select])

    color_dict={'beta(t)': '#d7191c','gamma(t)':'#e66101','delta(t)':'#0571b0','R_eff(t)':'#33a02c'}
    for y_select,namestr in zip(['beta(t)','gamma(t)','delta(t)','R_eff(t)'],['beta(t)','delta(t)','gamma(t)','R_eff(t)']):
        data_df = MSA_all[[y_select,'date','MSA_code']]
        Data_all_temp = data_df.groupby(['date']).agg({y_select: ['mean', 'std']}).reset_index()

        fig, ax = plt.subplots(figsize=(3.8, 2))
        ax.plot([i for i in range(len(Data_all_temp[y_select]['mean']))], Data_all_temp[y_select]['mean'],
                c=color_dict[y_select], linestyle='-')
        #print((np.arange(0, len(datelist), interval)))
        #print([datelist[i * interval] for i in range(0, len(np.arange(0, len(datelist), interval)))],)
        ax.set_xticks(np.arange(0, len(datelist), interval))
        ax.set_xticklabels([datelist[i * interval] for i in range(0, len(np.arange(0, len(datelist), interval)))],
            fontsize=10, rotation=45)

        if y_select=='beta(t)':
            #plt.ylim(0.01,1.1)
            plt.ylabel(r'$U_{\beta}(t)$', fontsize=10)
        if y_select=='gamma(t)':
            #plt.ylim(0.005,1)
            plt.ylabel(r'$U_{\gamma}(t)$', fontsize=10)
        if y_select=='delta(t)':
            #plt.ylim(0.00001,1)
            plt.ylabel(r'$U_{\delta}(t)$', fontsize=10)
        if y_select=='R_eff(t)':
            #plt.ylim(0.8,12)
            plt.ylabel(r'$R_{eff}(t)$', fontsize=10)

        #plt.yscale('log')
        plt.tight_layout()

        plt.savefig(path_files+"MSA_all_"+y_select+".png", dpi=600)
        plt.close()


if __name__ == '__main__':
    MSA_all = pandas.read_csv('/Volumes/SeagateDrive/US-mobility/covid19-intervention-data/MSA_summary_indicatorsreported.csv')

    path_files='/Users/lucinezhong/Documents/LuZHONGResearch/20200720COVID-Controllers/diagram/'
    namestr_list_output = ['ratio of excessive time at home', 'ratio of people wearing face masks',
                           'ratio of people taking testing']
    namestr_color_dict = dict(zip(namestr_list_output, ['#355F8C', '#CB7132', '#447E36']))
    output_scatter(MSA_all, namestr_list_output,namestr_color_dict, 'MSA',path_files)
    MSA_R_eff(MSA_all, path_files)