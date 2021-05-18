from input_library import *
import pickle
path_external='Dataset/'

def input_ctry():
    f_read_R = open(path_external+'modified_covid_data_US/ctry_recovery.pkl', 'rb')
    f_read_I = open(path_external+'modified_covid_data_US/ctry_infection.pkl', 'rb')
    f_read_S = open(path_external+'modified_covid_data_US/ctry_susecptible.pkl', 'rb')
    f_read_D = open(path_external+'modified_covid_data_US/ctry_dead.pkl', 'rb')
    ctry_recovered = pickle.load(f_read_R)
    ctry_infection= pickle.load(f_read_I)
    ctry_susecptible = pickle.load(f_read_S)
    ctry_dead = pickle.load(f_read_D)
    f_read_R.close();
    f_read_I.close();
    f_read_S.close();
    f_read_D.close()
    return ctry_susecptible, ctry_infection, ctry_recovered, ctry_dead

def input_MSA_statistics():
    f_save = open(path_external+'Population/MSA_demographics.pkl', 'rb')
    MSA_statistics = pickle.load(f_save)
    f_save.close()
    return MSA_statistics

def input_Ctry_statistics():
    f_save = open(path_external+'Population/Ctry_demographics.pkl', 'rb')
    Ctry_statistics = pickle.load(f_save)
    f_save.close()
    return Ctry_statistics

def input_State_statistics():
    f_save = open(path_external+'Population/State_demographics.pkl', 'rb')
    State_statistics = pickle.load(f_save)
    f_save.close()
    return State_statistics


def input_States_abbreviation():
    States_abbrev= {'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas','AS': 'American Samoa', 'AZ': 'Arizona',
                'CA': 'California', 'CO': 'Colorado','CT': 'Connecticut', 'DC': 'District of Columbia',
                'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia','GU': 'Guam', 'HI': 'Hawaii', 'IA': 'Iowa',
                'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana','KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana',
                'MA': 'Massachusetts', 'MD': 'Maryland', 'ME': 'Maine','MI': 'Michigan', 'MN': 'Minnesota', 'MO': 'Missouri',
                'MP': 'Northern Mariana Islands', 'MS': 'Mississippi','MT': 'Montana', 'NC': 'North Carolina',
                'ND': 'North Dakota', 'NE': 'Nebraska','NH': 'New Hampshire', 'NJ': 'New Jersey',
                'NM': 'New Mexico', 'NV': 'Nevada', 'NY': 'New York','OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
                'PA': 'Pennsylvania', 'PR': 'Puerto Rico','RI': 'Rhode Island', 'SC': 'South Carolina',
                'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas','UT': 'Utah', 'VA': 'Virginia','VI': 'Virgin Islands', 'VT': 'Vermont',
                'WAS': 'Washington', 'WA': 'Washington','WI': 'Wisconsin','WV': 'West Virginia', 'WY': 'Wyoming', 'RI': 'Rhode Island'}
    States_abbrev_code= {'AK': 2, 'AL': 1, 'AR': 5,'AS': 60, 'AZ': 4,
                'CA': 6, 'CO': 8,'CT': 9, 'DC': 11, 'DE': 10, 'FL': 12, 'GA': 13,'GU': 66, 'HI': 15, 'IA': 19,
                'ID': 16, 'IL': 17, 'IN': 18,'KS': 20, 'KY': 21, 'LA': 22,
                'MA': 25, 'MD': 24, 'ME': 23,'MI': 26, 'MN': 27, 'MO': 29,
                'MP': 69, 'MS': 28,'MT': 30, 'NC': 37,
                'ND': 38, 'NE': 31,'NH': 33, 'NJ': 34,
                'NM': 35, 'NV': 32, 'NY': 36,'OH': 39, 'OK': 40, 'OR': 41,
                'PA': 42, 'PR': 72,'RI': 44, 'SC':45,
                'SD': 46, 'TN': 47, 'TX': 48,'UT': 49, 'VA': 51,'VI': 78, 'VT': 50,
                'WAS': 53,'WA': 53, 'WI': 55,'WV': 54, 'WY': 56}
    return States_abbrev,States_abbrev_code


