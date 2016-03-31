import pandas as pd
import numpy as np
import numpy.random as npr
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.style.use('ggplot')

salary_url = 'http://umsalary.info/titlesearch.php?Title=%&Year=0&Campus=0'
salary_data = pd.read_html(salary_url)[2]

# Rename columns
salary_data.columns = salary_data.iloc[1]
salary_data = salary_data.iloc[2:]
salary_data = salary_data.rename(columns={'Deprtment': 'Department', 'FTR (Salary)': 'Salary'})
salary_data.head()

# Delete rows containing google_ad_client
ad_client = '<!--  google_ad_client = "pub-1413020312572192";  /* 728x15, created 2/10/09 */  ' \
            'google_ad_slot = "5343307882";  google_ad_width = 728;  google_ad_height = 15;  //-->'

salary_data = salary_data[salary_data.Name != ad_client]

# Extract just the first name and add it as a new column
salary_data['first_name'] = salary_data.Name.str.extract('[A-Za-z\.]+,\\s+([A-Za-z]+)').apply(str)

# Compare first names to US census data to get gender by doing a left join
census_data = pd.read_csv('us_census_data.csv')
salary_data = salary_data.merge(census_data, how='left', on='first_name')

# Fill the names that are not on the database as 'unknown'
salary_data[salary_data['gender'].isnull()]['gender'].fillna(value='unknown')
salary_data['gender'] = salary_data['gender'].fillna('unknown')

# Convert salary as a float from salary column
salary_data['Salary'] = salary_data.Salary.replace('[\$,]', '', regex=True).astype(float)

salary_data = salary_data[salary_data.Salary != 0]

# Convert titles to upper case
salary_data['Title'] = salary_data.Title.str.upper()

# create subsets of data
males = salary_data[salary_data['gender'] == 'male']
females = salary_data[salary_data['gender'] == 'female']
unknowns = salary_data[salary_data['gender'] == 'unknown']
m_f = salary_data[salary_data['gender'] != 'unknown']

# create scatter plot of male v. female salaries
salary_data['gender_int'] = np.nan
salary_data.loc[salary_data['gender'] == 'male', 'gender_int'] = 0
salary_data.loc[salary_data['gender'] == 'female', 'gender_int'] = 1

scatter = salary_data.plot(x='gender_int', y='Salary', kind='scatter', s=50, xticks=[],
                           alpha=0.25, yticks=[10000, 500000, 1000000, 1500000, 2000000, 2500000])
scatter.set_xlabel('Gender')
scatter.set_title('Scatter Plot of Salaries by Gender')

# create bar plots of female and male titles
f_titles = females.Title.value_counts().head().plot(kind='bar', color='r', alpha=0.25, rot=0)

m_titles = males.Title.value_counts().head().plot(kind='bar', color='r', alpha=0.25, rot=0)

# distribution plots
f_dist = females.Salary.plot(kind='kde', title='Probability Density')
f_dist.set_xlim(0, 500000)
f_dist.set_xlabel('Salary')

m_dist = males.Salary.plot(kind='kde', legend=True, ax=f_dist)
m_dist.legend(['Female', 'Male'])

# Top Professions
f_tp = females.Title.value_counts().head().plot(kind='bar', color='r', alpha=0.5, rot=0, title='Top Female Professions')
f_tp.set_ylabel('Count')

m_tp = males.Title.value_counts().head().plot(kind='bar', color='b', alpha=0.5, rot=0, title='Top Male Professions')
m_tp.set_ylabel('Count')

# Control for title, and see if the difference in salaries is significant with bootstrapping
t_set = list(set(salary_data.Title))
title_ctrl = []


def bootstrap(data, num_samples, statistic, alpha):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    n = len(data)
    idx = npr.randint(0, n, (num_samples, n))
    samples = data[idx]
    stat = np.sort(statistic(samples, 1))
    intvl = (int(stat[int((alpha / 2.0) * num_samples)]),
             int(stat[int((1 - alpha / 2.0) * num_samples)]))
    return intvl

for i in t_set:
    salary_m = males.loc[males.Title == i]['Salary']
    salary_f = females.loc[females.Title == i]['Salary']
    male_n = len(salary_m)
    female_n = len(salary_f)
    R = salary_f.mean() / salary_m.mean()
    if male_n >= 5 & female_n >= 5:
        intvl_m = bootstrap(np.array(salary_m), 10000, np.mean, 0.05)
        intvl_f = bootstrap(np.array(salary_f), 10000, np.mean, 0.05)
        if salary_f.mean() > salary_m.mean():
            if intvl_f[0] > intvl_m[1]:
                is_sig = 'yes'
            else:
                is_sig = 'no'
        else:
            if intvl_m[0] > intvl_f[1]:
                is_sig = 'yes'
            else:
                is_sig = 'no'

        title_ctrl.append((i, salary_m.mean(), salary_f.mean(), R, male_n, female_n, intvl_m, intvl_f, is_sig))
        print('done with ', i)
    else:
        pass

# Convert to dataframe
title_ctrl = pd.DataFrame(title_ctrl,
                          columns=['Title', 'male_mean', 'female_mean', 'Ratio', 'male_count', 'female_count',
                                   'male_interval', 'female_interval', 'is_sig']).sort_values(by='Ratio')
title_ctrl


# Slice for statistically significant data points
title_ctrl_sig = title_ctrl[title_ctrl.is_sig == 'yes']

# Plot wage ratio by titles
tc_plot = title_ctrl_sig.plot(x='Title', y='Ratio', kind='barh', alpha=0.5, legend=None,
                              title='Wage Ratio by Profession', fontsize=15)
tc_plot.set_xlabel('Wage Ratio (R)', fontsize=15)
tc_plot.text(1, 0, 'p <= 0.05, n >= 25', fontsize=15)

# Average salaries controlling for department & titles

titles = list(set(title_ctrl_sig2['Title']))
td_ctrl = []
for i in titles:
    deps = list(set(salary_data[salary_data.Title == i]['Department']))
    for j in deps:
        male_tdc = males[(males.Title == i) & (males.Department == j)]['Salary']
        female_tdc = females[(females.Title == i) & (females.Department == j)]['Salary']
        male_count = len(male_tdc)
        female_count = len(female_tdc)
        R = female_tdc.mean() / male_tdc.mean()
        p = ttest_ind(male_tdc, female_tdc)[1]

        td_ctrl.append((i, j, R, p, male_count, female_count))

td_ctrl = pd.DataFrame(td_ctrl, columns=['Title', 'Department', 'R', 'p_value', 'male_count', 'female_count'])

td_ctrl_sig = td_ctrl[(td_ctrl.p_value <= 0.05)]

print('done')


###Bootstrapping


def bootstrap(data, num_samples, statistic, alpha):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    n = len(data)
    idx = npr.randint(0, n, (num_samples, n))
    samples = data[idx]
    stat = np.sort(statistic(samples, 1))
    return (stat[int((alpha / 2.0) * num_samples)],
            stat[int((1 - alpha / 2.0) * num_samples)])
