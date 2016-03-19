import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import matplotlib

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

# Average salaries controlling for titles
t_set = list(set(salary_data.Title))
title_ctrl = []

for i in t_set:
    male_tc = males.loc[males.Title == i]['Salary']
    female_tc = females.loc[females.Title == i]['Salary']
    R = female_tc.mean() / male_tc.mean()
    p = ttest_ind(male_tc, female_tc)[1]
    male_n = len(male_tc)
    female_n = len(female_tc)
    title_ctrl.append((i, R, p, male_n, female_n))

# Convert to dataframe
title_ctrl = pd.DataFrame(title_ctrl, columns=['Title', 'Ratio', 'p_value', 'male_count', 'female_count']).sort_values(
    by='Ratio')

# Slice for statistically significant data points
title_ctrl_sig = title_ctrl[
    (title_ctrl['p_value'] <= 0.05) & (title_ctrl['male_count'] >= 25) & (title_ctrl['female_count'] >= 25)]
title_ctrl_sig2 = title_ctrl[(title_ctrl['p_value'] <= 0.05)]

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
