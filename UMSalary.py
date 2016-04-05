import pandas as pd
import numpy as np
import numpy.random as npr
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.style.use('ggplot')

# Define colors for plot in RGB
red = (214 / 255, 39 / 255, 40 / 255)
blue = (31 / 255, 119 / 255, 180 / 255)
red2 = (237 / 255, 102 / 255, 93 / 255)
blue2 = (114 / 255, 158 / 255, 206 / 255)

# Pull data from website and clean
salary_url = 'http://umsalary.info/titlesearch.php?Title=%&Year=0&Campus=0'
salary_data = pd.read_html(salary_url)[2]

# Rename columns
salary_data.columns = salary_data.iloc[1]
salary_data = salary_data.iloc[2:]
salary_data = salary_data.rename(columns={'Deprtment': 'Department', 'FTR (Salary)': 'Salary'})
salary_data.head()

# Delete rows with google client id
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
sns.set(font_scale=1)
scatter = sns.stripplot(salary_data[salary_data['gender'] != 'unknown']['gender'],
                        salary_data['Salary'], jitter=0.02, alpha=0.5)
scatter.set_ylim(0, )


# create bar plots of female and male titles
f_titles = females.Title.value_counts().head(25)
f_means = [females[females.Title == i]['Salary'].mean() for i in list(f_titles.keys())]
pd.Series(f_means, name='Avg Salary').plot(secondary_y=True, color=red, legend=True)
f_plot = f_titles.plot(kind='bar', color=red2, alpha=0.8)
f_plot.set_ylabel('Count'), f_plot.set_ylim(0, 1600), f_plot.right_ax.set_ylim(0, 200000),
f_plot.right_ax.set_ylabel('Avg Salary')
f_plot.set_title('Female Titles vs. Count & Avg Salary')

m_titles = males.Title.value_counts().head(25)
m_means = [males[males.Title == i]['Salary'].mean() for i in list(m_titles.keys())]
pd.Series(m_means, name='Avg Salary').plot(secondary_y=True, color=blue, legend=True)
m_plot = m_titles.plot(kind='bar', color=blue2, alpha=0.8)
m_plot.set_ylabel('Count'), m_plot.set_ylim(0, 1600), m_plot.right_ax.set_ylim(0, 200000)
m_plot.right_ax.set_ylabel('Avg Salary')
m_plot.set_title('Male Titles vs. Count & Avg Salary')

# distribution plots
f_dist = females.Salary.plot(kind='kde', title='Probability Density')
f_dist.set_xlim(0, 500000)
f_dist.set_xlabel('Salary')

m_dist = males.Salary.plot(kind='kde', legend=True, ax=f_dist)
m_dist.legend(['Female', 'Male'])

# Control for title, and see if the difference in salaries is significant with bootstrapping
def bootstrap(data, num_samples, statistic, alpha):
    n = len(data)
    idx = npr.randint(0, n, (num_samples, n))
    samples = data[idx]
    stat = np.sort(statistic(samples, 1))
    intvl = (int(stat[int((alpha / 2.0) * num_samples)]),
             int(stat[int((1 - alpha / 2.0) * num_samples)]))
    return (stat, intvl)


def perm_test(xs, ys, nmc):
    n, k = len(xs), 0
    diff = np.abs(np.mean(xs) - np.mean(ys))
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc


t_set = list(set(salary_data.Title))
title_ctrl = []
for i in t_set:
    salary_m = males.loc[males.Title == i]['Salary']
    salary_f = females.loc[females.Title == i]['Salary']
    male_n = len(salary_m)
    female_n = len(salary_f)
    m_range = (salary_m.min(), salary_m.max())
    f_range = (salary_f.min(), salary_f.max())
    R = salary_f.mean() / salary_m.mean()

    if male_n >= 5 & female_n >= 5:
        intvl_m = bootstrap(np.array(salary_m), 10000, np.mean, 0.05)[1]
        intvl_f = bootstrap(np.array(salary_f), 10000, np.mean, 0.05)[1]
        dist_m = bootstrap(np.array(salary_m), 10000, np.mean, 0.05)[0]
        dist_f = bootstrap(np.array(salary_f), 10000, np.mean, 0.05)[0]
        dist_diff = dist_m - dist_f
        p = perm_test(salary_m, salary_f, 10000)

        title_ctrl.append(
            (i, salary_m.mean(), salary_f.mean(), m_range, f_range, R, p, male_n, female_n, intvl_m, intvl_f))
        print('done with ', i)
    else:
        pass

# Convert to dataframe
title_ctrl = pd.DataFrame(title_ctrl,
                          columns=['Title', 'male_mean', 'female_mean', 'm_range', 'f_range', 'Ratio', 'p_value',
                                   'male_count', 'female_count',
                                   'male_interval', 'female_interval']).sort_values(by='Ratio')
# Plot interval distribution of salaries
tc2 = title_ctrl.sort_values(by='male_mean')

tc2[['male_min', 'male_max']] = tc2['male_interval'].apply(pd.Series)
tc2[['female_min', 'female_max']] = tc2['female_interval'].apply(pd.Series)

tc2['male_lerr'] = abs(tc2['male_mean'] - tc2['male_min'])
tc2['male_uerr'] = abs(tc2['male_mean'] - tc2['male_max'])
tc2['female_uerr'] = abs(tc2['female_mean'] - tc2['female_max'])
tc2['female_lerr'] = abs(tc2['female_mean'] - tc2['female_min'])

maledist_plot = tc2.plot(x='Title', y='male_mean', color=blue2, yerr=[tc2.male_lerr, tc2.male_uerr],
                         kind='bar', width=0, error_kw=dict(ecolor=blue2, lw=4, capsize=5, capthick=1))

maledist_plot.set_ylim(0, 200000)
femaledist_plot = tc2.plot(x='Title', y='female_mean', color=red2, yerr=[tc2.female_lerr, tc2.female_uerr],
                           kind='bar', width=0, ax=maledist_plot,
                           error_kw=dict(ecolor=red2, lw=4, capsize=5, capthick=1))
femaledist_plot.set_title('Distribution of Salary Means by Title')
femaledist_plot.set_ylabel('Salary')

# Slice for statistically significant data points
tc_sig = title_ctrl[(title_ctrl.p_value < 0.05) & (abs(title_ctrl.Ratio - 1) > 0.001)]

# Plot wage ratio deviation by titles
tc_sig['ratio_dev'] = tc_sig.Ratio - 1
tc_plot = tc_sig.plot(x='Title', y='ratio_dev', kind='barh', alpha=1, legend=None,
                      title='Wage Ratio Deviation by Profession', color=6 * [blue] + 4 * [red])
tc_plot.set_xlim(-0.4, 0.4)
tc_plot.set_xlabel('Wage Ratio Deviation')

for index, patch in enumerate(tc_plot.patches):
    if index > 5:
        tc_plot.text(patch.get_width() / 2, patch.get_y() + 0.16, round(tc_sig.Ratio.iloc[index] - 1, 2))
    else:
        tc_plot.text(patch.get_width() / -2, patch.get_y() + 0.16, round(tc_sig.Ratio.iloc[index] - 1, 2))

# Average salaries controlling for department & titles
titles = list(set(tc_sig['Title']))
td_ctrl = []
for i in titles:
    deps = list(set(salary_data[salary_data.Title == i]['Department']))
    for j in deps:
        salary_male = males[(males.Title == i) & (males.Department == j)]['Salary']
        salary_female = females[(females.Title == i) & (females.Department == j)]['Salary']
        male_n = len(salary_male)
        female_n = len(salary_female)
        R = salary_female.mean() / salary_male.mean()
        if male_n >= 5 & female_n >= 5:
            intvl_m = bootstrap(np.array(salary_male), 10000, np.mean, 0.05)
            intvl_f = bootstrap(np.array(salary_female), 10000, np.mean, 0.05)
            p = perm_test(salary_male, salary_female, 10000)
            td_ctrl.append((i, j, male_n, female_n, R, p))

        print('done with ', i, ', ', j)

td_ctrl = pd.DataFrame(td_ctrl, columns=['Title', 'Department', 'male_count', 'female_count', 'Ratio', 'p_value'])

tdc_sig = td_ctrl[(td_ctrl.p_value < 0.05)].sort_values(by='Ratio')
print('done with TD control')
