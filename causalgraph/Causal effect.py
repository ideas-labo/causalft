
import lingam
import numpy as np
import pandas as pd
import graphviz
from lingam.utils import make_dot,make_prior_knowledge,make_dot_highlight



import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


# load german credit risk dataset
data_path = ('proc_german_num_02 withheader-2.csv')
df = pd.read_csv(data_path)


# preprocess data
data = df.values.astype(np.int32)
data[:,0] = (data[:,0]==1).astype(np.int64)
bins_loan_nurnmonth = [0] + [np.percentile(data[:,2], percent, axis=0) for percent in [25, 50, 75]] + [80]
bins_creditamt = [0] + [np.percentile(data[:,4], percent, axis=0) for percent in [25, 50, 75]] + [200]
bins_age = [15, 25, 45, 65, 120]
list_index_num = [2, 4, 10]
list_bins = [bins_loan_nurnmonth, bins_creditamt, bins_age]
for index, bins in zip(list_index_num, list_bins):
    data[:, index] = np.digitize(data[:, index], bins, right=True)


# split data into training data and test data
X = data[:, 1:]
y = data[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# set constraints for each attribute, 839808 data points in the input space
constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T



df.columns

prior_knowledge = make_prior_knowledge(
    n_variables=25,
    exogenous_variables = [10],
    sink_variables=[0],
)
print(prior_knowledge)

model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
model.fit(data)

model.get_error_independence_p_values(data)

# labels = df.columns.values
labels =['CREDITRATING', 'BalanceCheque', 'Loan_NurnMonth', 'CreditHistory',
       'CreditAmt', 'SavingsBalance', 'Mths_employ', 'PersonStatusSex',
       'PresentResidence', 'Property', 'AgeInYears', 'OtherInstPlans',
       'NumCreditsThisBank', 'NumPplLiablMaint', 'Telephone',
       'ForeignWorker', 'Purpose_CarNew', 'Purpose_CarOld',
       'otherdebtor_noneVsGuar', 'otherdebt_coapplVsGuar',
       'house_rentVsFree', 'house_ownsVsFree', 'job_unemployedVsMgt',
       'jobs_unskilledVsMgt', 'job_skilledVsMgt']

make_dot(model.adjacency_matrix_,labels=labels)

def make_prior_knowledge_graph(prior_knowledge_matrix):
    d = graphviz.Digraph(engine='dot')

    labels = [f'x{i}' for i in range(prior_knowledge_matrix.shape[0])]
    for label in labels:
        d.node(label, label)

    dirs = np.where(prior_knowledge_matrix > 0)
    for to, from_ in zip(dirs[0], dirs[1]):
        d.edge(labels[from_], labels[to])

    dirs = np.where(prior_knowledge_matrix < 0)
    for to, from_ in zip(dirs[0], dirs[1]):
        if to != from_:
            d.edge(labels[from_], labels[to], style='dashed')
    return d

prior_knowledge = make_prior_knowledge(
    n_variables=25,
    exogenous_variables = [7,10],
    sink_variables=[0],
)
print(prior_knowledge)

labels = df.columns.values
labels =['CREDITRATING', 'BalanceCheque', 'Loan_NurnMonth', 'CreditHistory',
       'CreditAmt', 'SavingsBalance', 'Mths_employ', 'PersonStatusSex',
       'PresentResidence', 'Property', 'AgeInYears', 'OtherInstPlans',
       'NumCreditsThisBank', 'NumPplLiablMaint', 'Telephone',
       'ForeignWorker', 'Purpose_CarNew', 'Purpose_CarOld',
       'otherdebtor_noneVsGuar', 'otherdebt_coapplVsGuar',
       'house_rentVsFree', 'house_ownsVsFree', 'job_unemployedVsMgt',
       'jobs_unskilledVsMgt', 'job_skilledVsMgt']

model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
model.fit(data)

# dot =make_dot_highlight(model.adjacency_matrix_,node_index=0,labels = labels)
# dot1 = make_dot(model.adjacency_matrix_,labels = labels)

make_dot(model.adjacency_matrix_,labels = labels)

causal_effects = model.adjacency_matrix_
print(causal_effects)

# Assign to pandas.DataFrame for pretty display
df = pd.DataFrame(causal_effects)

# labels = cols
df['from'] = df['from'].apply(lambda x : label2[x])
df['to'] = df['to'].apply(lambda x : label2[x])
df

te = model.estimate_total_effect(data, 7, 0)
print(f'total effect: {te:.3f}')

te = model.estimate_total_effect(data, 10, 0)
print(f'total effect: {te:.3f}')

dot.format = 'png'
dot.render('grman')
dot1.format = 'png'
dot1.render('german1')

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn_pandas import DataFrameMapper
df_raw = pd.read_csv("compas_rec.csv")
df=df_raw.drop(columns = ['Name'])

X_train, X_test= train_test_split(df,test_size=0.4, random_state=42)
mapper = DataFrameMapper([('Sex', LabelEncoder()),('Race', LabelEncoder()),
                          ('Degree of Conviction', LabelEncoder())], df_out=True, default=None)
df5 = mapper.fit_transform(X_train.copy())

print(df5.columns)

label1 =['Sex', 'Race', 'Degree of Conviction', 'Age', 'Juvenile Felonies',
       'Juvenile Misdemeanors', 'Juvenile Others', 'Previous Convictions',
       'Days in Jail', 'COMPAS Score', 'Recidivism']
prior_knowledge = make_prior_knowledge(
    n_variables=11,
    exogenous_variables = [0,1,3],
    sink_variables=[10],
)
print(prior_knowledge)

model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
model.fit(df5)
# dot2 = make_dot(model.adjacency_matrix_,labels = label1)
# dot2.format = 'png'
# dot2.render('compas')

result = model.bootstrap(df5, n_sampling=20)

causal_effects = result.get_total_causal_effects(min_causal_effect=0.01)

# Assign to pandas.DataFrame for pretty display
df = pd.DataFrame(causal_effects)

# labels = cols
df['from'] = df['from'].apply(lambda x : label1[x])
df['to'] = df['to'].apply(lambda x : label1[x])
df

df_age = df[df['from']=='Age']

df_age1=df_age.sort_values('effect', ascending=True)
df_age1.to_csv("Age_effect_rank.csv")

df_race = df[df['from']=='Race']

df_race1=df_race.sort_values('effect', ascending=True)
df_race1.to_csv("Race_effect_rank.csv")

df_sex = df[df['from']=='Sex']

df_sex1 = df_sex.sort_values('effect', ascending=True)
df_sex1.to_csv("Sex_effect_rank.csv")

dot3 =make_dot_highlight(model.adjacency_matrix_,node_index=7,labels = label1)
dot3.format = 'png'
dot3.render('compas1')

df = pd.read_csv("dutch.csv")
data2 = df
data2['Sex'] = 1-(data2['sex'].astype('category')).cat.codes
df2 = data2.drop(columns=["sex"])
X_train, X_test= train_test_split(df2,test_size=0.8, random_state=42)
df1 = X_train
# df1.head()

label2 = ['age', 'household_position', 'household_size', 'prev_residence_place',
       'citizenship', 'country_birth', 'edu_level', 'economic_status',
       'cur_eco_activity', 'marital_status', 'occupation', 'Sex']
prior_knowledge = make_prior_knowledge(
    n_variables=12,
    exogenous_variables = [0,11],
    sink_variables=[10],
)
print(prior_knowledge)

model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
model.fit(df1)
result = model.bootstrap(df1, n_sampling=20)
causal_effects = result.get_total_causal_effects(min_causal_effect=0.01)

# Assign to pandas.DataFrame for pretty display
df = pd.DataFrame(causal_effects)

# labels = cols
df['from'] = df['from'].apply(lambda x : label2[x])
df['to'] = df['to'].apply(lambda x : label2[x])
df
# dot4 = make_dot(model.adjacency_matrix_,labels = label2)
# dot4.format = 'png'
# dot4.render('Dutch')
# dot5 =make_dot_highlight(model.adjacency_matrix_,node_index=10,labels = label2)
# dot5.format = 'png'
# dot5.render('Dutch1')

df_age = df[df['from']=='age']

df_age1=df_age.sort_values('effect', ascending=True)
df_age1.to_csv("Age_effect_rank3.csv")


df_sex = df[df['from']=='Sex']

df_sex1 = df_sex.sort_values('effect', ascending=True)
df_sex1.to_csv("Sex_effect_rank3.csv")

df4 = pd.read_csv("/content/communities_crime.csv")
data2 = df4
df3=data2.drop(columns=["communityname"])
df3.head()

prior_knowledge = make_prior_knowledge(
    n_variables=104,
    exogenous_variables = [102],
    sink_variables=[103],
)
print(prior_knowledge)

labels5 = list(df3.columns)
labels5

model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
model.fit(df3)
make_dot(model.adjacency_matrix_,labels = labels5)

df4 = pd.read_csv("/content/adult-clean.csv")
df4

"""测试Adult数据集，得到因果网络图后，drop掉最终导向不是income的feature"""

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn_pandas import DataFrameMapper
df4 = pd.read_csv("/content/adult-clean.csv")


cols = ['age', 'education', 'workclass', 'marital-status', 'occupation',
       'relationship', 'race', 'gender', 'native-country','fnlwgt',
       'educational-num', 'capital-gai','capital-loss','hours-per-week','income']


mapper = DataFrameMapper([('age', LabelEncoder()),('education', LabelEncoder()),
                          ('workclass', LabelEncoder()),('marital-status', LabelEncoder()),
                          ('occupation', LabelEncoder()),('relationship',LabelEncoder()),
                          ('race', LabelEncoder()),('gender', LabelEncoder()),
                          ('native-country', LabelEncoder())], df_out=True, default=None)
df5 = mapper.fit_transform(df4.copy())
df5.columns

df6=df5.drop(columns=['capital-loss','fnlwgt','capital-gain'])
cols1 = ['Age', 'Education', 'Workclass', 'Marital-status', 'Occupation',
       'Relationship', 'Race', 'Gender', 'Native-country', 'Educational-num','Hours-per-week',
       'Class-label']

prior_knowledge = make_prior_knowledge(
    n_variables=12,
    exogenous_variables = [0],
    sink_variables=[11],
)
print(prior_knowledge)

model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
model.fit(df6)
# dot_format = make_dot(model.adjacency_matrix_,labels = cols1)
# dot_format1 = nx.drawing.nx_pydot.to_pydot(dot_format).to_string()
G = make_dot(model.adjacency_matrix_,labels = cols1)

display(G)

# dot_format = nx.drawing.nx_pydot.to_pydot(G).to_string()
# dot_format
# import networkx as nx
# G = nx.DiGraph()
# colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightgoldenrodyellow']
# for i in range(adj_matrix.shape[0]):
#     G.add_node(f'X{i+1}', label=f'Node {i+1}', style='filled', fillcolor=colors[i % len(colors)])

# # 添加边（不显示权重）
# for i in range(adj_matrix.shape[0]):
#     for j in range(adj_matrix.shape[1]):
#         if adj_matrix[i, j] != 0:
#             G.add_edge(f'X{i+1}', f'X{j+1}')

# # 导出为DOT格式
# dot_format = nx.drawing.nx_pydot.to_pydot(G).to_string()

# 修改DOT格式字符串（如果需要）
# 例如，添加图的全局属性
# dot_format = 'digraph G {\n' + 'graph [rankdir=LR];\n' + dot_format[dot_format.find('{') + 1:dot_format.rfind('}')]
# s = Source(dot_format)
# display(s)
# 打印修改后的DOT格式字符串
# print(dot_format)
# dot6.format = 'png'
# dot6.render('Adult1')
# with open('causal_graph.dot', 'w') as f:
#     f.write(s)

import numpy as np
import pandas as pd

from lingam import RESIT
from sklearn.ensemble import RandomForestRegressor

from lingam.utils import make_dot, visualize_nonlinear_causal_effect
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
reg = RandomForestRegressor()
X = df6
train_df, test_df = train_test_split(X, test_size=0.3, random_state=42)
# model = lingam.RESIT(regressor=reg)
# model.fit(X)



# model = lingam.RESIT(regressor=reg)
# model.fit(train_df)
model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
model.fit(train_df)

labels = ['Age', 'Education', 'Workclass', 'Marital-status', 'Occupation',
       'Relationship', 'Race', 'Gender', 'Native-country', 'Educational-num','Hours-per-week',
       'Class-label']
G = make_dot(model.adjacency_matrix_,labels=cols1)
display(G)
G.format = 'png'
G.render('causal')

G.node('age', style='filled', fillcolor='red')

print(G.source)
with open('causal_graph.dot', 'w') as f:
   f.write(G.source)

from graphviz import Source
with open('causal_graph.dot', 'r') as file:
    dot_source = file.read()
dot = Source(dot_source)
display(dot)
dot.format = 'png'
dot.render('causal')

import matplotlib.pyplot as plt

# 定义颜色及其对应的图例说明
colors = ['#FF000080', '#00FF0080']
labels = ['Sensitive feature', 'Non-sensitive feature']

# 创建一个空白的图形
fig, ax = plt.subplots()

# 添加图例说明
for i, (color, label) in enumerate(zip(colors, labels)):
    ax.text(0.7, 0.72 - i*0.1, label, fontsize=12, ha='left', va='center')
    ax.add_patch(plt.Rectangle((0.64, 0.7 - i*0.1), 0.05, 0.05, color=color, transform=ax.transAxes))

# 设置图形的范围和标签
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.savefig('legend.png')
# 显示图形
plt.show()

import matplotlib.pyplot as plt

# 定义颜色及其对应的图例说明
colors = ['#FF000080', '#00FF0080']
labels = ['Concerned sensitive feature', 'Concerned non-sensitive feature']

# 创建一个空白的图形
fig, ax = plt.subplots(figsize=(10, 6))  # 调整图形的大小

# 添加图例说明
for i, (color, label) in enumerate(zip(colors, labels)):
    ax.text(0.155 + i * 0.7, 0.92, label, fontsize=24, ha='left', va='center')
    ax.add_patch(plt.Rectangle((0.1 + i * 0.7, 0.9), 0.05, 0.05, color=color, transform=ax.transAxes))

# 设置图形的范围和标签
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# 调整子图形和标签的距离
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# 保存图形
plt.savefig('legend.png')

# 显示图形
plt.show()

import matplotlib.pyplot as plt

# 定义颜色及其对应的图例说明
colors = ['#FF000080', '#00FF0080']
labels = ['Concerned sensitive feature', 'Concerned non-sensitive feature']

# 创建一个空白的图形
fig, ax = plt.subplots(figsize=(8, 2))  # 调整图形的大小

# 添加图例说明
handles = []
for color, label in zip(colors, labels):
    rect = plt.Rectangle((0, 0), 1, 1, color=color)
    handles.append(rect)
    # ax.text(0.2 + len(handles) * 0.3, 0.5, label, fontsize=12, ha='left', va='center')

# 创建图例并设置水平排列在顶部
ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=len(labels))

# 设置图形的范围和标签
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# 调整子图形和标签的距离
plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)

# 保存图形
plt.savefig('legend.png')

# 显示图形
plt.show()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 定义颜色及其对应的图例说明
colors = ['#FF000080', '#00FF0080']
labels = ['Concerned sensitive feature', 'Non-sensitive feature']

# 创建一个空白的图形
fig, ax = plt.subplots(figsize=(16,9))  # 调整图形的大小

# 添加图例说明和颜色块
patches = []
for i, (color, label) in enumerate(zip(colors, labels)):
    rect = mpatches.Rectangle((0.05 + i * 0.6, 0.9), 0.05, 0.05, color=color, transform=ax.transAxes)
    patches.append(rect)
    ax.text(0.105 + i * 0.6, 0.92, label, fontsize=30, ha='left', va='center')

# 添加颜色块到图例中，并设置边框属性
for patch in patches:
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)
    ax.add_patch(patch)

# 设置图形的范围和标签
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# 调整子图形和标签的距离
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# 保存图形
plt.savefig('legend.png')

# 显示图形
plt.show()

adj_matrix = model.adjacency_matrix_

import networkx as nx
G = nx.DiGraph()
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightgoldenrodyellow']
for i in range(adj_matrix.shape[0]):
    G.add_node(f'X{i+1}', label=f'Node {i+1}', style='filled', fillcolor=colors[i % len(colors)])

# 添加边（不显示权重）
for i in range(adj_matrix.shape[0]):
    for j in range(adj_matrix.shape[1]):
        if adj_matrix[i, j] != 0:
            G.add_edge(f'X{i+1}', f'X{j+1}')

# 导出为DOT格式
dot_format = nx.drawing.nx_pydot.to_pydot(G).to_string()

# 修改DOT格式字符串（如果需要）
# 例如，添加图的全局属性
# dot_format = 'digraph G {\n' + 'graph [rankdir=LR];\n' + dot_format[dot_format.find('{') + 1:dot_format.rfind('}')]
s = Source(dot_format)
display(s)
# 打印修改后的DOT格式字符串
# print(dot_format)
# dot6.format = 'png'
# dot6.render('Adult1')
# with open('causal_graph.dot', 'w') as f:
#     f.write(dot_format)

dot6= make_dot(model.adjacency_matrix_,labels = cols1)
dot6.format = 'png'
dot6.render('Adult1')

model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
result = model.bootstrap(df6, n_sampling=20)
# result1 = model.bootstrap(df6, n_sampling=10)
# result2 = model.bootstrap(df6, n_sampling=20)

pd = result.adjacency_matrices_

print(pd[0])

paths, effects = find_all_paths(pd , "x6", 'x8')

print(result.total_effects_)

causal_effects = result.get_total_causal_effects()

# Assign to pandas.DataFrame for pretty display
df = pd.DataFrame(causal_effects)

# labels = cols
df['from'] = df['from'].apply(lambda x : cols1[x])
df['to'] = df['to'].apply(lambda x : cols1[x])
s1= df.sort_values('effect', ascending=True)
# s1
df_race = df[df['from']=='race']

df_race1=df_race.sort_values('effect', ascending=True)
df_race1

df_age = df[df['from']=='age']

df_age1=df_age.sort_values('effect', ascending=True)
df_age1

df_sex = df[df['from']=='gender']

df_sex1=df_sex.sort_values('effect', ascending=True)
df_sex1

causal_effects1 = result1.get_total_causal_effects(min_causal_effect=0.01)

# Assign to pandas.DataFrame for pretty display
df = pd.DataFrame(causal_effects1)

# labels = cols
df['from'] = df['from'].apply(lambda x : cols1[x])
df['to'] = df['to'].apply(lambda x : cols1[x])
df
df_race = df[df['from']=='race']

df_race1=df_race.sort_values('effect', ascending=True)
df_race1

causal_effects2= result2.get_total_causal_effects(min_causal_effect=0.01)

# Assign to pandas.DataFrame for pretty display
df = pd.DataFrame(causal_effects2)

# labels = cols
df['from'] = df['from'].apply(lambda x : cols[x])
df['to'] = df['to'].apply(lambda x : cols[x])
df

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df6, test_size=0.999, random_state=42)
model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
model.fit(train_df)

# dot6= make_dot(model.adjacency_matrix_,labels = cols1)
# dot6.format = 'png'
# dot6.render('Adult1')
result = model.bootstrap(train_df, n_sampling=1)
# result1 = model.bootstrap(train_df, n_sampling=10)

from sklearn.linear_model import LinearRegression

target = 0 # mpg
features = [i for i in range(train_df.shape[1]) if i != target]
reg = LinearRegression()
reg.fit(train_df.iloc[:, features], train_df.iloc[:, target])

ce = lingam.CausalEffect(model)
effects = ce.estimate_effects_on_prediction(train_df, target, reg)

df_effects = pd.DataFrame()
df_effects['feature'] = train_df.columns
df_effects['effect_plus'] = effects[:, 0]
df_effects['effect_minus'] = effects[:, 1]
df_effects

causal_effects = result.get_total_causal_effects(min_causal_effect=0.01)

# Assign to pandas.DataFrame for pretty display
df = pd.DataFrame(causal_effects)

# labels = cols
df['from'] = df['from'].apply(lambda x : cols1[x])
df['to'] = df['to'].apply(lambda x : cols1[x])
# df_race = df[df['from']=='race']
s1= df.sort_values('effect', ascending=True)
s1
df_race = df[df['from']=='gender']

df_race1=df_race.sort_values('effect', ascending=True)
df_race1
# df_race1=df_race.sort_values('effect', ascending=True)
# df_race1

causal_effects1 = result1.get_total_causal_effects(min_causal_effect=0.01)

# Assign to pandas.DataFrame for pretty display
df = pd.DataFrame(causal_effects1)

# labels = cols
df['from'] = df['from'].apply(lambda x : cols1[x])
df['to'] = df['to'].apply(lambda x : cols1[x])
df
df_race = df[df['from']=='race']

df_race1=df_race.sort_values('effect', ascending=True)
df_race1

# from lingam.utils import print_dagc
# dagc = result.get_directed_acyclic_graph_counts(n_dags=3, min_causal_effect=0.01, split_by_causal_effect_sign=True)

# print_dagc(dagc, 20)

# col = [cols[0],cols[6],cols[7]]
# col

causal_effects = result.get_total_causal_effects(min_causal_effect=0.01)

# Assign to pandas.DataFrame for pretty display
df = pd.DataFrame(causal_effects)

# labels = cols
df['from'] = df['from'].apply(lambda x : cols[x])
df['to'] = df['to'].apply(lambda x : cols[x])
df

df_age = df[df['from']=='age']

df_age1=df_age.sort_values('effect', ascending=True)
df_age1.to_csv("Age_effect_rank.csv")

df_race = df[df['from']=='race']

df_race1=df_race.sort_values('effect', ascending=True)
df_race1.to_csv("Race_effect_rank.csv")

df_sex = df[df['from']=='gender']

df_sex1 = df_sex.sort_values('effect', ascending=True)
df_sex1.to_csv("Sex_effect_rank.csv")

from_index = 0
to_index = 14

path_age = pd.DataFrame(result.get_paths(from_index, to_index))
path_age.sort_values('effect', ascending=False).head(10)

from sklearn.linear_model import LinearRegression

X =df5.drop('Class-label', axis=1)
Y = df5['Class-label']
reg = LinearRegression()
reg.fit(X, Y)





ce = lingam.CausalEffect(model)
target =14
effects = ce.estimate_effects_on_prediction(df5,target, reg)
effects

ce = lingam.CausalEffect(causal_model=model)
ce

causal_effects = model.adjacency_matrix_
print(causal_effects)

# p_values = model.get_error_independence_p_values(df5)
# print(p_values)

"""age to income total effect"""

te = model.estimate_total_effect(df5, 0, 14)
print(f'total effect: {te:.3f}')

"""race to income total effect"""

te = model.estimate_total_effect(df5, 6, 14)
print(f'total effect: {te:.3f}')

"""sex to income total effect"""

te = model.estimate_total_effect(df5, 7, 14)
print(f'total effect: {te:.3f}')

"""测试lawSchool

"""

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn_pandas import DataFrameMapper
df_ls = pd.read_csv("/content/law_school_clean.csv")
mapper = DataFrameMapper([('race', LabelEncoder())], df_out=True, default=None)
df_ls1 = mapper.fit_transform(df_ls.copy())
col1=list(df_ls1.columns)
col1

prior_knowledge = make_prior_knowledge(
    n_variables=12,
    exogenous_variables = [0,9],
    sink_variables=[11],
)
print(prior_knowledge)

model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
model.fit(df_ls1)
# dot6= make_dot(model.adjacency_matrix_,labels = col1)
# dot6.format = 'png'
# dot6.render('LawSchool')
result = model.bootstrap(df_ls1, n_sampling=20)
causal_effects = result.get_total_causal_effects(min_causal_effect=0.01)

# Assign to pandas.DataFrame for pretty display
df = pd.DataFrame(causal_effects)

# labels = cols
df['from'] = df['from'].apply(lambda x : col1[x])
df['to'] = df['to'].apply(lambda x : col1[x])
df

model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
model.fit(df_ls1)

df_race = df[df['from']=='male']

df_race1=df_race.sort_values('effect', ascending=True)
df_race1

# df_race1.to_csv("Race_effect_rank1.csv")

# df_sex = df[df['from']=='male']

# df_sex1 = df_sex.sort_values('effect', ascending=True)
# df_sex1.to_csv("Sex_effect_rank1.csv")

df_2 =pd.concat([df[df['from']=='lsat'],df[df['to']=='lsat']])
df_2

df_3=pd.concat([df[df['from']=='decile1b'],df[df['to']=='decile1b']])
df_3

df_kdd = pd.read_csv('/content/kdd-census-income-clean.csv')

mapper = DataFrameMapper([('workclass', LabelEncoder()),('education', LabelEncoder()),("enroll-in-edu-inst-last-wk",LabelEncoder()),
                          ('marital-status', LabelEncoder()),('major-industry', LabelEncoder()),('major-occupation', LabelEncoder()),
                          ('race', LabelEncoder()),('hispanic-origin', LabelEncoder()),('sex', LabelEncoder()),
                          ('memner-union', LabelEncoder()),('reason-unemployment', LabelEncoder()),
                          ("employment-status",LabelEncoder()),("tax-filter-stat",LabelEncoder()),("region-previous-residence",LabelEncoder()),
                          ("state-previous-residence",LabelEncoder()),("detailed-household-and-family-stat",LabelEncoder()),("detailed-household-summary-in-household",LabelEncoder()),
                          ("live-hour-1-year-ago",LabelEncoder()),("family-members-under-18",LabelEncoder()),("country-father",LabelEncoder()),
                          ("country-mother",LabelEncoder()),("country-birth",LabelEncoder()),("citizenship",LabelEncoder()),
                          ("fill-questionnaire",LabelEncoder()),("income",LabelEncoder())], df_out=True, default=None)

df_kdd2 = mapper.fit_transform(df_kdd.copy())
col_kdd=list(df_kdd2.columns)
# df_kdd2.to_csv("problem.csv")
print(col_kdd[25])

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(df_kdd2, test_size=0.7, random_state=42)

prior_knowledge = make_prior_knowledge(
    n_variables=37,
    exogenous_variables = [6,8],
    sink_variables=[24],
)
print(prior_knowledge)

model = lingam.DirectLiNGAM(prior_knowledge)
model.fit(X_train)
# dot6= make_dot(model.adjacency_matrix_,labels = col_kdd)
# dot6.format = 'png'
# dot6.render('KDD')
result = model.bootstrap(X_train, n_sampling=20)
causal_effects = result.get_total_causal_effects(min_causal_effect=0.01)

# Assign to pandas.DataFrame for pretty display
df = pd.DataFrame(causal_effects)

# labels = cols
df['from'] = df['from'].apply(lambda x : col_kdd[x])
df['to'] = df['to'].apply(lambda x : col_kdd[x])
df

df_race = df[df['from']=='race']

df_race1=df_race.sort_values('effect', ascending=True)
df_race1.to_csv("Race_effect_rank2.csv")

df_sex = df[df['from']=='sex']

df_sex1 = df_sex.sort_values('effect', ascending=True)
df_sex1.to_csv("Sex_effect_rank2.csv")

df_age = df[df['from']=='age']

df_age1 = df_sex.sort_values('effect', ascending=True)
df_age1.to_csv("Age_effect_rank2.csv")

dot6= make_dot(model.adjacency_matrix_,labels = col_kdd)
dot6.format = 'png'
dot6.render('KDD1')