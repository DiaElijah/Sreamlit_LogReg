import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

st.write("""
# Приложение, которое делает логарифмическую регрессию
""")

st.sidebar.header('Пользовательская настройка')
st.sidebar.write("""
Dataset должен быть оцифрован. \n
Обязательно укажи нормирован dataset или нет.
Таргет должен быть последней колонкой.
""")

st.sidebar.header('Твой датасет нормирован?')
norm = st.sidebar.selectbox('Выбери',('Да','Нет'))

uploaded_file = st.sidebar.file_uploader('Перетащи свой Dataset', type=['csv'])
uploaded_df = pd.read_csv(uploaded_file)

st.sidebar.header('Выбери количество итераций')
n_inputs = st.sidebar.slider('Количество итераций', 20,100000,1000)

st.sidebar.header('Выбери learning rate')
learning_rate = st.sidebar.slider('learning rate', 0.0001,1.0,0.1)

if norm == 'Нет':
    ss = StandardScaler()
    X = ss.fit_transform(uploaded_df.iloc[:, [0,-2]])
    y = uploaded_df.iloc[:, -1]
else:
    X = uploaded_df.iloc[:, [0,-2]]
    y = uploaded_df.iloc[:, -1]


def sigma(x):
        return 1 / (1 + np.exp(-x)) 

def fit(X, y):
    X = np.array(X)
    y = np.array(y)
    coef_ = np.random.uniform(size=X.shape[1])
    intercept_ = 1
    for epoch in range(n_inputs):
        y_pred  = intercept_ + X@coef_
        d_w0 = -(y - sigma(y_pred))
        d_w1_w2 = np.array([-X[:, i] * (y - sigma(y_pred)) 
                            for i in range(X.shape[1])])
        intercept_new = intercept_ - learning_rate * d_w0.mean()
        coef_new = coef_ - learning_rate * d_w1_w2.mean(axis=1)
        intercept_ = intercept_new
        coef_ = coef_new
    return (coef_[0], coef_[1], intercept_)

st.subheader('Веса для ваших данных')
R = {uploaded_df.columns.tolist()[i] : fit(X, y)[i] for i in range(X.shape[1])}
R['intercept' ] = fit(X, y)[-1]
st.write(R)

st.write("""
### Для построения графика выберите 2 фичи
""")
st.subheader('Фича 1')
F1 = st.selectbox(uploaded_df.columns.tolist()[0], uploaded_df.columns.tolist()[:-1])

st.subheader('Фича 2')
F2 = st.selectbox(uploaded_df.columns.tolist()[1], uploaded_df.columns.tolist()[:-1])

st.subheader('Прекрасный график')
# plt.scatter(uploaded_df[uploaded_df.iloc[:, -1] == 0][F1], uploaded_df[uploaded_df.iloc[:, -1] == 0][F2], color='blue', label='0')
# plt.scatter(uploaded_df[uploaded_df.iloc[:, -1] == 1][F1], uploaded_df[uploaded_df.iloc[:, -1] == 1][F2], color='red', label='1')
# plt.xlabel(F1)
# plt.ylabel(F2)
# plt.legend()
# plt.show()
# st.pyplot(plt.gcf())

sns.set(style="whitegrid")
plt.figure(figsize=(10, 8))
sns.scatterplot(x=uploaded_df[F1], y=uploaded_df[F2], hue=uploaded_df.iloc[:, -1].name, 
                data=uploaded_df, palette={0: 'blue', 1: 'red'}, marker='o')
cmap_boundary = ListedColormap(['blue', 'red'])
plt.xlabel(F1)
plt.ylabel(F2)
plt.legend(title=uploaded_df.iloc[:, -1].name, loc='upper right', labels=['0', '1'])
plt.show()
st.pyplot(plt.gcf())