import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import plotly.express as px
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
try:
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
    R['intercept'] = fit(X, y)[-1]
    R1 = pd.DataFrame(list(R.items()))
    st.write(R1)

    st.write("""
    ### Для построения графика выберите 2 фичи
    """)
    st.subheader('Фича 1')
    F1 = st.selectbox(uploaded_df.columns.tolist()[0], uploaded_df.columns.tolist()[:-1])

    st.subheader('Фича 2')
    F2 = st.selectbox(uploaded_df.columns.tolist()[1], uploaded_df.columns.tolist()[:-1])


    X_d = pd.DataFrame(data=X)
    X_d.columns = uploaded_df.iloc[:, [0,-2]].columns
    X_d[uploaded_df.iloc[:, -1].name] = y.values

    st.subheader('Прекрасный график')
    # plt.scatter(X_d[X_d.iloc[:, -1] == 0][F1], X_d[X_d.iloc[:, -1] == 0][F2], color='blue', label='0')
    # plt.scatter(X_d[X_d.iloc[:, -1] == 1][F1], X_d[X_d.iloc[:, -1] == 1][F2], color='pink', label='1')
    xx = np.linspace(X_d[F1].min(), X_d[F1].max())
    yy = - ((xx * R[F1] + R['intercept']))/R[F2]
    # plt.plot(xx, yy, color='red', label='LogReg')
    # plt.xlabel(F1)
    # plt.ylabel(F2)
    # plt.legend()
    # plt.show()
    # st.pyplot(plt.gcf())
    fig = px.scatter(X_d, x=X_d[F1], y=X_d[F2], color=X_d.iloc[:, -1], hover_name=X_d.iloc[:, -1])
    fig.add_trace(px.line(x=xx, y=yy).data[0])
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
except:
    st.write("""
\n\n\n
# Дружочек, подготовь и загрузи Dataset!
\n\n # Если загрузил и видишь это, обрати внимание на требования к Dataset.
""")
