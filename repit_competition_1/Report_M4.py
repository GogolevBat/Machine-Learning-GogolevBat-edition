import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import pandas as pd

def while_to_year_predict(df, scaler):
    rfr = RandomForestRegressor(n_estimators=100, random_state=0)
    lr = LinearRegression()

    # lr.fit(scaler.fit_transform(np.array(df.drop("year", axis=1))[:-1].T), np.array(df.drop("year", axis=1))[-1])
    #
    rfr.fit(scaler.fit_transform(np.array(df.drop("year", axis=1))[:-1].T), np.array(df.drop("year", axis=1))[-1])
    #
    rfr_list = list(rfr.predict(scaler.fit_transform(np.array(df.drop("year", axis=1))[1:].T)))
    #
    # lr_list = list(lr.predict(scaler.fit_transform(np.array(df.drop("year", axis=1))[1:].T))) + [max(df["year"].unique()) + 1]
    # df.loc[len(df)] = [round((rfr_list[n] + lr_list[n]) / 2, 3) for n in range(6)] + [max(df["year"].unique()) + 1]
    df.loc[len(df)] = rfr_list + [max(df["year"].unique()) + 1]#[round((rfr_list[n] + lr_list[n]) / 2, 3) for n in range(6)] + [max(df["year"].unique()) + 1]

    return df

# plt.plot(pd.concat([df_cluster_one_rfr,df_cluster_one_gbr,df_cluster_one_lr]).groupby("year").mean().reset_index()["year"], pd.concat([df_cluster_one_rfr,df_cluster_one_gbr,df_cluster_one_lr]).groupby("year").mean().reset_index()[column],color='orange',marker='o')
def model_prredict(df, year):
    scaler = StandardScaler()


    while True:
        df = while_to_year_predict(df, scaler)
        if (max(df["year"].unique())) == year:
            return df
def main():
    st.title("Прогнозирование данных такси")
    if st.button("Перейти в информационное окно", help ="Нажать два раза"):
        st.session_state["page"] = 0
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Первый кластер"):
            st.session_state["cluster"] = 1
    with col2:
        if st.button("Второй кластер"):
            st.session_state["cluster"] = 2
    with col3:
        if st.button("Третий кластер"):
            st.session_state["cluster"] = 3

    year = st.slider("Введите до какого года будет предсказание", min_value=2022, max_value=2050)

    if st.button("Предсказать ▶️"):
        if st.session_state["cluster"] not in [1,2,3]:
            st.markdown("### Не выбран кластер")
            return
        st.markdown(f"### с 2019 по {year}, кластер {st.session_state['cluster']}")
        if st.session_state["cluster"] == 1:
            df = pd.read_csv(
                "/Homeworks/Repit_per_competition/repit_competition_1/df_cluster_one.csv")
        elif st.session_state["cluster"] == 2:
            df = pd.read_csv(
                "/Homeworks/Repit_per_competition/repit_competition_1/df_cluster_two.csv")
        else:
            df = pd.read_csv(
                "/Homeworks/Repit_per_competition/repit_competition_1/df_cluster_three.csv")

        df = model_prredict(df, year)
        for column in df.drop("year",axis=1).columns:
            st.markdown("#### "+column)
            fig = plt.figure(figsize=(15, 6))
            plt.plot(df["year"], df[column], color='blue', marker='o')
            # plt.legend(["adsfvg","sf","gthh"])
            plt.xlabel("year")
            plt.ylabel(column)
            st.pyplot(fig)
        st.dataframe(df)
        pass

def info_menu():
    if st.button("Вернутся в главное меню"):
        st.session_state["page"] = 1

    col1, col2 = st.columns(2)
    with col1:
        st.image("https://image.openmoviedb.com/kinopoisk-images/10703959/e8a2be4c-bd59-4923-ade3-0c79aaf1865c/orig")
    with col2:
        st.text("""Данный сайт поможет спроогназировать данные про такси по выбраному периоду и кластреру
        
        Вы должны выбрать кластер кнопкой
        Ползунок с выбором года окончания периода
        
        После выбора параметров нажимаем на кнопку 'предсказать'
        
        Вам будут показаны графики с динамикой определеного значения по годам до выбраного окончания периода
        
        Наслаждайтесь;)""")


if "page" not in st.session_state:
    st.session_state["page"] = 1
if "cluster" not in st.session_state:
    st.session_state["cluster"] = 0

if st.session_state["page"] == 1:
    main()
else:
    info_menu()