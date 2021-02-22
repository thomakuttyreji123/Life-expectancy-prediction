import numpy as np 
import pickle 
import streamlit as st 
import pandas as pd 


model = pickle.load(open('D:\my_folders\Git_local_repo\Hackathone-Life_Expectancy_Prediction\life_expectancy_model.pkl','rb'))  

def welcome():
    return "# Life Expectancy Prediction" 

def predict(Adult_Mortality,
       alcohol,bmi,
       under_five_deaths,Polio,hiv,thinness_1_19_years,schooling,Income_composition):

       prediction  = model.predict([[Adult_Mortality,
       alcohol,bmi,
       under_five_deaths,Polio,hiv,thinness_1_19_years,schooling,Income_composition]])

       print(prediction)
       return prediction
def main():
    st.title(" Life Expectancy Prediction")

    Adult_Mortality = st.number_input('adult mortality rate',)
    bmi = st.number_input('bmi',)
    alcohol = st.number_input('alcohol consumption rate',)
    under_five_deaths = st.number_input('uner_five_death_rate',)
    Polio = st.number_input('polio_rate',)
    hiv = st.number_input('hiv_rate',)
    thinness_1_19_years = st.number_input('thinness_1_19_years',)
    schooling = st.number_input('schooling',) 
    Income_composition = st.number_input('income_composition',)

    result = ""
    if st.button("Predict"):
        result = predict(Adult_Mortality,
       alcohol,bmi,
       under_five_deaths,Polio,hiv,thinness_1_19_years,schooling,Income_composition)
    st.success('The output is {} '.format(result))

    if st.button("about"):
        st.text("something related to this preditied value and model")   


if __name__ == "__main__":
    main()



