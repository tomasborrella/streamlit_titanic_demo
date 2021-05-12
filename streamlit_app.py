import streamlit as st
import pickle

# load the model from disk
model_filename = 'trained_models/titanic_model_logistic_regression.pkl'
loaded_model = pickle.load(open(model_filename, 'rb'))
# load the scaler from disk
scaler_filename = 'trained_models/titanic_scaler_logistic_regression.pkl'
loaded_scaler = pickle.load(open(scaler_filename, 'rb'))

st.title("Titanic")
st.write("""
         Modelo de predicción de supervivencia a partir de los datos de [Kaggle](https://www.kaggle.com/c/titanic).
         """)

name = st.text_input("Nombre del pasajero")
sex = st.selectbox("Sexo", options=['Hombre', 'Mujer'])
age = st.slider("Edad", 1, 100, 1)
p_class = st.selectbox("Clase", options=['Primera clase', 'Segunda clase', 'Tercera clase'])

sex = 0 if sex == 'Hombre' else 1

f_class = 1 if p_class == 'Primera clase' else 0
s_class = 1 if p_class == 'Segunda clase' else 0
t_class = 1 if p_class == 'Tercera clase' else 0

input_data = loaded_scaler.transform([[sex, age, f_class, s_class, t_class]])
prediction = loaded_model.predict(input_data)
predict_probability = loaded_model.predict_proba(input_data)

if name != "":
    if prediction[0] == 1:
        st.write(f":+1: El pasajero {name} **SI** hubiera sobrevivido con una probabilidad \
        de {round(predict_probability[0][1] * 100, 3)}%")
    else:
        st.write(f":cry: El pasajero {name} **NO** hubiera sobrevivido con una probabilidad \
        de {round(predict_probability[0][0] * 100, 3)}%")
