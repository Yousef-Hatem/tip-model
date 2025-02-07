import sklearn
import pickle as pk
import streamlit as st

fileModel = open("model.pkl", "rb")
model = pk.load(fileModel)

fileScaler = open("scaler.pkl", "rb")
scaler = pk.load(fileScaler)

def convert(x):
  if x == "Female" or x == "No":
    return 0
  if x == "Male" or x == "Yes":
    return 1
  return x

def getDummies(value: str, values: list):
  list = []
  for v in values:
    if value == v:
      list.append(True)
    else:
      list.append(False)
  return list

def scalerTransform(num: float):
  return scaler.transform([[num]])[0][0]

days = ["Thur", "Fri", "Sat", "Sun"]
times = ["Lunch", "Dinner"]

st.markdown(f"""
    <h1>Tip model, obtained a score of <span style="color:red;">83</span></h1>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
  totalBill = scalerTransform(st.number_input("Total Bill"))
  smoker = convert(st.selectbox("Smoker", ["Yes", "No"]))
  day = getDummies(st.selectbox("Day", days), days)
  
with col2:
  sex = convert(st.selectbox("Sex", ["Male", "Female"]))
  size = int(st.number_input("Size"))
  time = getDummies(st.selectbox("Time", times), times)

submit = st.button("Submit")

if submit:
  input = [totalBill, sex, smoker, size] + day + time
  tip = round(model.predict([input])[0], 2)
  st.markdown(f"""
    <h1>Tip: <span style="color:forestgreen;">{tip}</span></h1>
""", unsafe_allow_html=True)
