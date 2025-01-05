import streamlit as st
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

# Sarlavha
st.title("Qandli diabetni aniqlash tizimi")

# Modelni yuklash
try:
    model_path = 'decision_tree_model.pkl'
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model fayli topilmadi. Iltimos, 'decision_tree_model.pkl' faylini tekshiring va qaytadan urinib ko'ring.")
    st.stop()

# Foydalanuvchi ma'lumotlarini kiritish formasi
st.header("Foydalanuvchi ma'lumotlarini kiriting")

pregnancies = st.number_input("Homiladorliklar soni")
glucose = st.number_input("Qondagi glyukoza miqdori (mg/dL)")
blood_pressure = st.number_input("Qon bosimi (mmHg)")
skin_thickness = st.number_input("Teri qalinligi (mm)")
insulin = st.number_input("Insulin darajasi (IU/mL)")
bmi = st.number_input("Tana massasi indeksi (BMI)")
diabetes_pedigree = st.number_input("Diabet tarixi ko'rsatkichi")
age = st.number_input("Yoshingiz")

# Kiruvchi ma'lumotlarni birlashtirish
user_data = pd.DataFrame({
    'Homiladorliklar': [pregnancies],
    'Glyukoza': [glucose],
    'Qon bosimi': [blood_pressure],
    'Teri qalinligi': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'Diabet tarixi': [diabetes_pedigree],
    'Yosh': [age]
})

# Model yordamida bashorat qilish
if st.button("Diabetni aniqlash"):
    try:
        prediction = model.predict(user_data)[0]
        prediction_prob = model.predict_proba(user_data)[0]

        # Natijalarni ko'rsatish
        if prediction == 1:
            st.error("Sizda diabet aniqlanish ehtimoli yuqori!")
        else:
            st.success("Sizda diabet aniqlanmadi.")

        st.write("Ehtimollik darajasi:")
        st.metric(label="Diabet ehtimolligi", value=f"{prediction_prob[1] * 100:.2f}%")
        st.metric(label="Sog'lom ehtimolligi", value=f"{prediction_prob[0] * 100:.2f}%")

        # Foydalanuvchi ma'lumotlarini chiqarish
        st.subheader("Foydalanuvchi kiritgan ma'lumotlar")
        st.write(user_data)
    except Exception as e:
        st.error(f"Bashorat qilishda xatolik yuz berdi: {e}")
