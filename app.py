import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Funksiya: Modeli yükləmək
def load_model():
    with open('mod.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Funksiya: Skaler yükləmək (əgər modeli öyrədərkən skaler istifadə olunubsa)
def load_scaler():
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler

# Funksiya: İstifadəçi məlumatlarını emal etmək
def preprocess_user_input(user_data):
    # İstifadəçi məlumatlarını numpy array-ə çevir
    input_data = np.array(list(user_data.values())).reshape(1, -1)
    return input_data

# Funksiya: Dil seçimi
def translate_text(language, texts):
    translations = {
        'az': {
            'title': 'Şəkərli Diabet Xəstəlik Riskini Proqnozlaşdırma',
            'sidebar_header': 'Daxil edilən məlumatlar',
            'age': 'Yaşınızı daxil edin',
            'bmi': 'Bədən kütlə indeksi (BMI)',
            'hba1c': 'HbA1c Səviyyəsi (mmol/mol)',
            'blood_glucose': 'Qan Qlükoza Səviyyəsi (mg/dL)',
            'hypertension': 'Hipertoniya?',
            'heart_disease': 'Ürək xəstəliyi?',
            'gender': 'Cinsiyyət',
            'smoking': 'Siqaret Çəkmə Tarixi',
            'prediction_button': 'Proqnoz',
            'result': 'Nəticə',
            'high_risk': 'Yüksək Risk',
            'medium_risk': 'Orta Risk',
            'low_risk': 'Aşağı Risk',
            'high_risk_text': '⚠️ Şəkərli diabet riski yüksəkdir, mütləq həkimə müraciət edin!',
            'medium_risk_text': '⚠️ Şəkərli diabet riski orta səviyyədədir, ehtiyatlı olun.',
            'low_risk_text': '😊 Şəkərli diabet riski aşağıdır, sağlamlığınıza diqqət edin.',
            'gender_male': 'Kişi',
            'gender_female': 'Qadın',
            'smoking_current': 'Siqaret çəkirəm',
            'smoking_past': 'Əvvəl çəkmişəm',
            'smoking_never': 'Siqaret çəkmirəm'
        },
        'en': {
            'title': 'Diabetes Risk Prediction',
            'sidebar_header': 'Input Data',
            'age': 'Enter your age',
            'bmi': 'Body Mass Index (BMI)',
            'hba1c': 'HbA1c Level (mmol/mol)',
            'blood_glucose': 'Blood Glucose Level (mg/dL)',
            'hypertension': 'Hypertension?',
            'heart_disease': 'Heart Disease?',
            'gender': 'Gender',
            'smoking': 'Smoking History',
            'prediction_button': 'Predict',
            'result': 'Result',
            'high_risk': 'High Risk',
            'medium_risk': 'Medium Risk',
            'low_risk': 'Low Risk',
            'high_risk_text': '⚠️ High risk of diabetes, please consult a doctor!',
            'medium_risk_text': '⚠️ Medium risk of diabetes, be cautious.',
            'low_risk_text': '😊 Low risk of diabetes, take care of your health.',
            'gender_male': 'Male',
            'gender_female': 'Female',
            'smoking_current': 'I currently smoke',
            'smoking_past': 'I used to smoke',
            'smoking_never': 'I do not smoke'
        }
    }
    return translations[language][texts]

# Main Streamlit tətbiqi
def main():
    # Dil seçimi
    language = st.selectbox('Zəhmət olmasa, dil seçin.', ['az', 'en'], format_func=lambda x: 'Azərbaycan dili' if x == 'az' else 'English')
    
    # Modeli və scaler-i yüklə
    model = load_model()
    scaler = load_scaler()

    # Streamlit Tətbiq Başlığı
    st.title(translate_text(language, 'title'))

    # İstifadəçi Giriş Sahələri
    st.sidebar.header(translate_text(language, 'sidebar_header'))
    
    age = st.sidebar.number_input(translate_text(language, 'age'), 0, 200, 40)
    bmi = st.sidebar.number_input(translate_text(language, 'bmi'), 0.0, 50.0, 28.0)
    
    # HbA1c və Qan Qlükoza səviyyələri slider ilə təqdim olunur
    hba1c = st.sidebar.slider(translate_text(language, 'hba1c'), 0.0, 12.0, 5.5, step=0.1, format="%.1f")
    blood_glucose = st.sidebar.slider(translate_text(language, 'blood_glucose'), 0, 400, 100, step=1)
    
    hypertension = st.sidebar.selectbox(translate_text(language, 'hypertension'), 
                                        ["No", "Yes"] if language == 'en' else ["Yox", "Bəli"])
    heart_disease = st.sidebar.selectbox(translate_text(language, 'heart_disease'), 
                                         ["No", "Yes"] if language == 'en' else ["Yox", "Bəli"])
    gender = st.sidebar.selectbox(translate_text(language, 'gender'), 
                                  ['Male', 'Female'] if language == 'en' else ['Kişi', 'Qadın'])
    
    # Siqaret çəkmə tarixçəsi seçimi
    smoking_history = st.sidebar.selectbox(translate_text(language, 'smoking'), 
                                           ['I do not smoke', 'I currently smoke', 'I used to smoke'] if language == 'en' 
                                           else ['Siqaret çəkmirəm', 'Siqaret çəkirəm', 'Əvvəl çəkmişəm'])

    # İstifadəçi məlumatlarını dictionary kimi formalaşdır
    user_data = {
        'age': age,
        'bmi': bmi,
        'hba1c_level': hba1c,
        'blood_glucose_level': blood_glucose,
        'hypertension': 1 if hypertension == ("Yes" if language == 'en' else "Bəli") else 0,
        'heart_disease': 1 if heart_disease == ("Yes" if language == 'en' else "Bəli") else 0,
        'gender_male': 1 if gender == ('Male' if language == 'en' else 'Kişi') else 0,
        'smoking_current': 1 if smoking_history == ('I currently smoke' if language == 'en' else 'Siqaret çəkirəm') else 0,
        'smoking_past': 1 if smoking_history == ('I used to smoke' if language == 'en' else 'Əvvəl çəkmişəm') else 0
    }

    # Məlumatları oxunaqlı formada göstərmək
    st.write(translate_text(language, "sidebar_header") + ":")
    st.markdown(f"""
    **{translate_text(language, 'age')}**: {age}  
    **BMI**: {bmi}  
    **{translate_text(language, 'hba1c')}**: {hba1c} mmol/mol  
    **{translate_text(language, 'blood_glucose')}**: {blood_glucose} mg/dL  
    **{translate_text(language, 'hypertension')}**: {"Yes" if hypertension == "Yes" else "No"}  
    **{translate_text(language, 'heart_disease')}**: {"Yes" if heart_disease == "Yes" else "No"}  
    **{translate_text(language, 'gender')}**: {"Male" if gender == "Male" else "Female"}  
    **{translate_text(language, 'smoking')}**: {smoking_history}
    """)

    # Proqnoz düyməsi
    if st.button(translate_text(language, 'prediction_button')):
        # İstifadəçi məlumatlarını emal et
        processed_input = preprocess_user_input(user_data)

        # Məlumatları miqyasla
        processed_input_scaled = scaler.transform(processed_input)

        # Model ilə proqnoz et
        prediction = model.predict_proba(processed_input_scaled)

        # Nəticəni göstərmək
        st.subheader(translate_text(language, 'result'))

        # Risk ehtimalını faizlə hesablayın
        risk_percentage = prediction[0][1] * 100  # Diabet olma ehtimalını faizlə ifadə et

        # Göstərilən nəticəni yaradın
        if risk_percentage >= 70:
            st.markdown(
                f"""
                <div style='border:2px solid #FF6347; padding: 10px; background-color:#FFE4E1;'>
                <h3>{translate_text(language, 'high_risk')}</h3>
                <p>{translate_text(language, 'high_risk_text')}</p>
                <p style='font-size: 36px;'><strong>{risk_percentage:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
        elif risk_percentage >= 40:
            st.markdown(
                f"""
                <div style='border:2px solid #FFA500; padding: 10px; background-color:#FFF8DC;'>
                <h3>{translate_text(language, 'medium_risk')}</h3>
                <p>{translate_text(language, 'medium_risk_text')}</p>
                <p style='font-size: 36px;'><strong>{risk_percentage:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                f"""
                <div style='border:2px solid #32CD32; padding: 10px; background-color:#F0FFF0;'>
                <h3>{translate_text(language, 'low_risk')}</h3>
                <p>{translate_text(language, 'low_risk_text')}</p>
                <p style='font-size: 36px;'><strong>{risk_percentage:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
