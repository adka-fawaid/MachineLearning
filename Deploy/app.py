import streamlit as st
import pandas as pd
import pickle

# Load the pipeline and model
def load_model():
    with open('scaler.sav', 'rb') as file:
        scaler = pickle.load(file)
    
    with open('label_encoders.sav', 'rb') as file:
        le_dict = pickle.load(file)

    with open('svm_model.sav', 'rb') as file:
        clf = pickle.load(file)
        
    with open('columns.sav', 'rb') as file:
        columns = pickle.load(file)
    
    with open('metrics.sav', 'rb') as file:
        metrics = pickle.load(file)
    
    return scaler, le_dict, clf, columns, metrics

scaler, le_dict, clf, columns, metrics = load_model()

# Main function to run the Streamlit app
def main():
    st.title('Klasifikasi Feedback Pelanggan Untuk Memahami Sentimen menggunakan algoritma Support Vector Machine (SVM)')

    # Input form for manual data entry
    st.write("Modelling by:")
    st.write("Mohamad Adzka Fawaid")
    st.write("A11.2022.14656")
    st.write("Masukan Data Baru:")
    age = st.number_input('Age', min_value=0, max_value=120, value=25)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Prefer not to say'])
    occupation = st.selectbox('Occupation', ['Student', 'Employee', 'Self Elmpoyee', 'House Wife'])
    monthly_income = st.number_input('Monthly Income', min_value=0, value=5000)
    family_size = st.number_input('Family Size', min_value=1, value=4)
    edu_qualification = st.selectbox('Educational Qualifications', ['Graduate', 'Post Graduate', 'PhD'])

    # Create a dictionary with user input
    new_data = {
        "Age": [age],
        "Gender": [gender],
        "Marital Status": [marital_status],
        "Occupation": [occupation],
        "Monthly Income": [monthly_income],
        "Family size": [family_size],
        "Educational Qualifications": [edu_qualification],
    }

    # Convert to DataFrame
    new_data_df = pd.DataFrame(new_data)

    # Apply label encoding to new data
    for column in new_data_df.columns:
        if new_data_df[column].dtype == 'object':
            if column in le_dict:
                # Handle unseen labels
                new_data_df[column] = new_data_df[column].apply(lambda x: le_dict[column].transform([x])[0] if x in le_dict[column].classes_ else -1)
            else:
                new_data_df[column] = new_data_df[column].apply(lambda x: -1)

    # Reorder columns to match the training data
    new_data_df = new_data_df.reindex(columns=columns, fill_value=0)

    # Standardize new data
    new_data_prep = scaler.transform(new_data_df)

    hasil_klasifikasi = ''
    # Predict using trained model
    if st.button('Klasifikasikan'):
        prediction = clf.predict(new_data_prep)
        if(prediction[0]==1):
            hasil_klasifikasi = "Sentimen Pelanggan Memberikan : Negative"
        else :
            hasil_klasifikasi = "Sentimen Pelanggan Memberikan : Positive"
        st.success(hasil_klasifikasi)   
        
        # Display model evaluation metrics
        st.write("Hasil Evaluasi Model menggunakan Metrics:")
        st.write(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
        st.write("Classification Report:")
        st.text(metrics['classification_report'])
        st.write("Confusion Matrix:")
        st.write(metrics['confusion_matrix'])

if __name__ == '__main__':
    main()
