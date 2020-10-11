import streamlit as st
import pickle
import numpy as np
import json
import time
model=pickle.load(open('model.pickle','rb'))

f=open('columns.json')
column_name_data=json.load(f)

column_names=column_name_data["data_columns"]
area_names=column_name_data["area_type_column_names"]
location_names=column_name_data["location_column_names"]

def predict_forest(oxygen,humidity,temperature):
    input=np.array([[oxygen,humidity,temperature]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)


def predicting_using_model(total_sqft, no_of_bathrooms, no_of_balcony, area_type, bhk, location):
    col_names_in_numpy_format=np.array(column_names)
    loc_index_of_area_type = np.where(col_names_in_numpy_format == area_type)[0][0]
    loc_index_of_location = np.where(col_names_in_numpy_format == location)[0][0]
    predicting_values = np.zeros(len(column_names))
    predicting_values[0] = total_sqft
    predicting_values[1] = no_of_bathrooms
    predicting_values[2] = no_of_balcony
    predicting_values[3] = bhk

    if loc_index_of_area_type > 0:
        predicting_values[loc_index_of_area_type] = 1
    if loc_index_of_location > 0:
        predicting_values[loc_index_of_location] = 1


    #return predicting_values
    return  round(model.predict([predicting_values])[0])*100000


def main():
    st.title("Price Prediction")
    html_temp = """
    <div style="background-color:#e63946 ;padding:10px">
    <h2 style="color:white;text-align:center;">House Price Prediction ML App </h2>
    </div><br><br>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    #oxygen = st.text_input("Oxygen","Type Here")
    #humidity = st.text_input("Humidity","Type Here")
    #temperature = st.text_input("Temperature","Type Here")
    total_sqft = st.number_input("total_sqft", value=500)
    no_of_bathroom = st.number_input("no_of_bathroom",value=1)
    no_of_balcony = st.number_input("no_of_balcony", value=1)
    bhk = st.slider("BHK Number", 0,100,0,1)
    area_type=st.selectbox('area_type',area_names)
    location = st.selectbox('location', location_names)
    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> Your forest is safe</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> Your forest is in danger</h2>
       </div>
    """

    if st.button("Predict"):
        bar=st.progress(0)
        for i in range(100):
            bar.progress(i+1)
            time.sleep(0.01)
        output=predicting_using_model(total_sqft, no_of_bathroom, no_of_balcony, area_type, bhk, location)
        st.success(f'The predicted house price is {output} Indian rupees')



if __name__=='__main__':
    main()