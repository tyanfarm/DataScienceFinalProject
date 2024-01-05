import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import bentoml
import os

#################################################################################################################
#==================================DATA COLLECTING===============================================================
#################################################################################################################

def collect_data(cur_page, items_per_page):
    # Khai báo sẵn các mảng để đưa vào dataframe
    name = []
    region = []
    country = []
    overall = []
    AcademicReputation = []
    EmployerReputation = []
    FacultyStudentRatio = []
    CitationsPerFaculty = []
    InternationalFacultyRatio = []
    InternationalStudentsRatio = []
    InternationalResearchNetwork = []
    EmploymentOutcomes = []
    Sustainability = []
    
    url = "https://www.topuniversities.com/rankings/endpoint"

    querystring = {"nid":"3897789","page":f"{cur_page}","items_per_page":f"{items_per_page}","tab":"indicators","region":"","countries":"","cities":"","search":"","star":"","sort_by":"overallscore","order_by":"asc","program_type":""}

    payload = ""
    headers = ""

    # Gửi yêu cầu kết nối
    response = requests.request("GET", url, data=payload, headers=headers, params=querystring)
    
    # Nếu trả về Status 200 --> Success
    if response.status_code == 200:
        text = response.json()
        
        list_uni = text['score_nodes']  

        for i in range(items_per_page):
            # Tên đại học
            name.append(list_uni[i]['title'])

            # Châu lục
            region.append(list_uni[i]['region'])
            
            # Vị trí
            country.append(list_uni[i]['country'])
            
            # Điểm trung bình
            overall.append(list_uni[i]['overall_score'])
            
            # Danh sách điểm chi tiết
            scores = list_uni[i]['scores']
            
            AcademicReputation.append(scores[0]['score'])
            EmployerReputation.append(scores[1]['score'])
            FacultyStudentRatio.append(scores[2]['score'])
            CitationsPerFaculty.append(scores[3]['score'])
            InternationalFacultyRatio.append(scores[4]['score'])
            InternationalStudentsRatio.append(scores[5]['score'])
            InternationalResearchNetwork.append(scores[6]['score'])
            EmploymentOutcomes.append(scores[7]['score'])
            Sustainability.append(scores[8]['score'])
                
    data = pd.DataFrame({
        "Institution Name": name,
        "Region": region,
        "Country": country, 
        "Overall": overall,
        "Academic Reputation": AcademicReputation,
        "Employer Reputation": EmployerReputation,
        "Faculty Student Ratio": FacultyStudentRatio,
        "Citations Per Faculty": CitationsPerFaculty,
        "International Faculty Ratio": InternationalFacultyRatio,
        "International Students Ratio": InternationalStudentsRatio,
        "International Research Network": InternationalResearchNetwork,
        "Employment Outcomes": EmploymentOutcomes,
        "Sustainability": Sustainability,
    })
    
    return data

def Generate_Universities_Dataset(quantity, items_per_page):
    index = 0
    num_uni = 0
    data = pd.DataFrame()
    
    # Nếu số lượng vượt quá yêu cầu thì kết thúc
    while num_uni < quantity:
        data = pd.concat([data, collect_data(index, items_per_page)], ignore_index=True)
        
        num_uni += items_per_page
        index += 1

    return data

uni_df = Generate_Universities_Dataset(1200, 15)
file_path = "universities.csv"
uni_df.to_csv(file_path, index=False)


#################################################################################################################
#==================================DATA PREPROCESSING============================================================
#################################################################################################################


# Try to read the CSV file
try:
    uni_df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
except pd.errors.EmptyDataError:
    print(f"File '{file_path}' is empty.")
except pd.errors.ParserError as e:
    print(f"Error parsing CSV: {e}")


dtypes = None

# We format some column has int64 dtype
if uni_df is not None:
    for col in uni_df.columns:
        if uni_df[col].dtype == 'int64':
            uni_df[col] = uni_df[col].astype('float64')

if uni_df is not None:
    dtypes = uni_df.dtypes


def filling_missing_value(df: pd.DataFrame) -> pd.DataFrame:    
    numeric_cols = df.select_dtypes(include=['number']).columns
    cols_to_fill = [col for col in numeric_cols if col != 'Overall']
    
    df[cols_to_fill] = df[cols_to_fill] = df[cols_to_fill].fillna(df[cols_to_fill].mean())
    
    return df

uni_df = filling_missing_value(uni_df)

uni_df = uni_df.drop(uni_df[uni_df['Region'] == 'False'].index)

#################################################################################################################
#==================================DATA MODELING=================================================================
#################################################################################################################

y = np.array(uni_df.iloc[0:602,3])
X = np.array(uni_df.iloc[0:602,[4, 5, 6, 7, 8, 9, 10,11, 12]].values)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

regressor = LinearRegression()
regressor.fit(X_train,y_train)
nn_y_pred = regressor.predict(X_test)
print(f"R2 score is: {r2_score(y_test,nn_y_pred)}")

saved_model = bentoml.sklearn.save_model("universities_rank_regression",regressor)
print(f"Model saved: {saved_model}")

# Delete file
try:
    os.remove(file_path)
except OSError as e:
    print("Error: %s : %s" % (file_path, e.strerror))
