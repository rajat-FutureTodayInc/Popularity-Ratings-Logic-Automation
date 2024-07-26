#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
from datetime import datetime
import ast
#sudo pip3 install scikit-learn
from sklearn.preprocessing import MultiLabelBinarizer

# Function to automate your Excel files
def automate_excel(data_file, average_rating_file, automate):
    # Read the files (CSV or Excel)
    if data_file.name.endswith('.csv'):
        data = pd.read_csv(data_file)
    elif data_file.name.endswith(('.xls', '.xlsx')):
        data = pd.read_excel(data_file)

    if average_rating_file.name.endswith('.csv'):
        Average_Rating = pd.read_csv(average_rating_file)
    elif average_rating_file.name.endswith(('.xls', '.xlsx')):
        Average_Rating = pd.read_excel(average_rating_file)


    coeff = pd.DataFrame()
    desc_table = pd.DataFrame()
    
    with open('coefficients.pkl', 'rb') as f3:
        coeff = pickle.load(f3)
    
    with open('describe_table.pkl', 'rb') as f4:
        desc_table = pickle.load(f4)
    
    def changePos(df, col_name, col_pos):
        cols = list(df.columns)
        cols.insert(col_pos, cols.pop(cols.index(col_name)))
        df = df[cols]
        return df
    
    def desc_func(df_encoded):
    
        if(automate == 'Static'):
            return desc_table

        desc = df_encoded.describe()

        interQuartile = (desc.loc['75%'] - desc.loc['25%']).to_frame().T
        interQuartile.index = ['interQuartile']

        Upper_limit = (desc.loc['75%'] + 1.5 * interQuartile.loc['interQuartile']).to_frame().T
        Upper_limit.index = ['Upper_limit']

        Lower_limit = (desc.loc['25%'] - 1.5 * interQuartile.loc['interQuartile']).clip(lower=0).to_frame().T
        Lower_limit.index = ['Lower_limit']

        desc = pd.concat([desc, interQuartile, Upper_limit, Lower_limit])

        return desc
    
    def updateDesc():
    
        df1 = Internal_Factors.iloc[:, 0:5]
        df2 = External_Factors.iloc[:, 8:10]
        df3 = Normalised.iloc[:, [0, 5, 8, 10, 12, 14, 16]]
        df4 = Score.iloc[:, [6, 8]]
        df = pd.concat([df1, df2, df3, df4], axis = 1)

        desc = desc_func(df)

        with open('describe_table.pkl', 'wb') as file:
            pickle.dump(desc, file)
            
            
    def TakeLog(df, col_name):
        return np.log2(df[col_name] + 1.0001)
    
    
    def Normalise_func(df, col_name):
        desc1 = desc_func(df)
        df1 = (df[col_name]-desc1.loc['min'][col_name])/(desc1.loc['max'][col_name]-desc1.loc['min'][col_name])
        return df1
    
    def step_func(df, coeff_df, col_name1, col_name2):
        coefficient = coeff_df[col_name2][0]
        new_column = np.where(
            df[col_name1] <= 1000, 
            1, 
            1 + ((coefficient - 1) * (df[col_name1] - 1001) / (500000 - 1001))
        )
        return new_column
    
    def linear_reg(df, coeff):
        df_ans = pd.DataFrame(0, index=np.arange(df.shape[0]), columns=['Result'])
        for column in df.columns:
                df_ans['Result'] += df[column] * coeff[column][0]

        return df_ans
    
    def Score_Scaling(df, col_name):
        df_ans = pd.DataFrame()
        df_ans['Result'] = Normalise_func(df, col_name)
        df_ans['Result'] = df_ans['Result'] * (100-40)
        df_ans['Result'] = df_ans['Result'] + 40
        return df_ans['Result']
    
    
    def divReg(df, coefficient):
        df_ans = pd.DataFrame(0, index=np.arange(df.shape[0]), columns=['Result'])
        for col in df.columns:
            df_ans['Result'] = df_ans['Result'] + coefficient[col][0]/df[col]
        return df_ans
    
    def update_scores(row):
        if pd.isna(row['External Score']):
            if row['Views/Day in a month'] >= 35:
                row['Internal_Factors'] = 1
                row['External_Factors'] = 0
                row['External Score'] = 0
            #elif row['Views/Day in a month'] >= 5:
             #   row['External Score'] = row['Average of Popularity Rating']
        return row

    def update_scores_scaled(row):
        if pd.isna(row['External Score Scaled 1']):
            if row['Views/Day in a month'] >= 5 and row['Views/Day in a month'] < 35:
                row['External Score Scaled 1'] = row['Average of Exnternal Score Scaled 1']
        return row
    
    def update_popularityRating(row):
        if pd.isna(row['Popularity Rating']):
            if row['Views/Day in a month'] < 5:
                if row['Days Live on Channel'] > 60:
                    row['Popularity Rating'] = 20
                else:
                    row['Popularity Rating'] = row['Average of Popularity Rating']
        return row
    
    Internal_Factors = data.iloc[:, 7:13]
    
    External_Factors = data.iloc[:, 13:]
    
    def calculate_Combined_Star_Meter(row):
    
        # Sum of starMeters
        total_sum = row[['StarMeter1', 'StarMeter2', 'StarMeter3']].sum()

        # Count of non-zero values in T, U, and V
        non_zero_count = (row[['StarMeter1', 'StarMeter2', 'StarMeter3']] != 0).sum()

        # Calculate the result
        if non_zero_count == 0:
            return np.nan  # Avoid division by zero
        else:
            return total_sum / non_zero_count

    External_Factors['Combined_StarMeter'] = External_Factors.apply(calculate_Combined_Star_Meter, axis=1)

    External_Factors = changePos(External_Factors, 'Combined_StarMeter', 5)
    
    import ast
    #sudo pip3 install scikit-learn
    from sklearn.preprocessing import MultiLabelBinarizer
    
    def one_hot_encode_genres(df, genres_column):
        # Check if the column exists in the DataFrame
        if genres_column not in df.columns:
            raise ValueError(f"Column '{genres_column}' does not exist in the DataFrame.")

        # Convert string representation of lists into actual lists
        try:
            df[genres_column] = df[genres_column].apply(ast.literal_eval)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Error in converting column '{genres_column}' to list: {e}")

        # Initialize the MultiLabelBinarizer
        mlb = MultiLabelBinarizer()

        # Fit and transform the specified column
        one_hot_encoded_genres = mlb.fit_transform(df[genres_column])

        # Create a DataFrame with the one-hot encoded columns using genre names as column headers
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded_genres, columns=mlb.classes_, index=df.index)

        # Define the list of desired columns
        desired_columns = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 
                           'Family', 'Horror', 'Romance', 'SciFi', 'Thriller', 'Western']

        # Ensure all desired columns are present, filling missing columns with zeros
        for col in desired_columns:
            if col not in one_hot_encoded_df.columns:
                one_hot_encoded_df[col] = 0

        # Filter the DataFrame to include only the desired columns
        one_hot_encoded_df = one_hot_encoded_df[desired_columns]

        df = pd.concat([df, one_hot_encoded_df], axis=1)

        return df

    External_Factors = one_hot_encode_genres(External_Factors, 'IMDB_Genre')

    
    def add_genre_intersection_column(df, first_genre, second_genre):
        new_column_name = f'{first_genre}-{second_genre}'
        df = df.assign(**{new_column_name: df[first_genre] & df[second_genre]})
        return df

    External_Factors = add_genre_intersection_column(External_Factors, 'Thriller', 'Horror')
    External_Factors = add_genre_intersection_column(External_Factors, 'Romance', 'Comedy')
    External_Factors = add_genre_intersection_column(External_Factors, 'Horror', 'Comedy')
    External_Factors = add_genre_intersection_column(External_Factors, 'Romance', 'Drama')
    External_Factors = add_genre_intersection_column(External_Factors, 'Action', 'Crime')
    External_Factors = add_genre_intersection_column(External_Factors, 'Crime', 'Drama')
    
    
    Outlier_Check = pd.DataFrame()

    def Compare_upper_lower_limit(df, col_name):
        desc = desc_func(df)
        result= ((df[col_name] > desc.loc['Upper_limit', col_name]) | (df[col_name] < desc.loc['Lower_limit', col_name])).astype(int)
        return result

    Outlier_Check['Views'] = Compare_upper_lower_limit(Internal_Factors, 'Total video Views')
    Outlier_Check['Search Launches'] = Compare_upper_lower_limit(Internal_Factors, 'Search Launches')
    
    
    Outlier = pd.DataFrame()

    def check_value_exists(df, col_name, value, s_row, e_row):
        result = df[col_name].iloc[s_row:e_row].sum()
        return 0 if(result == 0) else 1

    results = {col: check_value_exists(Outlier_Check, col, 1, 0, 17) for col in Outlier_Check.columns}
    Outlier = pd.DataFrame([results.values()], columns=results.keys())

    Outlier['Views'] = 1
    Outlier['Search Launches'] = 1

    
    Normalised = pd.DataFrame()

    Normalised['Log(Views)'] = TakeLog(Internal_Factors, 'Total video Views')

    Normalised['Normalised Log(Views)'] = Normalise_func(Normalised, 'Log(Views)')

    Normalised['Views'] = Normalise_func(Internal_Factors, 'Total video Views')

    Normalised['Completion %'] = Normalise_func(Internal_Factors, 'Completion %')

    Normalised['AWT (mins)'] = Normalise_func(Internal_Factors, 'AWT (mins)')

    Normalised['Log(Search Launches)'] = TakeLog(Internal_Factors, 'Search Launches')

    Normalised['Normalised Log(Search Launches)'] = Normalise_func(Normalised, 'Log(Search Launches)')

    Normalised['Search Launches'] = Normalise_func(Internal_Factors, 'Search Launches')

    Normalised['Step(Movie_Meter)'] = step_func(External_Factors, coeff, 'Movie_Meter', 'Movie_Meter')

    Normalised['Movie_Meter'] = Normalise_func(Normalised, 'Step(Movie_Meter)')

    Normalised['Step(StarMeter1)'] = step_func(External_Factors, coeff, 'StarMeter1', 'StarMeter1')

    Normalised['StarMeter1'] = Normalise_func(Normalised, 'Step(StarMeter1)')

    Normalised['Step(Combined_StarMeter)'] = step_func(External_Factors, coeff, 'Combined_StarMeter', 'Combined_StarMeter')

    Normalised['Combined_StarMeter'] = Normalise_func(Normalised, 'Step(Combined_StarMeter)')

    Normalised['IMDB_Rating_Corrected'] = (((External_Factors['Votes']/(External_Factors['Votes']+41))*External_Factors['IMDB_Rating']) + ((41/(External_Factors['Votes']+41))*5.18))

    Normalised['IMDb_Rating'] = Normalise_func(Normalised, 'IMDB_Rating_Corrected')

    Normalised['Log(Votes)'] = TakeLog(External_Factors, 'Votes')

    Normalised['Votes'] = Normalise_func(Normalised, 'Log(Votes)')

    Normalised['popularity_TMDb'] = Normalise_func(External_Factors, 'popularity_TMDb')

    Normalised['no. of streaming OTTs'] = Normalise_func(External_Factors, 'no. of streaming OTTs')
    
    Weightage = pd.DataFrame()

    Weightage = Weightage.assign(Internal_Factors = np.where(Internal_Factors['Views/Day in a month'] > 35, 0.75, 
                np.where(Internal_Factors['Views/Day in a month'] < 5, 0, 
                (((Internal_Factors['Views/Day in a month'] - 5) / (35 - 5)) * (0.75 - 0)) + 0)))

    Weightage['External_Factors'] = 1 - Weightage['Internal_Factors']
    
    
    Final_Normalised_Internal_Factors = pd.DataFrame()

    Final_Normalised_Internal_Factors['Total video Views'] = np.where(Outlier['Views'] == 1, Normalised['Normalised Log(Views)'], Normalised['Views'])

    Final_Normalised_Internal_Factors = pd.concat([Final_Normalised_Internal_Factors, Normalised.iloc[:, 3:5]], axis =1)

    Final_Normalised_Internal_Factors['Search Launches'] = np.where(Outlier['Search Launches'] == 1, Normalised['Normalised Log(Search Launches)'], Normalised['Search Launches'])
    
    
    
    Final_Normalised_External_Factors = pd.DataFrame()

    Final_Normalised_External_Factors = Normalised.iloc[:, [8, 10, 12, 15, 17, 18, 19]]

    Final_Normalised_External_Factors.columns = ['Movie_Meter', 'StarMeter1', 'Combined_StarMeter', 'IMDB_Rating', 'Votes','popularity_TMDb','no. of streaming OTTs']

    Final_Normalised_External_Factors = pd.concat([Final_Normalised_External_Factors, External_Factors.iloc[:, 10:29]], axis = 1)

    
    Score = pd.DataFrame()

    Score['Internal Score'] = linear_reg(Final_Normalised_Internal_Factors, coeff)

    Score['Internal Score Scaled 1'] = Score_Scaling(Score, 'Internal Score')

    Score['External Score'] = divReg(Final_Normalised_External_Factors.iloc[:, 0:3], coeff) + linear_reg(Final_Normalised_External_Factors.iloc[:, 3:], coeff)

    Score['External Score1'] = divReg(Final_Normalised_External_Factors.iloc[:, 1:3], coeff)

    Score['External Score'] = np.where(Final_Normalised_External_Factors['Animation'] == 1,
                                       Score['External Score']-Score['External Score1'],
                                       Score['External Score'])

    Score.drop(columns=['External Score1'], inplace=True)

    Score = pd.concat([data['UID'], Internal_Factors['Views/Day in a month'], Weightage, Score], axis = 1)

    Score = pd.merge(Score, Average_Rating, on = 'UID', how = 'left')

    Score = changePos(Score, 'Average of Popularity Rating', 1)

    Score = changePos(Score, 'Average of Exnternal Score Scaled 1', 2)

    Score = Score.apply(update_scores, axis=1)

    Score['External Score Scaled 1'] = Score_Scaling(Score, 'External Score')

    Score = Score.apply(update_scores_scaled, axis=1)

    Score['Flat-Fee?'] = data['Flat-Fee?']
    
    Score['Days Live on Channel'] = data['DAR End Date'] - data['Submission Date']

    Score['Days Live on Channel'] = Score['Days Live on Channel'].dt.days

    Score['Final Score'] =  (Score['Internal Score Scaled 1']*Score['Internal_Factors']+
                            Score['External Score Scaled 1']*Score['External_Factors']+
                            Score['Flat-Fee?']*2)

    Score['Final Score Scaled'] = np.where(
        (Score['Days Live on Channel'] > 60) & (Internal_Factors['Views/Day in a month'] < 20),
        np.minimum((((Score['Final Score'] - 40) / (100 - 40)) * (105 - 20)) + 20, 64),
        (((Score['Final Score'] - 40) / (100 - 40)) * (105 - 20)) + 20
    )
    
    
    
    Final_Rank = pd.DataFrame()

    Final_Rank = Score.iloc[:, [0, 1, 2, 3, 11, 7, 9, 4, 5, 12]]

    Final_Rank.columns = ['UID', 'Average of Popularity Rating', 'Average of Exnternal Score Scaled 1', 'Views/Day in a month', 'Days Live on Channel', 'Internal Score', 'External Score', '%Internal Factors', '%External Factors', 
                          'Final Score']

    Final_Rank.loc[:, 'Rank'] = Final_Rank['Final Score'].rank(method='min', ascending=False) - 1

    Final_Rank = changePos(Final_Rank, 'Rank', 0)

    Final_Rank['Popularity Rating'] = Score['Final Score Scaled'].round().clip(lower=20, upper=100)

    Final_Rank = Final_Rank.apply(update_popularityRating, axis=1)

    Final_Rank['Popularity Rating'] = Final_Rank['Popularity Rating'].fillna(20)

    Final_Rank['Stars on UI'] = np.where(
                                    Final_Rank['Popularity Rating']+20 >= 80,
                                    5,
                                    np.where(
                                        Final_Rank['Popularity Rating']+20 >= 60,
                                        4,
                                        np.where(
                                            Final_Rank['Popularity Rating']+20 >= 40,
                                            3,
                                            np.where(
                                                Final_Rank['Popularity Rating']+20 >= 20,
                                                2,
                                                np.where(
                                                    Final_Rank['Popularity Rating']+20 >= 0,
                                                    1,
                                                    -1
                                                )
                                            )
                                        )
                                    )
                                )

    
    if(automate == 'Dynamic'):
        updateDesc()

    Final_Rank = pd.concat([data.iloc[:, [1, 5, 6]], Final_Rank], axis = 1)
        
    
    return Final_Rank
    
      
# Streamlit app
st.title('Popularity Ratings Logic Automation')

st.write("Please upload the files for processing:")

data_file = st.file_uploader('Upload your data file (CSV or Excel)', type=['csv', 'xls', 'xlsx'], key='data_file')
average_rating_file = st.file_uploader('Upload your Average Rating file (CSV or Excel)', type=['csv', 'xls', 'xlsx'], key='average_rating_file')

if data_file and average_rating_file:
    st.write('All files uploaded successfully!')

    # Create two columns for the buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button('Automate Statically'):
            # Automate the uploaded files
            automate = 'Static'
            result_df = automate_excel(data_file, average_rating_file, automate)

            # Display the automated dataframe
            st.write('Automated DataFrame:')
            st.dataframe(result_df)

            # Provide download link for the processed file
            output = io.StringIO()
            result_df.to_csv(output, index=False)
            processed_data = output.getvalue()

            st.download_button(
                label='Download Processed File',
                data=processed_data,
                file_name='processed_file.csv',
                mime='text/csv'
            )

    with col2:
        if st.button('Automate Dynamically'):
            # Automate the uploaded files
            automate = 'Dynamic'
            result_df = automate_excel(data_file, average_rating_file, automate)

            # Display the automated dataframe
            st.write('Automated DataFrame:')
            st.dataframe(result_df)

            # Provide download link for the processed file
            output = io.StringIO()
            result_df.to_csv(output, index=False)
            processed_data = output.getvalue()

            st.download_button(
                label='Download Processed File',
                data=processed_data,
                file_name='processed_file.csv',
                mime='text/csv'
            )
            
    with col3:
        if st.button("Reset Describe Table"):
            
            with open('describe_table_original.pkl', 'rb') as f4:
                desc1 = pickle.load(f4)
            
            with open('describe_table.pkl', 'wb') as file:
                pickle.dump(desc1, file)

else:
    st.write('Please upload both the files.')


# In[ ]:




