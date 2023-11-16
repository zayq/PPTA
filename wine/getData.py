import pandas as pd



def getData():
    data = pd.read_csv('wine.csv')
    wines = data.to_dict(orient='records')
    return wines