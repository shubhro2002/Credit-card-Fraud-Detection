def missing(df):
    '''
    Find the percenatge of values missing in each of the columns
    '''
    print('Missing values in each column: ')
    n = len(df)
    for col in df.columns:
        m = (df[col].isnull().sum()/n)*100
        print(f'{col}: {m}')