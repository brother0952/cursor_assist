import pandas as pd

# List of cities to filter
citys = ['上海', '北京', '广州', '深圳', '杭州', '南京', '武汉', '西安', '天津', '成都', '重庆']

def filter_cities_data(input_file='processed.csv', output_file='filtered_cities.csv'):
    """
    Filter data for specified cities from the input CSV file and save to a new CSV file
    
    Parameters:
    input_file (str): Path to the input CSV file
    output_file (str): Path to the output CSV file
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Convert 时间 column from float to integer
    df['时间'] = pd.to_numeric(df['时间'], errors='coerce').astype('Int64')
    
    # Convert city data to numeric (in case they are read as strings)
    for city in citys:
        if city in df.columns:
            df[city] = pd.to_numeric(df[city], errors='coerce')
    
    # Select only the time column and the specified cities
    columns_to_select = ['时间'] + citys
    filtered_df = df[columns_to_select]
    
    # Save the filtered data to a new CSV file
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")
    print(f"Selected columns: {columns_to_select}")
    return filtered_df

if __name__ == "__main__":
    # Filter the data and save to a new CSV file
    filtered_data = filter_cities_data()
    print("\nFirst few rows of filtered data:")
    print(filtered_data.head())
    print("\nData types:")
    print(filtered_data.dtypes)