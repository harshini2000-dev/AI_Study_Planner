import os
import pandas as pd
file_name = "student_performance_dataset.csv"
sub_directory = "LSTM_Web_App"
base_directory = os.getcwd() # Get the current working directory

# Construct the full dynamic path
dynamic_path = os.path.join(base_directory, file_name)
print(dynamic_path)

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
csv_path = os.path.join(script_dir, "student_performance_dataset.csv")
print(csv_path)

df = pd.read_csv(dynamic_path)
# try:
#     with open(dynamic_path, 'r') as file:
#         # Process the file, e.g., read text data
#         content = file.read()
#         print(content)
# except FileNotFoundError:
#     print(f"Error: File not found at {dynamic_path}")
# except Exception as e:
#     print(f"An error occurred: {e}")