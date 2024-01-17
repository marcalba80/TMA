import csv

with open('benign.csv', 'r') as input_file, open('benign.out.csv', 'w', newline='') as output_file:
    # Create a CSV reader object
    reader = csv.reader(input_file)
    # Create a CSV writer object
    writer = csv.writer(output_file)
    # Loop through each row in the input file
    
    for index, row in enumerate(reader):
        # Add a new column to the row
        if index == 0:
            new_column_value = 'malicious'
        else:
            new_column_value = 0
        row.append(new_column_value)
        # Write the updated row to the output file
        writer.writerow(row)
        
with open('s_malicious.csv', 'r') as input_file, open('s_malicious.out.csv', 'w', newline='') as output_file:
    # Create a CSV reader object
    reader = csv.reader(input_file)
    # Create a CSV writer object
    writer = csv.writer(output_file)
    # Loop through each row in the input file
    for index, row in enumerate(reader):
        # Add a new column to the row
        if index == 0:
            new_column_value = 'malicious'
        else:
            new_column_value = 1
        row.append(new_column_value)
        # Write the updated row to the output file
        writer.writerow(row)        