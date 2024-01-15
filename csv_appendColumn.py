import csv

with open('stateful_light-benign.csv', 'r') as input_file, open('stateful_light-benign.out.csv', 'w', newline='') as output_file:
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
        
with open('stateful_light-text.csv', 'r') as input_file, open('stateful_light-text.out.csv', 'w', newline='') as output_file:
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