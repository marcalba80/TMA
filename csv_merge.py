import csv

with open('malicious.out.csv', 'r') as input_file, \
    open('a_malicious.out.csv', 'r') as input_file2, \
    open('s_malicious.out.csv', 'r') as input_file3, \
    open('benign.out.csv', 'r') as input_file4, \
    open('featuresff.csv', 'w', newline='') as output_file:
    # Create a CSV reader object
    reader = csv.reader(input_file)
    reader2 = csv.reader(input_file2)
    reader3 = csv.reader(input_file3)
    reader4 = csv.reader(input_file4)
    # Create a CSV writer object
    writer = csv.writer(output_file)
    # Loop through each row in the input file
    for row in reader:
        # Write the updated row to the output file
        writer.writerow(row)
    next(reader2, None)
    for row in reader2:
        # Write the updated row to the output file
        writer.writerow(row)
    next(reader3, None)
    for row in reader3:
        # Write the updated row to the output file
        writer.writerow(row)
    next(reader4, None)
    for row in reader4:
        # Write the updated row to the output file
        writer.writerow(row)