import csv

with open('stateful_light-text.out.csv', 'r') as input_file, \
    open('stateful_light-benign.out.csv', 'r') as input_file2, \
    open('featuresf.csv', 'w', newline='') as output_file:
    # Create a CSV reader object
    reader = csv.reader(input_file)
    reader2 = csv.reader(input_file2)
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