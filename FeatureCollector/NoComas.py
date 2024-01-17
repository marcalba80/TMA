import sys

# File paths
input_file_path = sys.argv[1]  # Replace with your input file path
output_file_path = sys.argv[2]  # Replace with your desired output file path

# Open the input file and process each line
with open(input_file_path, 'r') as file_in, open(output_file_path, 'w') as file_out:
    for line in file_in:
        # Strip quotes from the start and end of the line
        processed_line = line.strip().strip('"')
        # Write the processed line to the output file
        file_out.write(processed_line + '\n')

print("File processing completed.")
