import csv
import json

# שם קובץ ה-CSV
csv_file_path = "example_test_data.csv"

# שם קובץ ה-JSON שיווצר
json_file_path = "output.json"

# קריאת השורה הראשונה והפיכתה ל-JSON
with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    first_row = next(csv_reader)  # קריאת השורה הראשונה

    # כתיבה לקובץ JSON
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(first_row, json_file, indent=4)

print(f"JSON file created successfully: {json_file_path}")
