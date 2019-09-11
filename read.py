import csv
import re

def read_labels(file_dir, filename, selected_languages):
    """
    Gets the language labels from the selected labels (file type csv)
    """
    selected_codes = []  # Selected language codes
    with open(file_dir + filename, encoding='utf-8') as file_labels:
        csv_reader = csv.reader(file_labels, delimiter=';')
        next(csv_reader)  # Skip first row

        #print(("{:<25s} {:>10s}").format("Language", "Code"))
        #print("--------------------------------")

        for row in csv_reader:
            lang = row[1]
            code = row[0]
            #print(("{:<25s} {:>10s}").format(lang, code))

            if lang in selected_languages:
                selected_codes.append(code)
        print("Selected language codes: ", selected_codes)
    return sorted(selected_codes)


def open_datafiles(file_dir, lang_labels, lang_data, selected_codes):
    """ Opens and reads the labels and data """
    x_data = []
    y_data = []
    # Read labels
    with open(file_dir + lang_labels, encoding='utf-8') as labels:
        with open(file_dir + lang_data, encoding='utf-8') as data:
            for label_line, data_line in zip(labels, data):

                label_line = re.sub(r'(\n)', '', label_line)  # Remove newlines
                data_line = re.sub(r'(\n)', '', data_line)
                if label_line in selected_codes:  # Get selected languages

                    y = label_line
                    x = data_line

                    x_data.append(x)
                    y_data.append(y)

    #for y,x in  zip( y_train[:10], x_train[:10]):
    #print(y, "::::", x)

    return x_data, y_data


#read_labels(dirr, "labels.csv", languages)
#x_train, y_train = open_datafiles(dirr, "y_train.txt", "x_train.txt")
#x_test, y_test = open_datafiles(dirr, "y_test.txt", "x_test.txt")
