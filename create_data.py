import argparse
import csv
import re

#languages = ["Swedish", "Danish", "Bokmål", "Icelandic", "Faroese", "English", "Welsh", "German", "Old English ", "Arabic"] # Selected languages
#l = "swe,cym,ang,ara,dan,eng,nob,isl,fao,deu"

def create_datafiles(y_data, x_data, filename_y, filename_x, lang_codes):
    """ 
    Opens and reads the labels and data and writes them to a new file
    """
    with open(y_data, encoding='utf-8') as labels:
        with open(x_data, encoding='utf-8') as data:
            for index, (label_line, data_line) in enumerate(zip(labels, data)):
                file_x = open(filename_x,"a+", encoding='utf-8')
                file_y = open(filename_y,"a+", encoding='utf-8') 
                label_line = re.sub(r'(\n)', '', label_line)  # Remove newlines
                data_line = re.sub(r'(\n)', '', data_line)
                if label_line in lang_codes:  # Get selected languages
                    if len(data_line) < 100:  # remove any instances that have less than 100 characters
                        break
                    data_line = [ch for ch in data_line]
                    data_line = data_line[:100]  # ignore everything past the first 100 characters
                    data_line = ("").join(data_line)
                    #print(data_line)
                    file_x.write(data_line)
                    file_y.write(label_line)
                    file_x.write("\n")
                    file_y.write("\n")

    file_x.close()
    file_y.close() 

def print_language_labels(filename):
    """ 
    Shows available languages corresponding labels 
    """
    with open(filename, encoding='utf-8') as file_labels:
        csv_reader = csv.reader(file_labels, delimiter=';')
        next(csv_reader) # Skip first row
        print(("{:<25s} {:>10s}").format("Language", "Code"))
        print("--------------------------------")
        for row in csv_reader:
            lang = row[1]
            code = row[0]
            print(("{:<25s} {:>10s}").format(lang, code))
 

parser = argparse.ArgumentParser(description="Creates training and test data for selected languages.")

parser.add_argument("-s", "--showall", metavar="s", dest="lang_labels", type=str,help="Lists all available languages and their language codes for the selected file.")
parser.add_argument("-y_file", type=str, dest="y_file", nargs='?', help="File containing training or test data language labels.")
parser.add_argument("-x_file", type=str, dest="x_file",nargs='?', help="File contaning training or test data sentences.")
parser.add_argument("-y_new", type=str,dest="y_new", nargs='?', help="File name of the new language labels")
parser.add_argument("-x_new", type=str, dest="x_new", nargs='?', help="File name of the new language data")
parser.add_argument("-languages", type=str,dest="languages", nargs='?', help="Selected languages codes separated by comma.")

args = parser.parse_args()

if args.lang_labels:
    print_language_labels(args.lang_labels)
    exit(1)

languages = args.languages.split(",")
# print(len(languages), "languages selected.")
# if len(languages) < 10:
#     exit("Error: less than 10 languages selected")

print("Writing data to file...")
create_datafiles(args.y_file, args.x_file, args.y_new, args.x_new, args.languages)
print("Finished writing data")

#python create_data.py -y_file data/y_train.txt -x_file data/x_train.txt -y_new y_small.txt -x_new x_small.txt -languages swe,ara,isl

#python create_data.py -y_file data/y_train.txt -x_file data/x_train.txt -y_new y_train.txt -x_new x_train.txt -languages swe,cym,ang,ara,dan,eng,nob,isl,fao,deu
#python create_data.py -y_file data/y_test.txt -x_file data/x_test.txt -y_new y_test.txt -x_new x_test.txt -languages swe,cym,ang,ara,dan,eng,nob,isl,fao,deu
