import argparse
import csv
import re

languages = ["Swedish", "Danish", "Bokmål", "Icelandic", "Faroese", "English", "Welsh", "German", "Old English ", "Arabic"] # Selected languages
l = "swe,cym,ang,ara,dan,eng,nob,isl,fao,deu"

def create_datafiles(y_data, x_data, filename_y, filename_x, lang_codes):
    """ 
    Opens and reads the labels and data and writes them to a new file
    """
    with open(y_data, encoding='utf-8') as labels:
        with open(x_data, encoding='utf-8') as data:
            for label_line, data_line in zip(labels, data):
                file_x = open(filename_x,"a+", encoding='utf-8')
                file_y = open(filename_y,"a+", encoding='utf-8') 
                label_line = re.sub(r'(\n)', '', label_line)  # Remove newlines
                data_line = re.sub(r'(\n)', '', data_line)
                if label_line in lang_codes:  # Get selected languages
                    if len(data_line) < 100:  # remove any instances that have less than 100 characters
                        return
                    data_line = [ch for ch in data_line]
                    data_line = data_line[:100]  # ignore everything past the first 100 characters
                    data_line = ("").join(data_line) + "\n"
                    #print(data_line)
                    file_x.write(data_line)
                    file_y.write(label_line + "\n")   
    
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

#if __name__ == "__main__":

parser = argparse.ArgumentParser(description="Creates training and test data for selected languages.")

parser.add_argument("-s", "--showall", metavar="s", dest="lang_labels", type=str, help="Lists all available languages and their language codes for the selected file.")
parser.add_argument("y_file", type=str, nargs='?', help="File containing training or test data language labels.")
parser.add_argument("x_file", type=str, nargs='?', help="File contaning training or test data sentences.")
parser.add_argument("y_new", type=str, nargs='?', help="File name of the new language labels")
parser.add_argument("x_new", type=str, nargs='?', help="File name of the new language data")
parser.add_argument("languages", type=str, nargs='?', help="Selected languages codes separated by space.")

args = parser.parse_args()

if args.lang_labels:
    print_language_labels(args.lang_labels)
    exit(1)

languages = args.languages.split(",")
print(len(languages), "languages selected.")
if len(languages) < 10:
    exit("Error: less than 10 languages selected")
