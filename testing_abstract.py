import re
import pandas as pd
import os


def clean_text(in_string, title=False):
    if title == True:
        in_string2 = in_string.lower()
        result = re.sub(' +', ' ', in_string2)
        result2 = re.sub(r'[\n]', ' ', result)
        return result2.lstrip()

    else:
        in_string2 = in_string.lower()
        remove_brackets = re.sub(r'\([^)]*\)', '', in_string)
        result0 = re.sub(r'[0-9]+[\.][0-9]+', ' number', remove_brackets)
        result1 = re.sub(r'[^a-zA-Z\s\.]', " ", result0)
        result2 = re.sub(r'[\n]', ' ', result1)
        result3 = re.sub(r'[\.]', ' .', result2)
        result4 = re.sub(' +', ' ', result3)
        result5 = re.sub(r'\s[^ai\.]\s', ' ', result4)
        return result5.lstrip()


with open('text//title_to_text.txt', 'r+') as f:
    titleread = f.readlines()

with open('text//abstract_to_text.txt', 'r+') as f:
    abstractread = f.readlines()


title = pd.DataFrame({'col': titleread})
article = pd.DataFrame({'col': abstractread})


test_article = article.values.tolist()
test_title = title.values.tolist()


outputs = [clean_text(test_title[0][0]).lower(), clean_text(test_article[0][0]).lower()]


if (len(test_title) != len(test_article)):
    raise ValueError("Test or train data not the same length")


if not os.path.exists("input_data/"):
    os.mkdir("input_data/")

if (os.path.exists("input_data//valid.article.filter.txt") == True) & (os.path.exists("input_data//valid.title.filter.txt") == True):
    os.remove("input_data//valid.article.filter.txt")
    os.remove("input_data//valid.title.filter.txt")
    print('Removed old article testing txt file')

else:
    None

with open("input_data//valid.article.filter.txt", 'a+') as text_file:
    text_file.write(outputs[1])

with open("input_data//valid.title.filter.txt", 'a+') as text_file:
    text_file.write(outputs[0])
