import xml.etree.ElementTree as ET
from xml.etree.ElementTree import iterparse
import re
import os
import pandas as pd
import csv


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


df = pd.read_csv("test2.csv")

titles = []
for value in df['TI']:
    titles.append(clean_text(str(value)).lower())

abstract = []
for value in df['AB']:
    abstract.append(clean_text(str(value)).lower())

if not os.path.exists("input_data/"):
    os.mkdir("input_data/")


with open("input_data//valid.article.filter.txt", 'a+') as text_file:
    for value in abstract:
        text_file.write(value + '\n')

with open("input_data//valid.title.filter.txt", 'a+') as text_file:
    for value in titles:
        text_file.write(value + '\n')
