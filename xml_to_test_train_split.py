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


def extract_text2(elem):
    title = []
    abstract = []
    body = []
    for children in elem.iter():
        for child in children.findall('title-group'):
            for child2 in child.iter():
                if (child2.text == 'Decision letter') | (child2.text == 'Author response'):
                    print('sub-title found')
                else:
                    title.append(child2.text)

                    for children in elem.iter():
                        for child in children.findall('abstract'):
                            for child2 in child.itertext():
                                abstract.append(child2)

                    for children in elem.iter():
                        for child in children.findall('body'):
                            for child2 in child.itertext():
                                body.append(child2)

    tit = ' '.join(title)
    abst = ' '.join(abstract)
    bod = ' '.join(body)
    return [clean_text(tit, title=False), clean_text(abst), clean_text(bod)]


if not os.path.exists("text/"):
    os.mkdir("text/")


def skip_exceptions(it):
    while True:
        try:
            yield next(it)
        except StopIteration:
            raise
        except Exception as e:
            logging.info('Skipping iteration because of exception {}'.format(e))


try:
    count = 0
    for evt, elem in skip_exceptions(iterparse('pmc_result_sm.xml')):  # , events=('start', 'end')):
        if elem.tag == 'article':
            try:
                output = extract_text2(elem)

                if (len(output[0]) < 50) | (len(output[1]) < 1):
                    print("too short or abstract error")

                else:
                    count += 1
                    print('Article found. Count = ' + str(count))
                    with open("text//title_to_text.txt", 'a+') as text_file:
                        text_file.write(output[0].lower().lstrip() + '\n')

                    with open("text//abstract_to_text.txt", 'a+') as text_file:
                        text_file.write(output[1].lower() + '\n')

                    with open("text//body_to_text.txt", 'a+') as text_file:
                        text_file.write(output[2] + '\n')

            except:
                print("Failed")
                continue
            elem.clear()

        else:
            None
except:
    print("Warning, xml extraction failed at COUNT = " + str(count) + ". Please check for errors in structure of xml file.")
    pass

with open('text//title_to_text.txt', 'r+') as f:
    titleread = f.readlines()

with open('text//abstract_to_text.txt', 'r+') as f:
    abstractread = f.readlines()

title = pd.DataFrame({'col': titleread})
article = pd.DataFrame({'col': abstractread})


length_art = int(len(title) * 0.8)

train_article = article[:length_art].values.tolist()
train_title = title[:length_art].values.tolist()
test_article = article[length_art:].values.tolist()
test_title = title[length_art:].values.tolist()


outputs = [train_article, train_title, test_title, test_article]

for value in outputs:
    print(len(value))

if (len(train_article) != len(train_title)) | (len(test_title) != len(test_article)):
    raise ValueError("Test or train data not the same length")


if not os.path.exists("input_data/"):
    os.mkdir("input_data/")


with open("input_data//train.article.txt", 'a+') as text_file:
    for value in train_article:
        text_file.write(value[0])

with open("input_data//train.title.txt", 'a+') as text_file:
    for value in train_title:
        text_file.write(value[0])

with open("input_data//valid.article.filter.txt", 'a+') as text_file:
    for value in test_article:
        text_file.write(value[0])

with open("input_data//valid.title.filter.txt", 'a+') as text_file:
    for value in test_title:
        text_file.write(value[0])
