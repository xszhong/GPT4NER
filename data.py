import os
import pandas as pd
from datasets import load_dataset
import re
import string

data_root = "dataset"


def get_row(func):
    def infunc(frame, i=None):
        if i is not None:
            frame = frame.loc[i, :]
        return func(frame, i=None)
    return infunc

def read_ob2(file_path):
    with open(file_path,encoding="utf-8") as file:
        lines = file.readlines()
    sentences = []
    entities = []
    types = []
    exact_types = []
    data = []
    sub_entities = []
    sub_types = {}
    sub_exact_types = []
    words = ""
    curr_entity = ""
    curr_type = None

    for i, line in enumerate(lines):
        if line.strip() == "" or line == "\n" or i == len(lines)-1:
            # save entity if it exists
            if curr_type is not None:
                sub_entities.append(curr_entity.strip())
                sub_types[curr_entity.strip()] = curr_type
                curr_entity = ""
                curr_type = None
            if words != "":
                sentences.append(words)
                entities.append(sub_entities)
                types.append(sub_types)
                exact_types.append(sub_exact_types)
                data.append([words, sub_entities, sub_types, sub_exact_types])
            sub_entities = []
            sub_types = {}
            sub_exact_types = []
            words = ""
            curr_entity = ""
            curr_type = None
        else:
            word, tag = line.split("\t")
            if words == "":
                words = word
            else:
                words = words + " " + word
            sub_exact_types.append(tag.strip())
            if tag.split() == "O" or "-" not in tag:  # if there was an entity before this then add it in full
                if curr_type is not None:
                    sub_entities.append(curr_entity.strip())
                    sub_types[curr_entity.strip()] = curr_type
                curr_entity = ""
                curr_type = None
            elif "B-" in tag or "I-" in tag:
                if "B-" in tag:
                    if curr_type is not None:
                        sub_entities.append(curr_entity.strip())
                        sub_types[curr_entity.strip()] = curr_type
                    curr_entity = word
                    curr_type = tag.split("-")[1].strip()
                else:  # I- in tag
                    if curr_type is None:
                        print(f"Should not be happening bug here")
                    curr_entity = curr_entity + " " + word
            else:
                main_type, subtype = tag.split("-")  # must assume that if curr_type is not None then its the same one because FewNERD doesn't contain B, I information
                if subtype.strip() == "government/governmentagency":
                    subtype = "government"
                if curr_type is None:
                    curr_entity = word
                    curr_type = main_type + "-" + subtype.strip()  # can change to make it subtype if we want
                else:
                    curr_entity = curr_entity + " " + word

    df = pd.DataFrame(columns=["text", "entities", "types", "exact_types"], data=data)
    #df = df.drop_duplicates(subset='text')
    #df.to_csv('conll_test_lastest.csv', index=False)
    return df

def load_conll2003(conll_path = "dataset/conll_test.ibo2"):
    return read_ob2(conll_path)

def load_Ontonotes_ten(Ontonotes_ten_path="dataset/Ontonotes5.0_ten_test.ibo2"):
    return read_ob2(Ontonotes_ten_path)

def scroll(dataset, start=0, exclude=None):
    cols = dataset.columns
    for i in range(start, len(dataset)):
        s = dataset.loc[i]
        print(f"Item: {i}")
        for col in cols:
            if exclude is not None:
                if col in exclude:
                    continue
            print(f"{col}")
            print(s[col])
            print(f"XXXXXXXXXXXXXXX")
        inp = input("Continue?")
        if inp != "":
            return
