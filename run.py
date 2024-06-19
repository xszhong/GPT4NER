import numpy as np
from algorithms import *
from data import *
from tqdm import tqdm
import time
import pandas as pd
import openai
import stanza
from seqeval.metrics import f1_score
def eval_dataset(val, model, algorithm, sleep_between_queries=None, print_every=100):
    algorithm.set_model_fn(model)
    stanza.download('en')
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos')
    preds, truths = [], []
    text=""
    for i, info in tqdm(enumerate(val.iterrows()), total=len(val)):
        index, q = info
        para = q['text']
        para_list=para.split(' ')
        entities = q['entities']
        prompt_para=""
        doc = nlp(para)
        for sentence in doc.sentences:
            for word in sentence.words:
                prompt_para+=word.text+'/'+word.xpos+' '
        algorithm.set_para(para,prompt_para)
        text = ""
        if sleep_between_queries is not None:
            time.sleep(sleep_between_queries)
        types = None
        flag = False
        while not flag:
            try:
                span_pred = algorithm.perform_span(verbose=False)
                preds.append(span_pred)
                span_truth = q['exact_types']
                truths.append(span_truth)
                flag = True
            except openai.error.RateLimitError:
                time.sleep(0.5)
        if print_every is not None:
            if i % print_every == 0:
                f1_micro = f1_score(truths, preds, average="micro")
                f1_macro = f1_score(truths, preds, average="macro")
                print(f"Iteration {i}: micro f1: {f1_micro}, macro f1: {f1_macro}")
        for i in range(len(para_list)):
            text += str(para_list[i]) + "\t" + str(span_truth[i]) + "\t" + str(span_pred[i]) + '\n'
        text += '\n'
        print(text)
        filepath="results/results_ontonotes_nopos.txt"
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(text)
    f1_micro = f1_score(truths, preds, average="micro")
    f1_macro = f1_score(truths, preds, average="macro")
    print(f"Finally: micro f1: {f1_micro}, macro f1: {f1_macro}")
    return f1_micro, f1_macro


def complete_eval(dataset, model, algorithm, n_runs=2, sleep_between_queries=None, limit=None):
    micros = []
    macros = []
    for i in range(n_runs):
        if limit is not None:
            small_dataset = dataset.sample(limit)
        else:
            small_dataset = dataset
        print(small_dataset['text'])
        f1_micro, f1_macro = eval_dataset(small_dataset, model, algorithm, sleep_between_queries=sleep_between_queries)
        micros.append(f1_micro)
        macros.append(f1_macro)
    micros = np.array(micros)
    macros = np.array(macros)
    return micros, macros


def eval_conll(model, algorithm, n_runs=2, sleep_between_queries=None, limit=None, exemplar=True, coT=True,
                        defn=True, tf=True, pos=True, **kwargs):
    config = ConllConfig()
    algorithm.split_phrases = False
    config.set_config(algorithm, exemplar=exemplar, coT=coT, defn=defn, tf=tf, pos=pos)
    conll = load_conll2003()
    return complete_eval(conll, model, algorithm, n_runs=n_runs, sleep_between_queries=sleep_between_queries,
                         limit=limit)

def eval_Ontonotes_ten(model, algorithm, n_runs=2, sleep_between_queries=None, limit=None, exemplar=True, coT=True,
                        defn=True, tf=True, pos=True, **kwargs):
    config = Ontonotes_ten_Config()
    algorithm.split_phrases = False
    config.set_config(algorithm, exemplar=exemplar, coT=coT, defn=defn, tf=tf, pos=pos)
    ontonotes_ten = load_Ontonotes_ten()
    return complete_eval(ontonotes_ten, model, algorithm, n_runs=n_runs, sleep_between_queries=sleep_between_queries,
                         limit=limit)

def run(dataset="conll", subdataset=None, gpt=True, exemplar=True, coT=True, defn=True, tf=True, pos=False, name_meta=""):
    res_path = "results"
    #gpt_limit = 10
    gpt_limit = None #all test set
    gpt_nruns = 1
    other_limit = 100
    other_nruns = 2
    Algorithm_class = Algorithm

    if dataset == "conll":
        eval_fn = eval_conll
    elif dataset == "Ontonotes_ten":
        eval_fn = eval_Ontonotes_ten
    if gpt:
        model = OpenAIGPT()
        micros, macros = eval_fn(model, Algorithm_class(), n_runs=gpt_nruns,
                                                      sleep_between_queries=model.seconds_per_query,
                                                      limit=gpt_limit,
                                                      exemplar=exemplar, coT=coT, defn=defn, tf=tf, pos=pos,
                                                      add_info=subdataset)
    print(f"Final Results For {name_meta} | {dataset} {'('+subdataset+')' if subdataset is not None else ''}) "
          f"|CoT {coT} | Exemplar {exemplar} (tf {tf}) |Defn {defn}")
    print(f"Micro f1_means: {micros.mean()}")
    print(f"Micro f1_stds: {micros.std()}")
    print(f"Macro f1_means: {macros.mean()}")
    print(f"Macro f1_stds: {macros.std()}")
    return micros, macros


if __name__ == "__main__":
    from models import OpenAIGPT
    run(dataset="Ontonotes_ten")
    #run(dataset="conll")


