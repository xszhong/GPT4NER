import string
import utils
from utils import AnswerMapping
from nltk.corpus import stopwords
from models import OpenAIGPT

class BaseAlgorithm:
    defn = "An entity is an object, place, individual, being, title, proper noun or process that has a distinct and " \
           "independent existence. The name of a collection of entities is also an entity. Adjectives, verbs, numbers, " \
           "adverbs, abstract concepts are not entities. Dates, years and times are not entities"

    chatbot_init = "You are an entity recognition system. "

    # if [] = n then there are O(n^2) phrase groupings

    def __init__(self, model_fn=None, split_phrases=True, identify_types=True):
        self.defn = self.defn
        self.para = None
        self.prompt_para = None
        self.model_fn = model_fn
        self.split_phrases = split_phrases
        self.exemplar_task = None
        self.format_task = None
        self.whole_task = None
        self.pos = None
        self.identify_types = identify_types

    def set_para(self, para, prompt_para):
        self.para = para
        self.prompt_para = prompt_para

    def set_model_fn(self, model_fn):
        self.model_fn = model_fn

    @staticmethod
    def clean_output(answers, typestrings=None):
        if typestrings is None:
            answers = list(set(answers))
            for trivial in ["", " ", ".", "-"] + stopwords.words('english'):
                while trivial in answers:
                    answers.remove(trivial)
        else:
            new_answers = []
            new_typestrings = []
            for i, ans in enumerate(answers):
                if ans in new_answers:
                    continue
                if ans in ["", " ", ".", "-"] + stopwords.words('english'):
                    continue
                new_answers.append(ans)
                new_typestrings.append(typestrings[i])
        for i in range(len(answers)):
            ans = answers[i]
            if "(" in ans:
                ans = ans[:ans.find("(")]
            ans = ans.strip().strip(''.join(string.punctuation)).strip()
            answers[i] = ans
        if typestrings is None:
            return answers
        else:
            return answers, typestrings


class Algorithm(BaseAlgorithm):
    def perform_span(self, verbose=False):
        assert self.identify_types and not self.split_phrases
        answers, typestrings, metadata = self.perform(verbose=verbose, deduplicate=False)
        para = self.para.lower()
        para_words = para.split(" ")
        span_pred = ["O" for word in para_words]
        completed_answers = []
        for i, answer in enumerate(answers):
            answer = answer.strip().lower()  # take any whitespace out and lowercase for matching
            #print(answer)
            if "(" in answer:
                answer = answer[:answer.find("(")].strip()  # in case some type annotation is stuck here
            if(i < len(typestrings)):
                types = typestrings[i]
            else:
                types = ""
            if "(" in types and ")" in types:
                types = types[types.rfind("(") + 1:types.rfind(")")]
                #types = "ENTITY"
            else:
                continue
            exists = answer in para
            answer_multi_word = len(answer.split(" ")) > 1
            if not exists:
                continue
            if not answer_multi_word:
                if answer not in para_words:
                    continue
                multiple = para.count(answer) > 1

                if not multiple:  # easiest case word should be in para_words only once
                    index = para_words.index(answer)
                    if span_pred[index] == "O":
                        if "-" in types:  # then its FEWNERD
                            span_pred[index] = types
                        else:
                            span_pred[index] = "B-" + types
                else:  # must find which occurance this is
                    n_th = completed_answers.count(answer.strip()) + 1
                    indexs = utils.find_nth_list_subset(para_words, answer, n_th)
                    #print(indexs)
                    for i, index in enumerate(indexs):
                        if span_pred[index] == "O":
                            if "-" in types:  # then its FEWNERD
                                span_pred[index] = types
                            else:
                                span_pred[index] = "B-" + types
                completed_answers.append(answer)
            else:
                answer_words = answer.split(" ")
                multiple = para.count(answer) > 1
                n_th = completed_answers.count(answer.strip()) + 1
                indexs = utils.find_nth_list_subset(para_words, answer, n_th)
                for i, index in enumerate(indexs):
                    end_index = index + len(answer_words)
                    if "-" in types:  # then its FEWNERD
                        span_pred[index] = types
                    else:
                        span_pred[index] = "B-" + types
                    for j in range(index+1, end_index):
                        if "-" in types:  # then its FEWNERD
                            span_pred[j] = types
                        else:
                            span_pred[j] = "I-" + types
                completed_answers.append(answer)
        return span_pred

    def perform(self, verbose=True, deduplicate=True):
        """

        :param model:
        :param paragraph:
        :return:
        """
        if isinstance(self.model_fn, OpenAIGPT):
            if not self.identify_types:
                if self.model_fn.is_chat():
                    answers, metadata = self.perform_chat_query(verbose=verbose)
                else:
                    answers, metadata = self.perform_single_query(verbose=verbose)
            else:
                if self.model_fn.is_chat():
                    answers, typestrings, metadata = self.perform_chat_query(verbose=verbose)
                else:
                    answers, typestrings, metadata = self.perform_single_query(verbose=verbose)
        else:
            if not self.identify_types:
                answers, metadata = self.perform_single_query(verbose=verbose)
            else:
                answers, typestrings, metadata = self.perform_single_query(verbose=verbose)
        if not self.identify_types:
            answers = list(set(answers))
        if self.split_phrases:
            new_answers = []
            if self.identify_types:
                new_typestrings = []
            for i, answer in enumerate(answers):
                if " " not in answer:
                    new_answers.append(answer)
                    if self.identify_types:
                        new_typestrings.append(typestrings[i])
                else:
                    minis = answer.split(" ")
                    for mini in minis:
                        new_answers.append(mini)
                        if self.identify_types:
                            new_typestrings.append(typestrings[i])
            answers = new_answers
            if self.identify_types:
                typestrings = new_typestrings
        if deduplicate:
            if self.identify_types:
                answers, typestrings = BaseAlgorithm.clean_output(answers, typestrings)
            else:
                answers = BaseAlgorithm.clean_output(answers)
        if not self.identify_types:
            return answers, metadata
        else:
            return answers, typestrings, metadata

    def perform_single_query(self, verbose=True):
        if self.exemplar_task is not None:
            if self.pos is not None:
                task = self.defn + "\n" + self.exemplar_task + f" {self.prompt_para} \nAnswer:"
            else:
                task = self.defn + "\n" + self.exemplar_task + f" {self.para} \nAnswer:"
            print(task)
            output = self.model_fn(task)
            final = AnswerMapping.exemplar_format_list(output, identify_types=self.identify_types, verbose=verbose)
        else:
            if self.pos is not None:
                task = self.defn + "\n" + self.format_task + f"\nParagraph: {self.prompt_para} \nAnswer:"
            else:
                task = self.defn + "\n" + self.format_task + f"\nParagraph: {self.para} \nAnswer:"
            print(task)
            output = self.model_fn(task)
            final = AnswerMapping.exemplar_format_list(output, identify_types=self.identify_types, verbose=verbose)
        if self.identify_types:
            final, typestrings = final
        filepath="results/output_ontonotes_nopos.txt"
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(output)
        if not self.identify_types:
            return final, output
        else:
            return final, typestrings, output

    def perform_chat_query(self, verbose=True):
        if self.exemplar_task is not None:
            system_msg = self.chatbot_init + self.defn + " " + self.whole_task
            msgs = [(system_msg, "system")]
            for exemplar in self.exemplars:
                if "Answer:" not in exemplar:
                    raise ValueError(f"Something is wrong, exemplar: \n{exemplar} \n Does not have an 'Answer:'")
                ans_index = exemplar.index("Answer:")
                msgs.append((exemplar[:ans_index+7].strip(), "user"))
                msgs.append((exemplar[ans_index+7:].strip(), "assistant"))
            msgs.append((f"\nParagraph: {self.prompt_para} \nAnswer:", "user"))
            output = self.model_fn(msgs)
            final = AnswerMapping.exemplar_format_list(output, identify_types=self.identify_types, verbose=verbose)
        else:
            system_msg = self.chatbot_init + self.defn + " " + self.format_task
            msgs = [(system_msg, "system"), (f"\nParagraph: {self.para} \nAnswer:", "user")]
            output = self.model_fn(msgs)
            final = AnswerMapping.exemplar_format_list(output, identify_types=self.identify_types, verbose=verbose)
        if self.identify_types:
            final, typestrings = final
        if not self.identify_types:
            return final, output
        else:
            return final, typestrings, output


class Config:
    cot_format = """
    Format: 
    
    1. First Candidate | True | Explanation why the word is an entity (entity_type)
    2. Second Candidate | False | Explanation why the word is not an entity (entity_type)
    """

    tf_format = """
    Format: 

    1. First Candidate | True | (entity_type)
    2. Second Candidate | False | (entity_type)
    """

    def set_config(self, alg, exemplar=True, coT=True, tf=True, defn=True, pos=True):
        if defn:
            alg.defn = self.defn
        else:
            alg.defn = ""
        if pos:
            alg.pos = "pos"
        else:
            alg.pos = None
        if not exemplar:
            alg.exemplar_task = None
            if coT:
                whole_task = "Q: Given the paragraph below, identify a list of possible entities " \
                                 "and for each entry explain why it either is or is not an entity. Answer in the format: \n"

                alg.format_task = whole_task + self.cot_format

            else:
                whole_task = "Q: Given the paragraph below, identify the list of entities " \
                             "Answer in the format: \n"

                alg.format_task = whole_task + self.exemplar_format
        else:
            alg.format_task = None
            if coT:
                if not pos:
                    whole_task = "Q: Given the paragraph below, identify a list of possible entities " \
                                 "and for each entry explain why it either is or is not an entity. \nParagraph:"
                    alg.whole_task = whole_task
                    alg.exemplars = self.no_pos_exemplars
                    exemplar_construction = ""
                    for exemplar in self.no_pos_exemplars:
                        exemplar_construction = exemplar_construction + whole_task + "\n"
                        exemplar_construction = exemplar_construction + exemplar + "\n"
                    exemplar_construction = exemplar_construction + whole_task + "\n"
                    alg.exemplar_task = exemplar_construction
                else:
                    whole_task = "Q: Given the paragraph below, identify a list of possible entities " \
                                 "and for each entry explain why it either is or is not an entity. \nParagraph:"
                    alg.whole_task = whole_task
                    alg.exemplars = self.cot_exemplars
                    exemplar_construction = ""
                    for exemplar in self.cot_exemplars:
                        exemplar_construction = exemplar_construction + whole_task + "\n"
                        exemplar_construction = exemplar_construction + exemplar + "\n"
                    exemplar_construction = exemplar_construction + whole_task + "\n"
                    alg.exemplar_task = exemplar_construction
            else:
                whole_task = "Q: Given the paragraph below, identify the list of entities \nParagraph:"
                exemplar_construction = ""
                e_list = self.tf_exemplars
                alg.whole_task = whole_task
                alg.exemplars = e_list
                for exemplar in e_list:
                    exemplar_construction = exemplar_construction + whole_task + "\n"
                    exemplar_construction = exemplar_construction + exemplar + "\n"
                exemplar_construction = exemplar_construction + whole_task + "\n"
                alg.exemplar_task = exemplar_construction


class ConllConfig(Config):
    defn = "An entity is a real object or concept representing a person (PER), named organization (ORG), location (LOC), country (LOC) or nationality (MISC)." \
           "Most entities are expressed as proper nouns (NNP or NNPS) while nationality and language entities under MISC are adjectives. " \
           "Names, first names, last names, countries are entities. Nationalities are entities even if they are " \
           "adjectives. Sports, sporting events, adjectives, verbs, numbers, " \
                "adverbs, abstract concepts, sports, are not entities. Dates, years, weekdays, months and times are not entities. " \
           "Possessive words and pronouns like I, you, him and me are not entities. " \
           "If a sporting team has the name of their location and the location is used to refer to the team, " \
           "it is an entity which is an organization, not a location"

    cot_exemplar_1 = """
    South/NNP African/JJ all/DT -/HYPH rounder/NN Shaun/NNP Pollock/NNP ,/, forced/VBN to/TO cut/VB short/IN his/PRP$ first/JJ season/NN with/IN Warwickshire/NNP to/TO have/VB ankle/NN surgery/NN ,/, has/VBZ told/VBN the/DT English/JJ county/NN he/PRP would/MD like/VB to/TO return/VB later/RBR in/IN his/PRP$ career/NN ./. 
    
    Answer:
    1. South African | True | as it is an adjective representing nationality (MISC)
    2. Shaun Pollock | True | as it is a person's name (PER)
    3. Warwickshire | True | as it refers to a cricket club, emphasizing an organization here (ORG)
    4. English | True | as it is an adjective representing nationality (MISC)
    5. county | False | as it is a common noun
    6. he | False | as it is a pronoun
    7. career | False | as it is a common noun
    """

    cot_exemplar_2 = """
    Chesterfield/NNP :/: Worcestershire/NNP 238/CD (/-LRB- W./NNP Weston/NNP 100/CD not/RB out/RB ,/, V./NNP Solanki/NNP 58/CD ;/: A./NNP Harris/NNP 4/CD -31/CD )/-RRB- ,/, Derbyshire/NNP 166/CD -/SYM 1/CD (/-LRB- K./NNP Barnett/NNP 83/CD not/RB out/RB )/-RRB- 
    
    Answer:
    1. Chesterfield | True | as it is a cricket ground (LOC)
    2. Worcestershire | True | as it is a cricket club, it is an organization here (ORG)
    3. W. Weston | True | as it is a person's name (PER)
    4. V. Solanki | True | as it is a person's name (PER)
    5. A. Harris | True | as it is a person's name (PER)
    6. Derbyshire | True | as it is a cricket club, it refers to an organization here (ORG)
    7. K. Barnett | True | as it is a person's name (PER)
    8. 83 | False | as it is a number
    """

    cot_exemplar_3 = """
    Best/JJS emerged/VBD from/IN Chapter/NNP 11/CD bankruptcy/NN protection/NN in/IN June/NNP 1994/CD after/IN 3/CD -/SYM 1//CD 2/CD years/NNS ./. 
    
    Answer:
    1. Best | True | as it is a company or organization (ORG)
    2. Chapter 11 | True | as it is a law document (MISC)
    3. bankruptcy protection | False | as it is common noun
    4. June 1994 | False | as it is a date, dates or time are not entities
    5. 3-1/2 | False | as it is a number
    """

    cot_exemplar_4 = """
    --/NFP Julie/NNP Tolkacheva/NNP ,/, Moscow/NNP Newsroom/NNP ,/, +7095/CD 941/CD 8520/CD 
    
    Answer:
    1. Julie Tolkacheva | True | as it is a person's name (PER)
    2. Moscow Newsroom | True | as it refers to an editorial office, an named organization (ORG)
    3. +7095 941 8520 | False | as it is a phone number
    
    """

    cot_exemplar_5 = """
    SOCCER/NNP -/, AUSTRIA/NNP BEAT/NNP SCOTLAND/NNP 4-0/CD IN/IN EUROPEAN/NNP UNDER/IN -21/CD MATCH/NN ./. 
    
    Answer:
    1. SOCCER | False | as it refers to a sport, not an entity
    2. AUSTRIA | True | as it is a country (LOC)
    3. SCOTLAND | True | as it is a country (LOC)
    4. EUROPEAN | True | as it is an adjective of nationality (MISC)
    5. MATCH | False | as it is not proper noun
    """

    cot_exemplar_6 = """
    Ssangbangwool/NNP 12/CD Hanwha/NNP 0/CD 
    
    Answer:
    1. Ssangbangwool | True | as it is a baseball team, hence an orgazation (ORG)
    2. 12 | False | as it is a number representing score
    3. Hanwha | True | as it is a baseball team, hence an orgazation (ORG)
    
    """

    cot_exemplar_7 = """
    TORONTO/NNP 63/CD 71/CD .470/CD 12/CD 
    
    Answer:
    1. TORONTO | True | as it refers to a soccer club here and numbers after it are sports data, hence it is an organization (ORG)
    2. 63 | False | as it is sports data
    """

    cot_exemplar_8 = """
    DETROIT/NNP AT/IN KANSAS/NNP CITY/NNP 
    
    Answer:
    1. DETROIT | True | as it refers to a basketball team (ORG)
    2. KANSAS CITY | True | as the word before it is at, it emphasizes a location here (LOC)
    """

    cot_exemplar_9 = """
    SEATTLE/NNP AT/IN BOSTON/NNP

    Answer:
    1. SEATTLE | True | as it refers to a basketball team (ORG)
    2. BOSTON | True | as the word before it is at, it emphasizes a location here (LOC)
    """
    cot_exemplars = [cot_exemplar_1, cot_exemplar_2, cot_exemplar_3, cot_exemplar_4, cot_exemplar_5, cot_exemplar_6, cot_exemplar_7, cot_exemplar_8, cot_exemplar_9]

    tf_exemplar_1 = """
    South/NNP African/JJ all/DT -/HYPH rounder/NN Shaun/NNP Pollock/NNP ,/, forced/VBN to/TO cut/VB short/IN his/PRP$ first/JJ season/NN with/IN Warwickshire/NNP to/TO have/VB ankle/NN surgery/NN ,/, has/VBZ told/VBN the/DT English/JJ county/NN he/PRP would/MD like/VB to/TO return/VB later/RBR in/IN his/PRP$ career/NN ./. 
    
    Answer:
    1. South African | True | (MISC)
    2. Shaun Pollock | True | (PER)
    3. Warwickshire | True | (ORG)
    4. English | True | (MISC)
    5. county | False | None
    6. he | False | None
    7. career | False | None
    """

    tf_exemplar_2 = """
    Chesterfield/NNP :/: Worcestershire/NNP 238/CD (/-LRB- W./NNP Weston/NNP 100/CD not/RB out/RB ,/, V./NNP Solanki/NNP 58/CD ;/: A./NNP Harris/NNP 4/CD -31/CD )/-RRB- ,/, Derbyshire/NNP 166/CD -/SYM 1/CD (/-LRB- K./NNP Barnett/NNP 83/CD not/RB out/RB )/-RRB- 
    
    Answer:
    1. Chesterfield | True | (LOC)
    2. Worcestershire | True | (ORG)
    3. W. Weston | True | (PER)
    4. V. Solanki | True | (PER)
    5. A. Harris | True | (PER)
    6. Derbyshire | True | (ORG)
    7. K. Barnett | True | (PER)
    8. 83 | False | None
    """

    tf_exemplar_3 = """
    Best/JJS emerged/VBD from/IN Chapter/NNP 11/CD bankruptcy/NN protection/NN in/IN June/NNP 1994/CD after/IN 3/CD -/SYM 1//CD 2/CD years/NNS ./. 
    
    Answer:
    1. Best | True | (ORG)
    2. Chapter 11 | True | (MISC)
    3. bankruptcy protection | False | None
    4. June 1994 | False | None
    5. 3-1/2 | False | None
    """

    tf_exemplar_4 = """
    --/NFP Julie/NNP Tolkacheva/NNP ,/, Moscow/NNP Newsroom/NNP ,/, +7095/CD 941/CD 8520/CD 
    
    Answer:
    1. Julie Tolkacheva | True | (PER)
    2. Moscow Newsroom | True | (ORG)
    3. +7095 941 8520 | False | None
    
    """

    tf_exemplar_5 = """
    SOCCER/NNP -/, AUSTRIA/NNP BEAT/NNP SCOTLAND/NNP 4-0/CD IN/IN EUROPEAN/NNP UNDER/IN -21/CD MATCH/NN ./. 
    
    Answer:
    1. SOCCER | False | None
    2. AUSTRIA | True | (LOC)
    3. SCOTLAND | True | (LOC)
    4. EUROPEAN | True | (MISC)
    5. MATCH | False | None
    """

    tf_exemplar_6 = """
    Ssangbangwool/NNP 12/CD Hanwha/NNP 0/CD 
    
    Answer:
    1. Ssangbangwool | True | (ORG)
    2. 12 | False | None
    3. Hanwha | True | (ORG)
    
    """

    tf_exemplar_7 = """
    TORONTO/NNP 63/CD 71/CD .470/CD 12/CD 
    
    Answer:
    1. TORONTO | True | (ORG)
    2. 63 | False | None
    """

    tf_exemplar_8 = """
    DETROIT/NNP AT/IN KANSAS/NNP CITY/NNP 
    
    Answer:
    1. DETROIT | True | (ORG)
    2. KANSAS CITY | True | (LOC)
    """

    tf_exemplar_9 = """
    SEATTLE/NNP AT/IN BOSTON/NNP

    Answer:
    1. SEATTLE | True | (ORG)
    2. BOSTON | True | (LOC)
    """

    tf_exemplars = [tf_exemplar_1, tf_exemplar_2, tf_exemplar_3, tf_exemplar_4, tf_exemplar_5, tf_exemplar_6, tf_exemplar_7, tf_exemplar_8, tf_exemplar_9]

    no_pos_exemplar_1 = """
    South African all - rounder Shaun Pollock , forced to cut short his first season with Warwickshire to have ankle surgery , has told the English county he would like to return later in his career . 
    
    Answer:
    1. South African | True | as it is an adjective representing nationality (MISC)
    2. Shaun Pollock | True | as it is a person's name (PER)
    3. Warwickshire | True | as it refers to a cricket club, not emphasizes a location hence it is an organization (ORG)
    4. English | True | as it is an adjective representing nationality (MISC)
    5. county | False | as it is a common noun, not a location
    6. he | False | as it is a pronoun
    7. career | False | as it is a common noun
    """

    no_pos_exemplar_2 = """
    Chesterfield : Worcestershire 238 ( W. Weston 100 not out , V. Solanki 58 ; A. Harris 4 -31 ) , Derbyshire 166 - 1 ( K. Barnett 83 not out ) 
    
    Answer:
    1. Chesterfield | True | as it is a cricket ground (LOC)
    2. Worcestershire | True | as it is a cricket club, hence it is an organization (ORG)
    3. W. Weston | True | as it is a person's name (PER)
    4. V. Solanki | True | as it is a person's name (PER)
    5. A. Harris | True | as it is a person's name (PER)
    6. Derbyshire | True | as it is a cricket club, hence it refers to an organization here (ORG)
    7. K. Barnett | True | as it is a person's name (PER)
    """

    no_pos_exemplar_3 = """
    Best emerged from Chapter 11 bankruptcy protection in June 1994 after 3-1/2 years .
    
    Answer:
    1. Best | True | as it is a company or organization (ORG)
    2. Chapter 11 | True | as it is a law document (MISC)
    3. bankruptcy protection | False | as it is common noun
    4. June 1994 | False | as it is a date
    5. 3-1/2 | False | as it is a number
    """

    no_pos_exemplar_4 = """
    -- Julie Tolkacheva , Moscow Newsroom , +7095 941 8520
    
    Answer:
    1. Julie Tolkacheva | True | as it is a person's name (PER)
    2. Moscow Newsroom | True | as it refers to an editorial office, an named organization (ORG)
    3. +7095 941 8520 | False | as it is a phone number
    
    """

    no_pos_exemplar_5 = """
    SOCCER - AUSTRIA BEAT SCOTLAND 4-0 IN EUROPEAN UNDER-21 MATCH .
    
    Answer:
    1. SOCCER | False | as it refers to a kind of sports, not an entity
    2. AUSTRIA | True | as it is a country (LOC)
    3. SCOTLAND | True | as it is a country (LOC)
    4. EUROPEAN | True | as it is an adjective of nationality (MISC)
    """

    no_pos_exemplar_6 = """
    Ssangbangwool 12 Hanwha 0
    
    Answer:
    1. Ssangbangwool | True | as it is a baseball team, hence an orgazation (ORG)
    2. 12 | False | as it is a number representing score
    3. Hanwha | True | as it is a baseball team, hence an orgazation (ORG)
    
    """

    no_pos_exemplar_7 = """
    TORONTO 63 71 .470 12
    
    Answer:
    1. TORONTO | True | as it refers to a soccer club here and numbers after it are sports data, hence it is an organization (ORG)
    2. 63 | False | as it is sports data
    """

    no_pos_exemplar_8 = """
    DETROIT AT KANSAS CITY
    
    Answer:
    1. DETROIT | True | as it refers to a basketball team (ORG)
    2. KANSAS CITY | True | as the word before it is at, it emphasizes a location here (LOC)
    """

    no_pos_exemplar_9 = """
    SEATTLE AT BOSTON

    Answer:
    1. SEATTLE | True | as it refers to a basketball team (ORG)
    2. BOSTON | True | as the word before it is at, it emphasizes a location here (LOC)   
    
    """
    no_pos_exemplars = [no_pos_exemplar_1, no_pos_exemplar_2, no_pos_exemplar_3, no_pos_exemplar_4, no_pos_exemplar_5, no_pos_exemplar_6, no_pos_exemplar_7, no_pos_exemplar_8, no_pos_exemplar_9]


class Ontonotes_ten_Config(Config):
    defn = "An entity is a real-world object or concept that represents an event, facility, country, language, location, nationality, organization, person, product, or work of art. " \
           "Typically, entities are expressed as proper nouns (NNP or NNPs) in linguistic terms." \
           "Event (EVENT) entities refer to proper nouns representing hurricanes, battles, wars, sports events, attacks. " \
           "Facility (FAC) entities refer to proper nouns associated with man-made structures like buildings, airports, highways, bridges. " \
           "Geographical/Social/Political (GPE) entities refer to proper nouns representing countries, cities, states, provinces, municipalities. " \
           "Language (LANGUAGE) entities refer to named languages. " \
           "Location (LOC) entities refer to proper nouns representing non-GPE locations including mountain ranges, planets, geo-coordinates, bodies of water, named regions, continents. " \
           "Nationalities, religious, or political groups (NORP) are expressed through adjectival forms of geographical, social, political entities, location names, named religions, heritage and political affiliation. " \
           "Organization (ORG) entities refer to proper nouns representing companies, government agencies, educational institutions, sports teams, also including adjectival forms of organization names and metonymic mentions of associated buildings or locations. " \
           "Person (PERSON) entities are represented by proper personal names, including fictional characters, first names, last names, nicknames and generational markers (such as Jr., IV), excluding occupational titles and honorifics. " \
           "Product (PRODUCT) entities refer to proper nouns representing model names, vehicles, or weapons. Manufacturer and product should be marked separately. " \
           "Works of art (WORK_OF_ART) refer to titles of books, songs, articles, television programs, or awards. " \
           "If an organization, an occupation title, and a person's name form a phrase, then the organization and person's name is marked separately. " \
           "Nominals and common nouns are not considered as entities. " \
           "Pronouns and pronominal elements are excluded from entities, as are contact information, plants, dates, years, times, numbers, legal documents, treaties, credit cards, checking accounts, CDs, credit plans, financial instruments, abstract concepts. "

    cot_exemplar_1 = """
    US/NNP officials/NNS described/VBD the/DT one/CD -/HYPH hour/NN White/NNP House/NNP meeting/NN ,/, Tuesday/NNP between/IN Mr./NNP Clinton/NNP and/CC Vice/NNP -/HYPH marshall/NNP Jo/NNP Myong/NNP Rok/NNP as/IN very/RB positive/JJ ,/, direct/JJ and/CC warm/JJ ./. 
    
    Answer:
    1. US | True | as it is a country (GPE)
    2. officials | False | as it is common noun, only mark the prefix modifier based on its own meaning
    3. one - hour | False | as it is time
    4. White House | True | as it emphasizes the location of the building rather than the organization here (FAC)
    5. Tuesday | False | as it is a date
    6. Clinton | True | as it is a person's name (PERSON)
    7. Jo Myong Rok | True | as it is a person's name (PERSON)
    
    """

    cot_exemplar_2 = """
    The/DT potential/JJ sales/NNS are/VBP nearly/RB $/$ 9.3/CD million/CD ,/, and/CC House/NNP Majority/NNP Whip/NNP William/NNP Gray/NNP -LRB-/-LRB D./NNP ,/, Pa/NNP ./. -RRB-/-RRB- began/VBD the/DT bidding/NN this/DT year/NN by/IN proposing/VBG language/NN that/WDT the/DT quota/NN be/VB allocated/VBN to/IN English/JJ -/HYPH speaking/NN countries/NNS of/IN the/DT Caribbean/NNP ,/, such/JJ as/IN Jamaica/NNP and/CC Barbados/NNP ./. 

    Answer:
    1. potential sales | False | as it is not proper noun
    2. House | True | as it refers to House of Representatives, emphasizing a specific organization (ORG)
    3. William Gray | True | as it is a person's name (PERSON)
    4. D. | True | as "D." after a person's name often signifies the political party affiliation (NORP)
    5. Pa . | True | as it stands for Pennsylvania, a state abbreviation (GPE)
    6. English | True | as it is a named language (LANGUAGE)
    7. Caribbean | True | as Caribbean refers to a geographical region here (LOC)
    8. Jamaica | True | as it is a country (GPE)
    9. Barbados | True | as it is a country (GPE)
    """

    cot_exemplar_3 = """
    --/NFP ``/`` Arms/NNPS Control/NNP Reality/NNP ,/, ''/'' Nov./NNP 20/CD ,/, 1984/CD ,/, the/DT first/JJ of/IN some/DT 20/CD Journal/NNP editorials/NNS saying/VBG that/IN Krasnoyarsk/NNP violated/VBD the/DT ABM/NNP treaty/NN ./. 

    Answer:
    1. Arms Control Reality | True | as it is the title of a editorial (WORK_OF_ART)
    2. Journal | True | as it refers to the name of a publication company (ORG)
    3. editorials | False | as it is not NNP
    4. Krasnoyarsk | True | as it is a city (GPE)
    5. violated | False | as it is an action
    6. ABM | False | as it is a law or treaty, not marked
    """

    cot_exemplar_4 = """
    From/IN Ramstein/NNP ,/, the/DT bodies/NNS will/MD go/VB to/IN the/DT Cole/NNP 's/POS home/NN port/NN ,/, Norfolk/NNP ,/, Virginia/NNP ,/, just/RB as/IN Fleet/NNP Week/NNP is/VBZ beginning/VBG to/TO celebrate/VB the/DT anniversary/NN of/IN the/DT American/NNP Navy/NNP ./. 

    Answer:
    1. Ramstein | True | as it is a military airbase, not a natural location, so it is a FAC entity rather than a LOC entity (FAC)
    2. Cole | True | as it is a specific ship name (PRODUCT)
    3. Norfolk | True | as it is a city in Virginia (GPE)
    4. Virginia | True | as it is a state (GPE)
    5. Fleet Week | True | as it is a specific event (EVENT)
    6. American | True | as it is an adjective of nationality (NORP)
    7. Navy | True | as it refers to a specific organization (ORG)
    """

    cot_exemplar_5 = """
    In/IN addition/NN to/IN aircraft/NN from/IN Boeing/NNP Co./NNP ,/, Cathay/NNP announced/VBD earlier/RBR this/DT year/NN an/DT order/NN for/IN as/RB many/JJ as/IN 20/CD Airbus/NNP A330/NNP -/HYPH 300s/NNS ./. 

    Answer:
    1. Boeing Co. | True | as it is a specific company (ORG)
    2. Cathay | True | as it is a specific organization in the aviation industry (ORG)
    3. Airbus | True | as it refers to a specific aircraft manufacturer. The manufacturer and the product should be identified separately (ORG)
    4. A330 - 300s | True | as it represents a specific model of aircraft (PRODUCT)
    """

    cot_exemplar_6 = """
    At/IN a/DT photo/NN opportunity/NN in/IN the/DT Oval/NNP Office/NNP ,/, Bush/NNP thanked/VBD the/DT President/NNP and/CC fended/VBD off/RP an/DT interruption/NN from/IN a/DT veteran/JJ White/NNP House/NNP correspondent/NN ./. 
    
    Answer:
    1. the Oval Office | True | as it is the specific name of a building (FAC)
    2. Bush | True | as it is a person name (PERSON) 
    3. White House | True | as it emphasizes the organization rather than the location of the building here (ORG)
    4. correspondent | False | as it is a common noun
    """

    cot_exemplar_7 = """
    The/DT company/NN is/VBZ operating/VBG under/IN Chapter/NNP 11/CD of/IN the/DT federal/NNP Bankruptcy/NNP Code/NNP , giving/VBG it/PRP court/NN protection/NN from/IN creditors/NNS ' lawsuits/NNS while/IN it/PRP attempts/VBZ to/TO work/VB out/RP a/DT plan/NN to/TO pay/VB its/PRP$ debts/NNS .
    
    Answer:
    1. company | False | as it is not NNP
    2. Chapter 11 | False | as it represents a law, not a work of art, not marked
    3. Bankruptcy Code | False | as it refers to a law, not a work of art, not marked
    """

    cot_exemplar_8 = """
    The/DT dollar/NN began/VBD Friday/NNP on/IN a/DT firm/NN note/NN ,/, gaining/VBG against/IN all/DT major/JJ currencies/NNS in/IN Tokyo/NNP dealings/NNS and/CC early/JJ European/JJ trading/NN despite/IN reports/NNS that/IN the/DT Bank/NNP of/IN Japan/NNP was/VBD seen/VBN unloading/VBG dollars/NNS around/IN 142.70/CD yen/NNS ./. 
    
    Answer:
    1. dollar | False | as it is currency or financial instrument, not a product
    2. Tokyo | True | as it is a city (GPE)
    3. European | True | as it is an adjective of nationality (NORP)
    4. the Bank of Japan | True | as it is the name of a organization (ORG)
    5. 142.70 yen | False | as it is a monetary value, not a product
    """

    cot_exemplar_9 = """
    But/CC those/DT dollars/NNS have/VBP been/VBN going/VBG into/IN such/JJ ``/`` safe/JJ ''/'' products/NNS as/IN money/NN market/NN funds/NNS ,/, which/WDT do/VBP n/RB 't/RB generate/VB much/JJ in/IN the/DT way/NN of/IN commissions/NNS for/IN the/DT brokerage/NN firms/NNS ./.
    
    Answer:
    1. dollars | False | as it is a currency, not a specific entity
    2. money market funds | False | as it is a financial instrument, not an entity
    3. brokerage firms | False | as it is not NNP
    """

    cot_exemplar_10 = """
    This/DT may/MD have/VB been/VBN the/DT case/NN in/IN the/DT 18th/JJ century/NN , or/CC even/RB in/IN the/DT 1970s/CD ,
    
    Answer:
    1. 18th century | False | as it is time, not marked
    2. 1970s | False | as it is time, not marked
    """    

    cot_exemplar_11 = """
    Cycads/NNS , the/DT most/RBS popular/JJ of/IN which/WDT is/VBZ the/DT Sago Palm/NNP , are/VBP doll - sized/JJ versions/NNS of/IN California/NNP 's famous/JJ long - necked/JJ palms/NNS , with/IN stubby/JJ trunks/NNS and/CC fern - like/JJ fronds/NNS .

    Answer:
    1. Cycads | False | it is not product and not NNP
    2. Sago Palm | False | though it is NNP of plant, it is not product
    3. California | True | as it is a state (GPE)
    """

    cot_exemplar_12 = """
    Success/NN is/VBZ expected/VBN to/TO gain/VB at/IN least/JJS because/IN of/IN the/DT recent/JJ folding/NN of/IN rival/JJ Venture/NNP ,/, another/DT magazine/NN for/IN growing/VBG companies/NNS ./.     
    
    Answer:
    1. Success | True | although it is the title of a magazine, it emphasizes the company here (ORG)
    2. Venture | True | although it is the title of a magazine, it emphasizes the company here (ORG)
    """
    cot_exemplars = [cot_exemplar_1, cot_exemplar_2, cot_exemplar_3, cot_exemplar_4, cot_exemplar_5, cot_exemplar_6, cot_exemplar_7, cot_exemplar_8, cot_exemplar_9, cot_exemplar_10, cot_exemplar_11, cot_exemplar_12]

    tf_exemplar_1 = """
    US officials described the one - hour White House meeting , Tuesday between Mr. Clinton and Vice - marshall Jo Myong Rok as very positive , direct and warm . 
    
    Answer:
    1. US | True | (GPE)
    2. officials | False | None
    3. one - hour | False | None
    4. White House | True | (FAC)
    5. Tuesday | False | None
    6. Clinton | True | (PERSON)
    7. Jo Myong Rok | True | (PERSON)
    
    """

    tf_exemplar_2 = """
    The potential sales are nearly $ 9.3 million , and House Majority Whip William Gray -LRB- D. , Pa . -RRB- began the bidding this year by proposing language that the quota be allocated to English - speaking countries of the Caribbean , such as Jamaica and Barbados . 

    Answer:
    1. potential sales | False | None
    2. House | True | (ORG)
    3. William Gray | True | (PERSON)
    4. D. | True | (NORP)
    5. Pa . | True | (GPE)
    6. English | True | (LANGUAGE)
    7. Caribbean | True | (LOC)
    8. Jamaica | True | (GPE)
    9. Barbados | True | (GPE)
    """

    tf_exemplar_3 = """
    -- `` Arms Control Reality , '' Nov. 20 , 1984 , the first of some 20 Journal editorials saying that Krasnoyarsk violated the ABM treaty . 

    Answer:
    1. Arms Control Reality | True | (WORK_OF_ART)
    2. Journal | True | (ORG)
    3. editorials | False | None
    4. Krasnoyarsk | True | (GPE)
    5. violated | False | None
    6. ABM | False | None
    """

    tf_exemplar_4 = """
    From Ramstein , the bodies will go to the Cole 's home port , Norfolk , Virginia , just as Fleet Week is beginning to celebrate the anniversary of the American Navy . 

    Answer:
    1. Ramstein | True | (FAC)
    2. Cole | True | (PRODUCT)
    3. Norfolk | True | (GPE)
    4. Virginia | True | (GPE)
    5. Fleet Week | True | (EVENT)
    6. American | True | (NORP)
    7. Navy | True | (ORG)
    """

    tf_exemplar_5 = """
    In addition to aircraft from Boeing Co. , Cathay announced earlier this year an order for as many as 20 Airbus A330 - 300s . 

    Answer:
    1. Boeing Co. | True | (ORG)
    2. Cathay | True | (ORG)
    3. Airbus | True | (ORG)
    4. A330 - 300s | True | (PRODUCT)
    """

    tf_exemplar_6 = """
    At a photo opportunity in the Oval Office , Bush thanked the President and fended off an interruption from a veteran White House correspondent . 
    
    Answer:
    1. the Oval Office | True | (FAC)
    2. Bush | True | (PERSON) 
    3. White House | True | (ORG)
    4. correspondent | False | None
    """

    tf_exemplar_7 = """
    The company is operating under Chapter 11 of the federal Bankruptcy Code , giving it court protection from creditors ' lawsuits while it attempts to work out a plan to pay its debts .
    
    Answer:
    1. company | False | None
    2. Chapter 11 | False | None
    3. Bankruptcy Code | False | None
    """

    tf_exemplar_8 = """
    The dollar began Friday on a firm note , gaining against all major currencies in Tokyo dealings and early European trading despite reports that the Bank of Japan was seen unloading dollars around 142.70 yen . 
    
    Answer:
    1. dollar | False | None
    2. Tokyo | True | (GPE)
    3. European | True | (NORP)
    4. the Bank of Japan | True | (ORG)
    5. 142.70 yen | False | None
    """

    tf_exemplar_9 = """
    But those dollars have been going into such `` safe '' products as money market funds , which do n 't generate much in the way of commissions for the brokerage firms .
    
    Answer:
    1. dollars | False | None
    2. money market funds | False | None
    3. brokerage firms | False | None
    """

    tf_exemplar_10 = """
    This may have been the case in the 18th century , or even in the 1970s ,
    
    Answer:
    1. 18th century | False | None
    2. 1970s | False | None
    """    

    tf_exemplar_11 = """
    Cycads , the most popular of which is the Sago Palm , are doll - sized versions of California 's famous long - necked palms , with stubby trunks and fern - like fronds .

    Answer:
    1. Cycads | False | None
    2. Sago Palm | False | None
    3. California | True | (GPE)
    """

    tf_exemplar_12 = """
    Success is expected to gain at least because of the recent folding of rival Venture , another magazine for growing companies .    
    
    Answer:
    1. Success | True | (ORG)
    2. Venture | True | (ORG)   
    
    """
    tf_exemplars = [tf_exemplar_1, tf_exemplar_2, tf_exemplar_3, tf_exemplar_4, tf_exemplar_5, tf_exemplar_6, tf_exemplar_7, tf_exemplar_8, tf_exemplar_9, tf_exemplar_10, tf_exemplar_11, tf_exemplar_12]

    no_pos_exemplar_1 = """
    US officials described the one - hour White House meeting , Tuesday between Mr. Clinton and Vice - marshall Jo Myong Rok as very positive , direct and warm . 
    
    Answer:
    1. US | True | as it is a country (GPE)
    2. officials | False | as it is common noun, only mark the prefix modifier based on its own meaning
    3. one - hour | False | as it is time
    4. White House | True | as it emphasizes the location of the building rather than the organization here (FAC)
    5. Tuesday | False | as it is a date
    6. Clinton | True | as it is a person's name (PERSON)
    7. Jo Myong Rok | True | as it is a person's name (PERSON)
    
    """

    no_pos_exemplar_2 = """
    The potential sales are nearly $ 9.3 million , and House Majority Whip William Gray -LRB- D. , Pa . -RRB- began the bidding this year by proposing language that the quota be allocated to English - speaking countries of the Caribbean , such as Jamaica and Barbados . 

    Answer:
    1. potential sales | False | as it is not proper noun
    2. House | True | as it refers to House of Representatives, emphasizing a specific organization (ORG)
    3. William Gray | True | as it is a person's name (PERSON)
    4. D. | True | as "D." after a person's name often signifies the political party affiliation (NORP)
    5. Pa . | True | as it stands for Pennsylvania, a state abbreviation (GPE)
    6. English | True | as it is a named language (LANGUAGE)
    7. Caribbean | True | as Caribbean refers to a geographical region here (LOC)
    8. Jamaica | True | as it is a country (GPE)
    9. Barbados | True | as it is a country (GPE)
    """

    no_pos_exemplar_3 = """
    -- `` Arms Control Reality , '' Nov. 20 , 1984 , the first of some 20 Journal editorials saying that Krasnoyarsk violated the ABM treaty . 

    Answer:
    1. Arms Control Reality | True | as it is the title of a editorial (WORK_OF_ART)
    2. Journal | True | as it refers to the name of a publication company (ORG)
    3. editorials | False | as it is not NNP
    4. Krasnoyarsk | True | as it is a city (GPE)
    5. violated | False | as it is an action
    6. ABM | False | as it is a law or treaty, not marked
    """

    no_pos_exemplar_4 = """
    From Ramstein , the bodies will go to the Cole 's home port , Norfolk , Virginia , just as Fleet Week is beginning to celebrate the anniversary of the American Navy . 

    Answer:
    1. Ramstein | True | as it is a military airbase, not a natural location, so it is a FAC entity rather than a LOC entity (FAC)
    2. Cole | True | as it is a specific ship name (PRODUCT)
    3. Norfolk | True | as it is a city in Virginia (GPE)
    4. Virginia | True | as it is a state (GPE)
    5. Fleet Week | True | as it is a specific event (EVENT)
    6. American | True | as it is an adjective of nationality (NORP)
    7. Navy | True | as it refers to a specific organization (ORG)
    """

    no_pos_exemplar_5 = """
    In addition to aircraft from Boeing Co. , Cathay announced earlier this year an order for as many as 20 Airbus A330 - 300s . 

    Answer:
    1. Boeing Co. | True | as it is a specific company (ORG)
    2. Cathay | True | as it is a specific organization in the aviation industry (ORG)
    3. Airbus | True | as it refers to a specific aircraft manufacturer. The manufacturer and the product should be identified separately (ORG)
    4. A330 - 300s | True | as it represents a specific model of aircraft (PRODUCT)
    """

    no_pos_exemplar_6 = """
    At a photo opportunity in the Oval Office , Bush thanked the President and fended off an interruption from a veteran White House correspondent . 
    
    Answer:
    1. the Oval Office | True | as it is the specific name of a building (FAC)
    2. Bush | True | as it is a person name (PERSON) 
    3. White House | True | as it emphasizes the organization rather than the location of the building here (ORG)
    4. correspondent | False | as it is a common noun
    """

    no_pos_exemplar_7 = """
    The company is operating under Chapter 11 of the federal Bankruptcy Code , giving it court protection from creditors ' lawsuits while it attempts to work out a plan to pay its debts .
    
    Answer:
    1. company | False | as it is not NNP
    2. Chapter 11 | False | as it represents a law, not a work of art, not marked
    3. Bankruptcy Code | False | as it refers to a law, not a work of art, not marked
    """

    no_pos_exemplar_8 = """
    The dollar began Friday on a firm note , gaining against all major currencies in Tokyo dealings and early European trading despite reports that the Bank of Japan was seen unloading dollars around 142.70 yen . 
    
    Answer:
    1. dollar | False | as it is currency or financial instrument, not a product
    2. Tokyo | True | as it is a city (GPE)
    3. European | True | as it is an adjective of nationality (NORP)
    4. the Bank of Japan | True | as it is the name of a organization (ORG)
    5. 142.70 yen | False | as it is a monetary value, not a product
    """

    no_pos_exemplar_9 = """
    But those dollars have been going into such `` safe '' products as money market funds , which do n 't generate much in the way of commissions for the brokerage firms .
    
    Answer:
    1. dollars | False | as it is a currency, not proper noun
    2. money market funds | False | as it is a financial instrument, not an entity
    3. brokerage firms | False | as it is not NNP
    """

    no_pos_exemplar_10 = """
    This may have been the case in the 18th century , or even in the 1970s ,
    
    Answer:
    1. 18th century | False | as it is time, not marked
    2. 1970s | False | as it is time, not marked
    """    

    no_pos_exemplar_11 = """
    Cycads , the most popular of which is the Sago Palm , are doll - sized versions of California 's famous long - necked palms , with stubby trunks and fern - like fronds .

    Answer:
    1. Cycads | False | it is not product and not NNP
    2. Sago Palm | False | though it is NNP of plant, it is not product
    3. California | True | as it is a state (GPE)
    """

    no_pos_exemplar_12 = """
    Success is expected to gain at least because of the recent folding of rival Venture , another magazine for growing companies .    
    
    Answer:
    1. Success | True | although it is the title of a magazine, it emphasizes the company here (ORG)
    2. Venture | True | although it is the title of a magazine, it emphasizes the company here (ORG)   
    
    """
    no_pos_exemplars = [no_pos_exemplar_1, no_pos_exemplar_2, no_pos_exemplar_3, no_pos_exemplar_4, no_pos_exemplar_5, no_pos_exemplar_6, no_pos_exemplar_7, no_pos_exemplar_8, no_pos_exemplar_9, no_pos_exemplar_10, no_pos_exemplar_11, no_pos_exemplar_12]
