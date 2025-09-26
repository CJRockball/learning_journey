# #%% 11.1 Code to run spacy's NER on a sentence
# import spacy
# nlp = spacy.load('en_core_web_md')

# doc = nlp('I bought two books on Amazon')
# for ent in doc.ents:
#     print(ent.text, ent.label_)


# %% 11.2 Code to extract the data from the input file using pandas
import pandas as pd
df = pd.read_csv('data_ch11/articles1.csv')

# 11.3 Code to extract the information on the news sources only
sources = df['publication'].unique()
print(sources)

# 11.4 Code to extract the countent of artiles from a spectific source
condition = df['publication'].isin(['New York Times'])
content_df = df.loc[condition,:]['content'][:1000]
print(content_df.shape)

# %% 11.5 Code to populate a dictionary with NEs extracted from news articles
import spacy
nlp = spacy.load('en_core_web_md')

def collect_entites(data_frame):
    named_enteties = {}
    processed_docs = []
    
    for item in data_frame:
        doc = nlp(item)
        processed_docs.append(doc)

        for ent in doc.ents:
            entity_text = ent.text
            entity_type = str(ent.label_)
            current_ents = {}
            if entity_type in named_enteties.keys():
                    current_ents = named_enteties.get(entity_type)
            current_ents[entity_text] = current_ents.get(entity_text, 0) + 1
            named_enteties[entity_type] = current_ents
    return named_enteties, processed_docs

named_enteties, processed_docs = collect_entites(content_df)

# %% 11.6 Code to print out the named entites dictionary

def print_out(named_entities):
    for key in named_entities.keys():
        print(key)
        entities = named_entities.get(key)
        sorted_keys = sorted(entities, key=entities.get, reverse=True)
        for item in sorted_keys[:10]:
            if (entities.get(item)>1):
                print('  ' + item + ": " + str(entities.get(item)))
print_out(named_enteties)

# %% 11.7 Code to aggregate the counts on all named entity types

rows = []
rows.append(['Type:', 'Entries:', 'Total:'])
for ent_type in named_enteties.keys():
    rows.append([ent_type, str(len(named_enteties.get(ent_type))),
                str(sum(named_enteties.get(ent_type).values()))])

columns = zip(*rows)
column_widths = [max(len(item) for item in col) for col in columns]
for row in rows:
    print(''.join(' {:{width}} '.format(row[i], width=column_widths[i])
                  for i in range(0, len(row))))


# %% 11.8 Code to extract the indexes of the words covered by the NE

def extract_span(sent, entity):
    indexes = []
    for ent in sent.ents:
        if ent.text == entity:
            for i in range(int(ent.start), int(ent.end)):
                indexes.append(i)
    return indexes

# %% 11.9 Code to extract information about the main participants of the action

def extract_information(sent, entity, indexes):
    actions = []
    action = ''
    participant1 = ''
    participant2 = ''
    
    for token in sent:
        if token.pos_== 'VERB' and token.dep_== 'ROOT':
            subj_ind = -1
            obj_ind = -1
            action = token.text
            children = [child for child in token.children]
            for child1 in children:
                if child1.dep_ == 'nsubj':
                    participant1 = child1.text
                    subj_ind = int(child1.i)
                if child1.dep_ == 'prep':
                    participant2 = ''
                    child1_children = [child for child in child1.children]
                    for child2 in child1_children:
                        if child2.pos_ == 'NOUN' or child2.pos_ == 'PROPN':
                            participant2 = child2.text
                            obj_ind = int(child2.i)
                    if not participant2 == '':
                        if subj_ind in indexes:
                            actions.append(entity + " " + action +
                                           ' ' + child1.text + ' '+
                                            participant2) 
                        elif obj_ind in indexes:
                            actions.append(participant1 + ' ' + action +
                                           " " + child1.text + ' ' + entity)
                if child1.dep_=='dobj' and (child1.pos_ == 'NOUN' or child1.pos_ == 'PROPN'):
                    participant2 = child1.text
                    obj_ind = int(child1.i)
                    if subj_ind in indexes:
                        actions.append(entity + ' ' + action + ' ' + participant2)
                    elif obj_ind in indexes:
                        actions.append(participant1 + ' ' + action + ' ' + entity)

    if not len(actions) == 0:
        print(f'\nSentence = {sent}')
        for item in actions:
            print(item)




# %% 11.10 Code to extract information on the specific entity

def entity_detector(processed_docs, entity, ent_type):
    output_sentences = []
    for doc in processed_docs:
        for sent in doc.sents:
            if entity in [ent.text for ent in sent.ents if ent.label_ == ent_type]:
                output_sentences.append(sent)
    return output_sentences

entity = "Apple"
ent_sentences = entity_detector(processed_docs, entity, "ORG")
print(len(ent_sentences))

for sent in ent_sentences:
    indexes = extract_span(sent, entity)
    extract_information(sent, entity, indexes)

# %% 11.11 Code to visualize named entities of various types in their contexts of use
from spacy import displacy

def visualize(processed_docs, entity, ent_type):
    for doc in processed_docs:
        for sent in doc.sents:
            if entity in [ent.text for ent in sent.ents if ent.label_ == ent_type]:
                displacy.render(sent, style='ent')
                
visualize(processed_docs, 'Apple', 'ORG')

#%% 11.12 Code to visualize named entities of a specific type only

def count_ents(sent, ent_types):
    return len([ent.text for ent in sent.ents if ent.label_ == ent_type])

def entity_detector_custom(processed_docs, entity, ent_type):
    output_sentences = []
    for doc in processed_docs:
        for sent in doc.sents:
            if entity in [ent.text for ent in sent.ents if ent.label_ == ent_type and count_ents(sent, ent_type)>1]:
                output_sentences.append(sent)
    return output_sentences

output_sentences = entity_detector_custom(processed_docs, 'Apple', 'ORG')
print(len(output_sentences))

def visualize_type(sents, entity, ent_type):
    colors = {'ORG': 'linear-gradient(90deg, #64B5F6, #E0F7FA)'}
    options = {'ents':['ORG'], 'colors':colors}
    for sent in sents:
        displacy.render(sent, style='ent', options=options)

visualize_type(processed_docs, 'Apple', 'ORG')


# %%
