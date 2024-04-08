import spacy # version 3.5
import scispacy
from scispacy.umls_linking import UmlsEntityLinker
from ipdb import set_trace
from scispacy.linking import EntityLinker
import en_core_sci_md
import pandas as pd
from collections import defaultdict
from contextlib import ContextDecorator
import time

class timeit(ContextDecorator):
    def __init__(self, var="", echo=False) -> None:
        super().__init__()
        self.var = var
        self.echo = echo

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *exc):
        if self.echo:
            print(f"Time taken for {self.var}: {time.time() - self.start_time}")
        return
"""
nlp = spacy.load('en_core_web_sm')
# add pipeline (declared through entry_points in setup.py)
nlp.add_pipe("entityLinker", last=True)

doc = nlp("I watched the Pirates of the Caribbean last silvester")

# returns all entities in the whole document
all_linked_entities = doc._.linkedEntities
# iterates over sentences and prints linked entities
for sent in doc.sents:
    sent._.linkedEntities.pretty_print()
"""   
class KnowledgeRetrieval(object):
    def __init__(self):

        self.spacy_web_sm = spacy.load("en_core_web_sm")
        self.spacy_web_sm.add_pipe("entityLinker", last=True)

    def extract_noun_phrases(self, text):

        data = pd.read_csv("tui.txt",  header=None)
        doc = self.spacy_web_sm(text)
        all_linked_entities = doc._.linkedEntities
        
        # save the entities and definitions is a dict; or what is the best way to save the name and definiton of the entity?
        entity_list = []
        entity_def = defaultdict(dict)

        for sent in doc.sents:
            sent._.linkedEntities.pretty_print()


        # for umls_ent in doc.noun_chunks:
        #     chunk_text = umls_ent.text
        #     spacy_web_doc = self.spacy_web_sm(chunk_text)
        #     all_linked_entities = doc._.linkedEntities

        #     if spacy_web_doc.ents:
        #         
        #         for each_extracted_chunk in spacy_web_doc.ents:
        #             entity_list.append(each_extracted_chunk.text)

        return entity_list

    
    def link_with_source(self, extracted_entities):

        entity_list = []
        entity_def = defaultdict(dict)

        for each_entity in extracted_entities:
            try: 
                doc = self.spacy_web_sm(each_entity)
            except: 
                if isinstance(each_entity, list):
                    each_entity = each_entity[0]
                elif isinstance(each_entity, dict):
                    each_entity = each_entity["name"]
                    doc = self.spacy_web_sm(each_entity)

            if doc._.linkedEntities.entities:
                definitions = doc._.linkedEntities.entities[0].get_description()
            else:
                definitions = "None"
            entity_def[each_entity] = definitions
    

        return entity_def


    def extract_noun_phrases_and_link_with_umls_remove_repeats(self, text, extracted_entities):

        data = pd.read_csv("tui.txt",  header=None)
        list_of_tui = set(data[0].to_list()) 

        doc = self.spacy_web_sm(text)
        linker = self.spacy_web_sm.get_pipe("entityLinker")
    
        # save the entities and definitions is a dict; or what is the best way to save the name and definiton of the entity?
        entity_def = defaultdict(dict)

        for umls_ent in doc.noun_chunks:
            chunk_text = umls_ent.text

            if chunk_text in extracted_entities:
                pass
            else: 
                spacy_web_doc = self.spacy_web_sm(chunk_text)
                if spacy_web_doc.ents:
                    for each_extracted_chunk in spacy_web_doc.ents:
                        # if chunk_text != each_extracted_chunk.text:
                        #     pass 
                        # else:
                            # this loop is for multiple top k'S in decreasing order but in this case I only need one. so I am not looping and taking only one.
                            # for umls_ent_cui in entity._.kb_ents:
                            #     if set(linker.kb.cui_to_entity[umls_ent_cui[0]].types).intersection(list_of_tui):
                            #         entity_def[linker.kb.cui_to_entity[umls_ent_cui[0]].canonical_name] = linker.kb.cui_to_entity[umls_ent_cui[0]].definition
                            # for some entities, each_extracted_chunk._.kb_ents is empty? 
                        if each_extracted_chunk._.kb_ents:
                            umls_ent_cui = each_extracted_chunk._.kb_ents[0]
                            if set(linker.kb.cui_to_entity[umls_ent_cui[0]].types).intersection(list_of_tui):
                                entity_def[linker.kb.cui_to_entity[umls_ent_cui[0]].canonical_name] = linker.kb.cui_to_entity[umls_ent_cui[0]].definition
 
        return entity_def


    def extract_noun_phrases_and_link_with_umls(self, text):

        data = pd.read_csv("tui.txt",  header=None)
        list_of_tui = set(data[0].to_list()) 

        doc = self.spacy_web_sm(text)
        linker = self.spacy_web_sm.get_pipe("entityLinker")
        
        # save the entities and definitions is a dict; or what is the best way to save the name and definiton of the entity?
        entity_def = defaultdict(dict)
        for umls_ent in doc.noun_chunks:
            chunk_text = umls_ent.text
            spacy_web_doc = self.spacy_web_sm(chunk_text)
            if spacy_web_doc.ents:
                for each_extracted_chunk in spacy_web_doc.ents:
                    # if chunk_text != each_extracted_chunk.text:
                    #     pass 
                    # else:
                        # this loop is for multiple top k'S in decreasing order but in this case I only need one. so I am not looping and taking only one.
                        # for umls_ent_cui in entity._.kb_ents:
                        #     if set(linker.kb.cui_to_entity[umls_ent_cui[0]].types).intersection(list_of_tui):
                        #         entity_def[linker.kb.cui_to_entity[umls_ent_cui[0]].canonical_name] = linker.kb.cui_to_entity[umls_ent_cui[0]].definition
                        # for some entities, each_extracted_chunk._.kb_ents is empty? 
                    if each_extracted_chunk._.kb_ents:
                        umls_ent_cui = each_extracted_chunk._.kb_ents[0]
                        if set(linker.kb.cui_to_entity[umls_ent_cui[0]].types).intersection(list_of_tui):
                            entity_def[linker.kb.cui_to_entity[umls_ent_cui[0]].canonical_name] = linker.kb.cui_to_entity[umls_ent_cui[0]].definition

        return entity_def


    

    

