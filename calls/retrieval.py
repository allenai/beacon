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

class KnowledgeRetrieval(object):
    def __init__(self):

        self.spacy_web_sm = spacy.load("en_core_web_sm")
        self.scispacy_md = spacy.load("en_core_sci_md")
        self.scispacy_md.add_pipe("merge_noun_chunks")
        self.scispacy_md.add_pipe("scispacy_linker", config={"linker_name": "umls", "resolve_abbreviations": True})


    def extract_noun_phrases_and_link_with_umls_remove_repeats(self, text, extracted_entities):

        data = pd.read_csv("tui.txt",  header=None)
        list_of_tui = set(data[0].to_list()) 

        doc = self.spacy_web_sm(text)
        linker = self.scispacy_md.get_pipe("scispacy_linker")
    
        # save the entities and definitions is a dict; or what is the best way to save the name and definiton of the entity?
        entity_def = defaultdict(dict)

        for umls_ent in doc.noun_chunks:
            chunk_text = umls_ent.text
            if chunk_text in extracted_entities:
                pass
            else: 
                scispacy_doc = self.scispacy_md(chunk_text)
                if scispacy_doc.ents:
                    for each_extracted_chunk in scispacy_doc.ents:
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
        linker = self.scispacy_md.get_pipe("scispacy_linker")
        
        # save the entities and definitions is a dict; or what is the best way to save the name and definiton of the entity?
        entity_def = defaultdict(dict)
        for umls_ent in doc.noun_chunks:
            chunk_text = umls_ent.text
            scispacy_doc = self.scispacy_md(chunk_text)
            if scispacy_doc.ents:
                for each_extracted_chunk in scispacy_doc.ents:
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


    def link_with_umls(self, extracted_entities):

        data = pd.read_csv("tui.txt",  header=None)
        list_of_tui = set(data[0].to_list())

        linker = self.scispacy_md.get_pipe("scispacy_linker")        
        # save the entities and definitions is a dict; or what is the best way to save the name and definiton of the entity?
        entity_def = defaultdict(dict)
        for chunk_text in extracted_entities:
            scispacy_doc = self.scispacy_md(chunk_text)
            #  in case it mathes with multiple entiites in scispacy model? 
            if scispacy_doc.ents:
                for each_extracted_chunk in scispacy_doc.ents:
                    # extract tui 
                    # If there are different types; using intersection
                    if each_extracted_chunk._.kb_ents:
                        umls_ent_cui = each_extracted_chunk._.kb_ents[0]
                        if set(linker.kb.cui_to_entity[umls_ent_cui[0]].types).intersection(list_of_tui):
                            entity_def[linker.kb.cui_to_entity[umls_ent_cui[0]].canonical_name] = linker.kb.cui_to_entity[umls_ent_cui[0]].definition

        return entity_def

    def extract_noun_phrases(self, text):

        data = pd.read_csv("tui.txt",  header=None)
        list_of_tui = set(data[0].to_list()) 

        doc = self.spacy_web_sm(text)
        linker = self.scispacy_md.get_pipe("scispacy_linker")
        entity_list = []
        # save the entities and definitions is a dict; or what is the best way to save the name and definiton of the entity?
        entity_def = defaultdict(dict)
        for umls_ent in doc.noun_chunks:
            chunk_text = umls_ent.text
            scispacy_doc = self.scispacy_md(chunk_text)
            if scispacy_doc.ents:
                for each_extracted_chunk in scispacy_doc.ents:
                    entity_list.append(each_extracted_chunk.text)

        return entity_list

