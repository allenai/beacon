DEF_JSON = ("""
    entity_type = [Event, Activity, Behavior , Social Behavior, Individual Behavior, Daily  or Recreational Activity, Occupational Activity, Health care Activity, Research Activity, Government or Regulatory Activity, Educational Activity, Machine Activity. Phenomenon or Process, Injury or Poisoning, Human-caused Phenomenon or Process, Environmental Effects of Humans, Natural Phenomenon or Process, Biological Function
    Entity, Physical Object, Organism, Virus, Bacterium, Archaeon, Eukaryote, Anatomical Structure, Manufactured Object, Medical Device, Research Device, Clinical Drug, Substance, Body Structure, Chemical, Food, Conceptual Identity, Organism Attribute, Clinical Attribute, Finding, Idea or Concept, Temporal Concept, Qualitative Concept, Quantitative Concept, Spatial Concept, Functional Concept, Body System, Occupation or Discipline, Biomedical Occupation or Discipline, Organization, Group, Professional or Organizational group, Population Group, Family Group, Age Group, Patient or Disabled group, Group Attribute, Intellectual Product, Language]
    Given a sentence from an abstract, extract all the entities which can be found in a UMLS and that could be used in a knowledge base.
    Do not say anything like  "Here is the extracted", just output the json only output the json of the format {"entities": list of extracted entities}
    """
)

TEXT_JSON = ("""
    Given a sentence from an abstract, extract all the entities which can be found in a UMLS and that could be used in a knowledge base.
    Do not say anything like  "Here is the extracted", just output the json only output the json of the format {"entities": list of extracted entities}
    """
)

CODE_TEXT = (
    """
    def named_entity_recognition(input_text):
    \""" 
    Given a sentence from an abstract and all the entity types above, extract all the entities which can be found in a UMLS and that could be used in a knowledge base.
    Each entity is a dictionary of the format {"text": "entities"}
    Find all the entities in input_text and append only the entities and not the type of the entity to entity_list one by one. Do not output any other information to entity_list or any comments.\"""
    input text = """
)

CODE_DEF = (
    """
    def named_entity_recognition(input_text):
    \""" 
    
    entity_type = [Event, Activity, Behavior , Social Behavior, Individual Behavior, Daily  or Recreational Activity, Occupational Activity, Health care Activity, Research Activity, Government or Regulatory Activity, Educational Activity, Machine Activity. Phenomenon or Process, Injury or Poisoning, Human-caused Phenomenon or Process, Environmental Effects of Humans, Natural Phenomenon or Process, Biological Function
    Entity, Physical Object, Organism, Virus, Bacterium, Archaeon, Eukaryote, Anatomical Structure, Manufactured Object, Medical Device, Research Device, Clinical Drug, Substance, Body Structure, Chemical, Food, Conceptual Identity, Organism Attribute, Clinical Attribute, Finding, Idea or Concept, Temporal Concept, Qualitative Concept, Quantitative Concept, Spatial Concept, Functional Concept, Body System, Occupation or Discipline, Biomedical Occupation or Discipline, Organization, Group, Professional or Organizational group, Population Group, Family Group, Age Group, Patient or Disabled group, Group Attribute, Intellectual Product, Language]
    Given a sentence from an abstract and all the entity types above, extract all the entities which can be found in a UMLS and that could be used in a knowledge base.
    Each entity is a dictionary of the format {"text": "entities"}
    Find all the entities in input_text and append only the entities and not the type of the entity to entity_list one by one. Do not output any other information to entity_list or any comments.\"""
    input text = """
)

GPT_DEF = (
    """Given the sentence from an abstract, extract the entities which can be found in a UMLS and that could be used in a knowledge base. \n Sentence: 
    [Event, Activity, Behavior , Social Behavior, Individual Behavior, Daily  or Recreational Activity, Occupational Activity, Health care Activity, Research Activity, Government or Regulatory Activity, Educational Activity, Machine Activity. Phenomenon or Process, Injury or Poisoning, Human-caused Phenomenon or Process, Environmental Effects of Humans, Natural Phenomenon or Process, Biological Function
    Entity, Physical Object, Organism, Virus, Bacterium, Archaeon, Eukaryote, Anatomical Structure, Manufactured Object, Medical Device, Research Device, Clinical Drug, Substance, Body Structure, Chemical, Food, Conceptual Identity, Organism Attribute, Clinical Attribute, Finding, Idea or Concept, Temporal Concept, Qualitative Concept, Quantitative Concept, Spatial Concept, Functional Concept, Body System, Occupation or Discipline, Biomedical Occupation or Discipline, Organization, Group, Professional or Organizational group, Population Group, Family Group, Age Group, Patient or Disabled group, Group Attribute, Intellectual Product, Language]
    Given a sentence from an abstract, extract all the entities which can be found in a UMLS and that could be used in a knowledge base and from the entity_type list above. 
    """
)

GPT_TEXT = (
    """
    Given a sentence from an abstract, extract all the entities from the abstract which can be found in a UMLS.  
    """
)