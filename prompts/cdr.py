DEF_JSON = ("""Chemicals: A chemical substance is a form of matter having constant chemical composition and characteristic properties. 
    Chemical substances cannot be separated into their constituent elements by physical separation methods, i.e., without breaking chemical bonds. 
    Chemical substances can be simple substances (substances consisting of a single chemical element) chemical compounds, or alloys.
    Diseases: A disease is a particular abnormal condition that negatively affects the structure or function of all or part of an organism, and that is not immediately due to any external injury. Diseases are often known to be medical conditions that are associated with specific signs and symptoms. A disease may be caused by external factors such as pathogens or by internal dysfunctions.
    Given the sentence from an abstract, extract all the chemical and diseases and return as json.
    Do not say anything like  "Here is the extracted", just output the json of the format {"chemicals": list of extracted chemicals , "diseases": list of diseases} """
)
 
TEXT_JSON = ("""
    Given the sentence from an abstract, extract all the chemical and diseases and return as json.
    Do not say anything like  "Here is the extracted", just output the json of the format {"chemicals": list of extracted chemicals , "diseases":
    list of diseases}"
    """
)

CODE_DEF = (
    """
    def named_entity_recognition(input_text):
    \"""  
    Given a sentence from an abstract, and the definitions of chemicals and diseases, extract all the chemicals and diseases. 
    Chemicals: A chemical substance is a form of matter having constant chemical composition and characteristic properties. Chemical substances cannot be separated into their constituent elements by physical separation methods, i.e., without breaking chemical bonds. Chemical substances can be simple substances (substances consisting of a single chemical element) chemical compounds, or alloys.
    Diseases: A disease is a particular abnormal condition that negatively affects the structure or function of all or part of an organism, and that is not immediately due to any external injury. Diseases are often known to be medical conditions that are associated with specific signs and symptoms. A disease may be caused by external factors such as pathogens or by internal dysfunctions.
    A chemical entity is a dictionary of the format {"text": "extracted entities", "type": "chemicals"}
    A disease entity is a dictionary of the format {"text": "extracted entities", "type": "diseases"}
    Find all the entites in input_text and append only the entities one by one. Do not output any other information to entity_list or any comments. \"""
    input text = """
)

CODE_TEXT = (
    """
    def named_entity_recognition(input_text):
    \"""  
    Given a sentence from an abstract, and the definitions of chemicals and diseases, extract all the chemicals and diseases. 
    A chemical entity is a dictionary of the format {"text": "extracted entities", "type": "chemicals"}
    A disease entity is a dictionary of the format {"text": "extracted entities", "type": "diseases"}
    Find all the entites in input_text and append only the entities one by one. Do not output any other information to entity_list or any comments. \"""
    input text = """
)

GPT_DEF = (
    """Chemicals: A chemical substance is a form of matter having constant chemical composition and characteristic properties. Chemical substances cannot be separated into their constituent elements by physical separation methods, i.e., without breaking chemical bonds. Chemical substances can be simple substances (substances consisting of a single chemical element) chemical compounds, or alloys.
    Diseases: A disease is a particular abnormal condition that negatively affects the structure or function of all or part of an organism, and that is not immediately due to any external injury. Diseases are often known to be medical conditions that are associated with specific signs and symptoms. A disease may be caused by external factors such as pathogens or by internal dysfunctions.
    Given the sentence from an abstract, extract all the chemicals and diseases and return as a json. \n Sentence: """    
)

GPT_TEXT = ("""
    Given the sentence from an abstract, extract all the chemicals and diseases and return as a json. \n Sentence:
    """
)

PROMPT_ITER = (
    """Using the following examples of chemicals and diseases as a reference, correct errors if any, and output the final json only.  Please only output the final json. 
    Please remember this is an extraction task, only extract entities from the 'Sentence' and as they appear in the 'Sentence'."""
)