DEF_JSON = ("""
    Diseases: A disease is a particular abnormal condition that negatively affects the structure or function of all or part of an organism, and that is not immediately due to any external injury. Diseases are often known to be medical conditions that are associated with specific signs and symptoms.
    A disease may be caused by external factors such as pathogens or by internal dysfunctions.
    Given the definition of disease and sentence from an abstract, extract the diseases as a json with "diseases" as the key and the extractions as list of values.
    """
)

TEXT_JSON = ("""
    Given the definition of disease and sentence from an abstract, and return as json. Do not say anything like  "Here is the extracted", just output the json only output the json of the format {"diseases": 
    list of diseases}
    """
)

DEF_CODE = (
    """
    def named_entity_recognition(input_text): 
    \""" 
        Diseases: A disease is a particular abnormal condition that negatively affects the structure or function of all or part of an organism, and that is not immediately due to any external injury. Diseases are often known to be medical conditions that are associated with specific signs and symptoms. A disease may be caused by external factors such as pathogens or by internal dysfunctions.
        Given the definition of diseases, extract all the diseases from the input_text. 
        A Disease entity is a dictionary of the format: {"text: "disease entities", "type", "disease"}
        Find all entites in input_text and append them to entity_list one by one. Do not output any other information to entity_list or any comments.
    \"""
    input_text = """
)

TEXT_CODE = (
    """
    def named_entity_recognition(input_text): 
    \""" 
        Extract all the diseases from the input_text. 
        A Disease entity is a dictionary of the format: {"text: "disease entities", "type", "disease"}
        Find all entites in input_text and append them to entity_list one by one. Do not output any other information to entity_list or any comments.
    \"""
    input_text = """
)

GPT_DEF = (
    """Diseases: A disease is a particular abnormal condition that negatively affects the structure or function of all or part of an organism, and that is not immediately due to any external injury. Diseases are often known to be medical conditions that are associated with specific signs and symptoms.
    A disease may be caused by external factors such as pathogens or by internal dysfunctions.
    Given the definition of disease and sentence from an abstract, extract the diseases and return as a json.
    """
)

GPT_TEXT = (
    "Given the following sentence from an abstract, extract all the disease spans and return as a json, where the key is 'diseases' and the value is a list of diseases. "  

)