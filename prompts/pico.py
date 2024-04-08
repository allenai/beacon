DEF_JSON = ("""
    Population: Identify the specific characteristics or criteria defining the population group mentioned in the research question. Focus on demographic details, health conditions, age range, gender, sample size or any other relevant factors.
    Intervention: Determine the treatment, procedure, therapy, or action (including behavioral, educational, physical, pharmacological or surgical) that is being investigated or implemented in the study mentioned.
    Comparator: Identify the group or condition against which the intervention in the research question is being compared. It could be a different treatment, placebo, control group, standard of care, or any other relevant comparison.
    Outcome: Determine the specific outcome measure or endpoint that is being assessed or measured to evaluate the effectiveness or impact of the intervention mentioned including adverse effects, mental outcomes or mortality, pain or physical outcomes.
    Given a clinical trial abstract, extract the Population, Intervention, Comparator and Outcome spans and return them a json.
    Do not output anyting that say - Here is the...., just output the json. 
    """
)

TEXT_JSON = ("""
    Given a clinical trial abstract, extract the Population, Intervention, Comparator and Outcome spans and return them as json.
    Do not say anything like  "Here is the extracted", just output the json only output the json of the format {"population": list of extracted population entities , "intervention": 
    list of extracted intervention entities, "comparator": list of extracted comparator entities, "outcome": list of extracted outcome entities}
    """
)

CODE_TEXT = (
    """
    def named_entity_recognition(input_text):
        \""" 
        Extract a list of all population, intervention, comparator and outcomespans from input_text.
        A population entity is a dictionary of the format {"text": "entities", "type": "population"}
        A intervention entity is a dictionary of the format {"text": "entities", "type": "intervention"}
        A comparator entity is a dictionary of the format {"text": "entities", "type": "comparator"}
        A outcome entity is a dictionary of the format {"text": "entities", "type": "outcome"}
        Find all entites in input_text and append them to entity_list one by one. Do not output any other information to entity_list or any comments. \""" 
    """
)

CODE_DEF = (
    """
    def named_entity_recognition(input_text):
        \""" 
        Population: Identify the specific characteristics or criteria defining the population group mentioned in the research question. Focus on demographic details, health conditions, age range, gender, sample size or any other relevant factors.
        Intervention: Determine the treatment, procedure, therapy, or action (including behavioral, educational, physical, pharmacological or surgical) that is being investigated or implemented in the study mentioned.
        Comparator: Identify the group or condition against which the intervention in the research question is being compared. It could be a different treatment, placebo, control group, standard of care, or any other relevant comparison.
        Outcome: Determine the specific outcome measure or endpoint that is being assessed or measured to evaluate the effectiveness or impact of the intervention mentioned including adverse effects, mental outcomes or mortality, pain or physical outcomes.

        Given the above definitions of entities, Extract a list of all population, intervention, comparator and outcomespans from input_text.
        A population entity is a dictionary of the format {"text": "entities", "type": "population"}
        A intervention entity is a dictionary of the format {"text": "entities", "type": "intervention"}
        A comparator entity is a dictionary of the format {"text": "entities", "type": "comparator"}
        A outcome entity is a dictionary of the format {"text": "entities", "type": "outcome"}
        Find all entites in input_text and append them to entity_list one by one. Do not output any other information to entity_list or any comments. \""" 
    """
)

GPT_TEXT = (
    "\nGiven a clinical trial abstract, extract the Population, Intervention, Comparator and Outcome spans and return them a json. \n "
)

GPT_CODE = (
    """
    Given the sentence, extract a list of all population, intervention, comparator and outcome spans and return them as Json.
    A population entity is a dictionary of the format {"text": entity, "type": "population"}
    An intervention entity is a dictionary of the format {"text": entity, "type": "intervention"}
    A comparator entity is a dictionary of the format {"text": entity, "type": "comparator"}
    A outcome entity is a dictionary of the format {"text": entity, "type": "outcome"}
    Do not repeat the input or say "Here are the entities", just output the json of the current sentence.
    """
)