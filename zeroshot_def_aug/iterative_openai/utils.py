from calls import openai_call


BOOLEAN_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "boolean", "description": "true or false."},
    },
    "required": ["answer"],
}

BOOLEAN_PROMPT = """
Using the following definition as a reference, answer only true / false for the following question. Please only output the final answer. 
Entity : {entity}, Definition : {definition}
Does this entity "{entity}" belong to the entity type "{entity_type}"?
"""

BOOLEAN_PROMPT_NO_DEF = """
Answer only true / false for the following question. Please only output the final answer. 
Does this entity "{entity}" belong to the entity type "{entity_type}"?
"""

BOOLEAN_NOUN_PROMPT = """
Using the following definition as a reference, answer only true / false for the following question. Please only output the final answer. 
Entity : {noun_phrase}, Definition : {definition}
Does this "{noun_phrase}" belong to any of the {entity_types}?
"""

BOOLEAN_NOUN_PROMPT_NO_DEF = """
Answer only true / false for the following question. Please only output the final answer. 
Does this entity "{noun_phrase}" belong to the entity type "{entity_types}"?
"""


def get_bool_openai_response(query):
    response = openai_call.generate_text(
        query, BOOLEAN_SCHEMA, temperature=0, max_tokens=256
    )
    if "true" in response.lower():
        # remove the entity from extracted sent
        return True
    elif "false" in response.lower():
        return False
    else:
        print("Response does not have true/false", response)
        return False


def is_entity_in_entity_type(entity, definition, entity_type):
    #query_message = [{"role": "user", "content": BOOLEAN_PROMPT.format(entity=entity, definition=definition, entity_type=entity_type)}]
    query_message = [{"role": "user", "content": BOOLEAN_PROMPT_NO_DEF.format(entity=entity, entity_type=entity_type)}]
    return get_bool_openai_response(query_message)
    

def get_entity_types_for_entity(entity, definition, ENTITY_TYPES):
    output_entity_types = []
    for entity_type in ENTITY_TYPES:
        if is_entity_in_entity_type(entity, definition, entity_type):
            output_entity_types.append(entity_type)
    
    return output_entity_types