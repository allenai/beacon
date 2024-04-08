CODE_DEF = (
    """
    def named_entity_recognition(input_text):
    \"""  
    Given a sentence from an abstract, and definitions of chemicals and proteins extract all the chemicals and proteins. 
    Chemicals: A chemical substance is a form of matter having constant chemical composition and characteristic properties. Chemical substances cannot be separated into their constituent elements by physical separation methods, i.e., without breaking chemical bonds. Chemical substances can be simple substances (substances consisting of a single chemical element) chemical compounds, or alloys.
    Proteins: Proteins are large biomolecules and macromolecules that comprise one or more long chains of amino acid residues. Proteins perform a vast array of functions within organisms, including catalysing metabolic reactions, DNA replication, responding to stimuli, providing structure to cells and organisms, and transporting molecules from one location to another.
    A chemical entity is a dictionary of the format {"text": "entities", "type": "chemicals"}
    A proteins entity is a dictionary of the format {"text": "entities", "type": "proteins"}
    Find all the entites in input_text and append only the entities and not other information to entity_list one by one. Do not output any other information to entity_list or any comments.\"""
    input text = """
)

CODE_TEXT = (
    """
    def named_entity_recognition(input_text):
    \"""  
    Given a sentence from an abstract, and definitions of chemicals and proteins extract all the chemicals and proteins. 
    A chemical entity is a dictionary of the format {"text": "entities", "type": "chemicals"}
    A proteins entity is a dictionary of the format {"text": "entities", "type": "proteins"}
    Find all the entites in input_text and append only the entities and not other information to entity_list one by one. Do not output any other information to entity_list or any comments.\"""
    input text = """
)


GPT_TEXT = (
    "Given the following sentence from an abstract, extract all the chemical and gene mentions and return as a json. "
)


GPT_DEF = (
    """Chemicals: A chemical substance is a form of matter having constant chemical composition and characteristic properties. Chemical substances cannot be separated into their constituent elements by physical separation methods, i.e., without breaking chemical bonds. Chemical substances can be simple substances (substances consisting of a single chemical element) chemical compounds, or alloys.
    Proteins: Proteins are large biomolecules and macromolecules that comprise one or more long chains of amino acid residues. Proteins perform a vast array of functions within organisms, including catalysing metabolic reactions, DNA replication, responding to stimuli, providing structure to cells and organisms, and transporting molecules from one location to another.
    Given the following sentence from an abstract, extract all the chemical and gene mentions and return as a json. """
)

TEXT_JSON = (
    "Given the following sentence from an abstract, extract all the chemical and gene mentions and return as a json. "
)


DEF_JSON = (
    """Chemicals: A chemical substance is a form of matter having constant chemical composition and characteristic properties. Chemical substances cannot be separated into their constituent elements by physical separation methods, i.e., without breaking chemical bonds. Chemical substances can be simple substances (substances consisting of a single chemical element) chemical compounds, or alloys.
    Proteins: Proteins are large biomolecules and macromolecules that comprise one or more long chains of amino acid residues. Proteins perform a vast array of functions within organisms, including catalysing metabolic reactions, DNA replication, responding to stimuli, providing structure to cells and organisms, and transporting molecules from one location to another.
    Given the following sentence from an abstract, extract all the chemical and gene mentions and return as a json. """
)

