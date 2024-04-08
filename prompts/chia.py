DEF_JSON = ("""
    extract the following entity types if present: Condition, Device, Drug, Measurement, Mood, Multiplier, Negation, Observation, Person, Procedure, Qualifier, Reference_point, Scope, Temporal, Value, Visit (in alphabetical order) if exists and return them as json.
    Here are the definitions of these entities:
    Condition: the presence of a disease or medical condition stated as a diagnosis, a sign, or a symptom.
    Device: exposure to a foreign physical object or instrument which is used for diagnostic or therapeutic purposes through a mechanism beyond chemical action; includes implantable objects, medical equipment and supplies other instruments used in medical procedures and material used in clinical care.
    Drug: a biochemical substance formulated in such a way that when administered to a Person it will exert a certain physiological effect; includes prescription and over-the-counter medicines, vaccines, and large-molecule biologic therapies.
    Measurement: structured values (numerical or categorical) obtained through systematic and standardized examination or testing of a Person or Person's sample.
    Observation: clinical facts about a Person obtained in the context of examination, questioning or a procedure; includes any data that cannot be represented by any other domains, such as social and lifestyle facts, medical history, family history, etc.
    Person: demographic information used to describe a Person, including age, gender, race, ethnicity, etc.
    Procedure: activities or processes ordered by, or carried out by, a healthcare provider on the patient to have a diagnostic or therapeutic purpose.
    Visit: location or setting in which a Person is receiving medical services from one or more providers, including outpatient care, inpatient confinement, emergency room, and long-term care.
    Temporal: represents a point in the line of time.
    Value: represents a structured value, either as a number or as a concept. When specifying a number value, the only components accepted inside its free text are: logical operator, numeral, unit of measure.
    Scope: distributes all relationships coming to it, and departing from it, to all roots of annotation graphs included in it. The critical purpose of the Scope is to allow the creation of layered logic (e.g., A and (B or C)), but it is also used as a shortcut to expedite the annotation task when multiple entities are connected in "combinatorial" fashion, such as multiple Qualifiers being all applied to multiple Conditions.
    Negation: provokes a Boolean negation on its parent entity. If the truth value of the parent evaluates to false, it then becomes true, and vice-versa.
    Qualifier: subsets the meaning of its parent by imposing a further constraint. The value of a Qualifier oftentimes serves as a supplement to the value of its parent, that is, it may be the case that the free text contained by a Qualifier can be concatenated with the free text contained by its parent to form one string that can then be linked to a single code. Another common case is for Qualifiers to express the anatomic location of a Condition or the severity of a Condition.
    Multiplier: specifies either dosage of a Drug entity, or repetition type of entity.
    Reference_point: Always comes downstream (usually directly) from a parent Temporal, and specifies a concept whose timestamp anchors that Temporal.
    Mood: transforms the meaning of its parent into a different kind of statement that is not about the literal presence of the parent.  \n
    Do not say anything like  "Here is the extracted", just output the json only output the json of the format {
    "condition": list of extracted conditions,
    "device": list of extracted devices
    "drug": list of drugs
    "measurement" : list if measurements
    "observation" : list of observations
    "person" : list of observations
    "procedure" : list of procedures
    "visit": list of visits
    "temporal" : list of temporal entities
    "value" : list of values
    "scope" : list of scope entities
    "negation" : list of negatoions
    "qualifier" : list of qualifiers
    "multiplier" : list of multipliers
    "reference_point" : list of reference points
    "mood": list of mood entities}
    """
)

TEXT_JSON = ("""
    extract the following entity types if present: Condition, Device, Drug, Measurement, Mood, Multiplier, Negation, Observation, Person, Procedure, Qualifier, Reference_point, Scope, Temporal, Value, Visit (in alphabetical order) if exists and return them as json.
    Do not say anything like  "Here is the extracted", just output the json only output the json of the format {
    "condition": list of extracted conditions,
    "device": list of extracted devices
    "drug": list of drugs
    "measurement" : list if measurements
    "observation" : list of observations
    "person" : list of observations
    "procedure" : list of procedures
    "visit": list of visits
    "temporal" : list of temporal entities
    "value" : list of values
    "scope" : list of scope entities
    "negation" : list of negatoions
    "qualifier" : list of qualifiers
    "multiplier" : list of multipliers
    "reference_point" : list of reference points
    "mood": list of mood entities}
    """
)

CODE_TEXT = (
    """
    extract the following entity types if present: Condition, Device, Drug, Measurement, Mood, Multiplier, Negation, Observation, Person, Procedure, Qualifier, Reference_point, Scope, Temporal, Value, Visit (alphabetical) from input_text.
    Each entity is a dictionary of the format {"text": "extracted entities", "type": "type of the entity"}
    Find all entites in input_text and append them to entity_list one by one.  Do not output any other information to entity_list or any unnncessary comments.\"""
    """
)

CODE_DEF = (
    """
        extract the following entity types if present: Condition, Device, Drug, Measurement, Mood, Multiplier, Negation, Observation, Person, Procedure, Qualifier, Reference_point, Scope, Temporal, Value, Visit (alphabetical) from input_text.
        Condition: the presence of a disease or medical condition stated as a diagnosis, a sign, or a symptom.edure; includes any data that cannot be represented by any other domains, such as social and lifestyle facts, medical history, family history, etc.
        Person: demographic information used to describe a Person, including age, gender, race, ethnicity, etc.
        Procedure: activities or processes ordered by, or carried out by, a healthcare provider on the patient to have a diagnostic or therapeutic purpose.
        Visit: location or setting in which a Person is receiving medical services from one or more providers, including outpatient care, inpatient confinement, emergency room, and long-term care.
        Temporal: represents a point in the line of time. 
        Value: represents a structured value, either as a number or as a concept. When specifying a number value, the only components accepted inside its free text are: logical operator, numeral, unit of measure.
        Scope: distributes all relationships coming to it, and departing from it, to all roots of annotation graphs included in it. The critical purpose of the Scope is to allow the creation of layered logic (e.g., A and (B or C)), but it is also used as a shortcut to expedite the annotation task when multiple entities are connected in "combinatorial" fashion, such as multiple Qualifiers being all applied to multiple Conditions. 
        Negation: provokes a Boolean negation on its parent entity. If the truth value of the parent evaluates to false, it then becomes true, and vice-versa.
        Qualifier: subsets the meaning of its parent by imposing a further constraint. The value of a Qualifier oftentimes serves as a supplement to the value of its parent, that is, it may be the case that the free text contained by a Qualifier can be concatenated with the free text contained by its parent to form one string that can then be linked to a single code. Another common case is for Qualifiers to express the anatomic location of a Condition or the severity of a Condition.
        Multiplier: specifies either dosage of a Drug entity, or repetition type of entity.
        Reference_point: Always comes downstream (usually directly) from a parent Temporal, and specifies a concept whose timestamp anchors that Temporal.
        Mood: transforms the meaning of its parent into a different kind of statement that is not about the literal presence of the parent. \n
        Device: exposure to a foreign physical object or instrument which is used for diagnostic or therapeutic purposes through a mechanism beyond chemical action; includes implantable objects, medical equipment and supplies other instruments used in medical procedures and material used in clinical care.
        Drug: a biochemical substance formulated in such a way that when administered to a Person it will exert a certain physiological effect; includes prescription and over-the-counter medicines, vaccines, and large-molecule biologic therapies.
        Measurement: structured values (numerical or categorical) obtained through systematic and standardized examination or testing of a Person or Person's sample.
        Observation: clinical facts about a Person obtained in the context of examination, questioning or a procedure.
        
        Each entity is a dictionary of the format {"text": "extracted entities", "type": "type of the entity"}
        Find all entites in input_text and append them to entity_list one by one.  Do not output any other information to entity_list or any unnncessary comments.\"""
    """
)

GPT_DEF = (
    """extract the following entity types if present: Condition, Device, Drug, Measurement, Mood, Multiplier, Negation, Observation, Person, Procedure, Qualifier, Reference_point, Scope, Temporal, Value, Visit (alphabetical) from input_text and return them a json. \n 
    Each entity is a dictionary of the format {"text": "extracted entities", "type": "type of the entity"}
    Condition: the presence of a disease or medical condition stated as a diagnosis, a sign, or a symptom.edure; includes any data that cannot be represented by any other domains, such as social and lifestyle facts, medical history, family history, etc.
    Person: demographic information used to describe a Person, including age, gender, race, ethnicity, etc.
    Procedure: activities or processes ordered by, or carried out by, a healthcare provider on the patient to have a diagnostic or therapeutic purpose.
    Visit: location or setting in which a Person is receiving medical services from one or more providers, including outpatient care, inpatient confinement, emergency room, and long-term care.
    Temporal: represents a point in the line of time. 
    Value: represents a structured value, either as a number or as a concept. When specifying a number value, the only components accepted inside its free text are: logical operator, numeral, unit of measure.
    Scope: distributes all relationships coming to it, and departing from it, to all roots of annotation graphs included in it. The critical purpose of the Scope is to allow the creation of layered logic (e.g., A and (B or C)), but it is also used as a shortcut to expedite the annotation task when multiple entities are connected in "combinatorial" fashion, such as multiple Qualifiers being all applied to multiple Conditions. 
    Negation: provokes a Boolean negation on its parent entity. If the truth value of the parent evaluates to false, it then becomes true, and vice-versa.
    Qualifier: subsets the meaning of its parent by imposing a further constraint. The value of a Qualifier oftentimes serves as a supplement to the value of its parent, that is, it may be the case that the free text contained by a Qualifier can be concatenated with the free text contained by its parent to form one string that can then be linked to a single code. Another common case is for Qualifiers to express the anatomic location of a Condition or the severity of a Condition.
    Multiplier: specifies either dosage of a Drug entity, or repetition type of entity.
    Reference_point: Always comes downstream (usually directly) from a parent Temporal, and specifies a concept whose timestamp anchors that Temporal.
    Mood: transforms the meaning of its parent into a different kind of statement that is not about the literal presence of the parent. \n
    Device: exposure to a foreign physical object or instrument which is used for diagnostic or therapeutic purposes through a mechanism beyond chemical action; includes implantable objects, medical equipment and supplies other instruments used in medical procedures and material used in clinical care.
    Drug: a biochemical substance formulated in such a way that when administered to a Person it will exert a certain physiological effect; includes prescription and over-the-counter medicines, vaccines, and large-molecule biologic therapies.
    Measurement: structured values (numerical or categorical) obtained through systematic and standardized examination or testing of a Person or Person's sample.
    Observation: clinical facts about a Person obtained in the context of examination, questioning or a procedure."""   
)

GPT_TEXT = (
    """
    extract the following entity types if present: Condition, Device, Drug, Measurement, Mood, Multiplier, Negation, Observation, Person, Procedure, Qualifier, Reference_point, Scope, Temporal, Value, Visit (alphabetical) from input_text and return them a json. \n 
    Each entity is a dictionary of the format {"text": "extracted entities", "type": "type of the entity"}
    Sentence: 
    """
)
