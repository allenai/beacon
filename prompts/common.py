PROMPT_RET = "To assist you with extraction, here are the definitions of the extracted entities: \n"
PROMPT_NOUN = "Additionally, here are the definitions for some of the biomedical noun phrases from the sentence: \n"
PROMPT_END  = "\n Using these definitions only as a reference, add or remove incorrect entities from the output json only if you think the entities in the output json are wrong else don't change the output. Please remember this is an extraction task, only extract entities from the 'Sentence' and as they appear in the 'Sentence'. Please only output the final json, do not say 'here are the entities or here are the updates entities' or any other explanation."
PROMPT_END_NO_DEF  = "\n Output the same JSON without any changes. Please only output the final json, do not say 'here are the entities or here are the updates entities' or any other explanation."
CALIBRATION_PROMPT = ""

PROMPT_ITER = "Are you sure these entities are correct? Correct errors if any, and output the final json only."

PROMPT_DEF = "Given a sentence from a biomedical abstract, please provide definitions for some terms from the sentence, which could be understood by a researcher. \nSentence: "
PROMPT_DEF2 = "Following is the list of terms to define: "

CDR_EXAMPLES = """ Here are the examples of some chemicals :
    'Phenobarbitone', 'appetite suppressants', 'Nitrofurantoins', 'riboflavin', 'oral-contraceptive', 'fluorescein sodium', 'atropine sulfate', 'fluorescein', 'Apomorphine', 'streptomycin', 'PDTC', 'Vigabatrin', 'flurbiprofen',
    Here are the ecamples of some diseases :
    'tooth wear', 'sleep disturbances', 'dizziness', 'hypoxaemia', 'Parkinson disability', 'orthostatic hypotension', 'abnormal movements', 'intracranial vascular disturbances', 'generalized seizures', 'Argentine hemorrhagic fever', 'memory deficiency', 'Haematological toxicity', 'white matter edema',
    """