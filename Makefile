run:
	PYTHONPATH=./ python ${TYPE}/${MODEL}/${DATASET}.py

eval: 
	PYTHONPATH=./ python evaluation/${OUTPUT_TYPE}/${DATASET}.py

pip-install:
	pip install -r requirements/requirements.txt

