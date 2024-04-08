run:
	PYTHONPATH=./ python ${TYPE}/${MODEL}/${DATASET}.py

pip-install:
	pip install -r requirements/requirements.txt

