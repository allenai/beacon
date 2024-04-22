# On-the-fly Definition Augmentation of LLMs for Biomedical NER
This repository contains code to run the inference and evaluation of NER as described in our NAACL 2024 paper: [On-the-fly Definition Augmentation of LLMs for Biomedical NER](https://arxiv.org/abs/2404.00152)

## Code Setup
This code was developed in python 3.9 using the libraries listed in environment.yml. The easiest way to run this code is to set up a conda environment using the .yml file via the following command:

```conda env create -f environment.yml```

Activate the conda environment using the command: ```conda activate fsdar```

In addition to environment setup, you will need to dowbnload datasets from huggingface (open source) [download here](https://huggingface.co/bigbio). 


## Datasets and Splits

Our paper evaluates the performance of these models on NER inference on the following datasets:

1. CDR
2. CHEMPROT
3. MEDM
4. NCBI
5. PICO
6. CHIA

These datasets are curated by [Fries et al.](https://arxiv.org/abs/2206.15076). Note that `data` directory contains the `document id` for the subsampled test split for each dataset. Please run the script to save the subsampled datasets before running `retrieval` and `fewshot_retrieval`. 

Additionally for `CHIA` dataset, we create train, validation and test splits with the most recent ones in the test set. We also release the `document ids` for each of these splits in `data/chia`. 

To run Definition Augmentation, please make sure you generte subsampled data using the Article IDs from `data` and then change the paths.

## Inference with open-sourced and closed-sourced models


To run inference:

```shell
make run TYPE=$TYPE MODEL=$MODEL DATASET=$DATASET
```

The following are different inference settings: 

TYPE: zeroshot, fewshot, zeroshot_def_aug, fewshot_def_aug
<br>
MODEL: openai, llama, claude
<br>
DATATSET: cdr, chemprot, ncbi, medm, pico, chia

### Fewshot 

To create the shots run

`fewshot/shot_selection` for each dataset and save these samples in DATA_DIR. Use this to run the `fewshot_def_aug`.

## Evaluation
To process the evaluation scrips, there are two different formats (JSON/CODE) which can be done using the following command:

```shell
make run OUTPUT_TYPE=$OUTPUT_TYPE DATASET=$DATASET
```

The following are different evaluation settings: 

OUTPUT_TYPE: eval_code, eval_json
<br>
DATATSET: cdr, chemprot, ncbi, medm, pico, chia

## Finetuned Model

### Data Formatting 
To create the data with he 5 shots we have used to fo few-shot run `finetuing_data/make_data.py`. These files follow the CoNLL 2003 format and consist of four space-separate columns. Each word must be placed in a separate line, with the four columns containing the word itself, its POS tag, its syntactic chunk tag, and its named entity tag. After each sentence, there must be an empty line. An example sentence would look as follows:

```
Acute           NN O O
low             NN O B-DIS
back            NN O I-DIS
pain            NN O I-DIS
during          NN O O
intravenous     NN O O
administration  NN O O
of              NN O O
amiodarone      NN O B-CHE
.               NN O O
```

### Running the model 
To finetune a Flan-XL model, run the following command 

```shell
python peft_llm_trainer.py --model_name_or_path google/flan-t5-xl --output_dir <OUTDIR/> --train_file <TRAIN_PATH/> --validation_file <VAL_PATH> --test_file <TEST_PATH/> --do_train --do_eval --do_predict --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --learning_rate 3e-5 --num_train_epoch 5 --save_steps 10 --logging_steps 10 --load_best_model_at_end --predict_with_generate --eval_steps 10 --evaluation_strategy steps
```

where TRAIN_PATH, VAL_PATH and TEST_PATH are where the CONLL format files are saved.

### Evaluation

Run the following command with the updated path to the output.

`python finetuning/eval.py` with the correct output paths.


If you face any issues with the code, models, or with reproducing our results, please contact monicam@allenai.org or raise an issue here.

If you find our code useful, please cite the following paper:

```
@misc{munnangi2024onthefly,
      title={On-the-fly Definition Augmentation of LLMs for Biomedical NER}, 
      author={Monica Munnangi and Sergey Feldman and Byron C Wallace and Silvio Amir and Tom Hope and Aakanksha Naik},
      year={2024},
      eprint={2404.00152},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
