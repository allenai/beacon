from dataclasses import dataclass, field
from typing import Optional
import os
import subprocess
import evaluate
import nltk
import numpy as np
import nvidia_smi
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    set_seed,
)

from deepspeed.utils.zero_to_fp32 import (
    load_state_dict_from_zero_checkpoint,
    get_fp32_state_dict_from_zero_checkpoint,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    # checkpoint_dir: Optional[str] = field(
    #     metadata={"help": "Path to dumped deepspeed checkpoint"}
    # )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(
        default=None, metadata={"help": "Language id for summarization."}
    )

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."
        },
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the summaries (for summarization)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a jsonlines or csv file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`. "
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."
        },
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id. "
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )


def get_gpu_memory_usage():
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print(
            "Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(
                i,
                nvidia_smi.nvmlDeviceGetName(handle),
                100 * info.free / info.total,
                info.total,
                info.free,
                info.used,
            )
        )


if __name__ == "__main__":

    nvidia_smi.nvmlInit()

    # Additional arguments to provide (from the HuggingFace Trainer)
    # --model_name_or_path, --cache_dir, --train_file, --validation_file, --test_file, --output_dir,
    # --overwrite_output_dir, --do_train, --do_eval, --do_predict, --deepspeed, --per_device_train_batch_size,
    # --per_device_eval_batch_size, --learning_rate, --num_train_epochs, --logging_steps,
    # --save_steps, --fp16, --eval_steps, --load_best_model_at_end
    # --gradient_checkpointing if OOM
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load config, tokenizer and model
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir="cache/",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir="cache/",
        use_fast=False,
    )
    # CHANGE: Load model in 8-bit
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=True,
        device_map="auto",
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir="cache/",
    )

    # CHANGE: Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    # CHANGE: prepare int-8 model for training
    model = prepare_model_for_int8_training(model)

    # CHANGE: add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    special_tokens_dict = {"additional_special_tokens": ["<sep>", "<ent>", "</ent>"]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    set_seed(training_args.seed)

    # Load dataset
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir="cache/",
    )

    # Prefix is needed for T5 models only
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Tokenize inputs and targets to preprocess data
    # Set length and padding parameters
    def preprocess_function(examples):
        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])
            # When extractions are empty, learn to output "None"
            elif examples[text_column][i] and not examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append("None")

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs,
            max_length=data_args.max_source_length,
            padding=padding,
            truncation=True,
        )

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=padding,
            truncation=True,
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # column_names = []
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
    text_column, summary_column = column_names
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = raw_datasets["validation"]
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        predict_dataset = raw_datasets["test"]
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Define data collator
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Reset generation length and number of beams if needed
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams
        if data_args.num_beams is not None
        else training_args.generation_num_beams
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if training_args.predict_with_generate
        else None,
    )
    # print(trainer.model)

    if training_args.do_train:
        print("Started Training")
        get_gpu_memory_usage()
        train_result = trainer.train()
        get_gpu_memory_usage()
        print("Train Function Completed")
        trainer.save_model()  # Saves the tokenizer too for easy upload
        get_gpu_memory_usage()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        get_gpu_memory_usage()
    print("Training Completed")

    results = {}
    if training_args.do_eval:
        print("*** Evaluate ***")
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_ds_name, eval_ds in eval_dataset.items():
                dataset_metrics = trainer.evaluate(
                    eval_dataset=eval_ds, metric_key_prefix=f"eval_{eval_ds_name}"
                )
                metrics.update(dataset_metrics)
        else:
            metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = len(eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        print("*** Predict ***")
        # TODO: Figure out Deepspeed loading/inference later
        # state_dict = get_fp32_state_dict_from_zero_checkpoint(model_args.checkpoint_dir)
        # trainer.model.load_state_dict(state_dict)

        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        max_predict_samples = len(predict_dataset)
        metrics["predict_samples"] = len(predict_dataset)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = predict_results.predictions
                predictions = np.where(
                    predictions != -100, predictions, tokenizer.pad_token_id
                )
                predictions = tokenizer.batch_decode(
                    predictions,
                    # skip_special_tokens=True, # Don't skip special tokens while dumping
                    clean_up_tokenization_spaces=True,
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(
                    training_args.output_dir, "generated_predictions.txt"
                )
                input_instances = raw_datasets["test"][text_column]
                final_outputs = [
                    "{}|||{}".format(a, b) for a, b in zip(input_instances, predictions)
                ]
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(final_outputs))
