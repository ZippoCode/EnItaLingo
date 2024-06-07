from transformers import AdamWeightDecay, DataCollatorForSeq2Seq, TFAutoModelForSeq2SeqLM, KerasMetricCallback, \
    PushToHubCallback

from src.utils import checkpoint, tokenizer, compute_metrics


def train_model(tokenized_dataset):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="tf")
    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

    model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tf_train_set = model.prepare_tf_dataset(
        tokenized_dataset["train"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_test_set = model.prepare_tf_dataset(
        tokenized_dataset["test"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_validation_set = model.prepare_tf_dataset(
        tokenized_dataset["val"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )
    model.compile(optimizer=optimizer)

    metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
    push_to_hub_callback = PushToHubCallback(
        output_dir="my_awesome_opus_books_model",
        tokenizer=tokenizer,
    )
    callbacks = [metric_callback, push_to_hub_callback]
    model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=callbacks)
