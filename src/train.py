from transformers import AdamWeightDecay, DataCollatorWithPadding, TFAutoModelForSeq2SeqLM

from src.utils import checkpoint, tokenizer


def train_model(tokenized_dataset):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

    model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tf_train_set = model.prepare_tf_dataset(
        tokenized_dataset["training"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_test_set = model.prepare_tf_dataset(
        tokenized_dataset["testing"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_validation_set = model.prepare_tf_dataset(
        tokenized_dataset["validation"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )
    model.compile(optimizer=optimizer)
    model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3)
    model.save_pretrained('checkpoints/enita_lingo')
