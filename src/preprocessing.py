import spacy
from rich.progress import Progress

en_nlp = spacy.load("en_core_web_sm")
it_nlp = spacy.load("it_core_news_sm")


def get_tokens(sentence, nlp):
    doc = nlp(sentence)
    tokens = []
    for token in doc:
        if not token.is_punct and not token.is_space and not token.is_stop and token.text != '(' and token.text != ')':
            tokens.append(token.lemma_.lower())
    return ' '.join(tokens)


def verify_alignment(source_sentences, target_sentences):
    assert len(source_sentences) == len(target_sentences), "The lists are not aligned!"
    for i, (source, target) in enumerate(zip(source_sentences, target_sentences)):
        if not source.strip() or not target.strip():
            print(f"Misalignment at line {i}: '{source}' -> '{target}'")
            return False
    return True


def get_tokenized_sentences(english_sentences, italian_sentences):
    en_tokenized_sentences = []
    it_tokenized_sentences = []
    is_aligned = verify_alignment(english_sentences, italian_sentences)
    print("Correct alignment:", is_aligned)
    with Progress() as progress:
        task = progress.add_task("[blue]Tokenizing sentences...", total=len(english_sentences))
        for en_sentence, it_sentence in zip(english_sentences, italian_sentences):
            en_tokens = get_tokens(en_sentence, en_nlp)
            it_tokens = get_tokens(it_sentence, en_nlp)

            en_tokenized_sentences.append(en_tokens)
            it_tokenized_sentences.append(it_tokens)
            progress.update(task, advance=1)

    return en_tokenized_sentences, it_tokenized_sentences


def tokenize_sentences(english_sentences, italian_sentences, test_size=0.2, val_size=0.2, random_state=None):
    en_train, en_test, it_train, it_test = train_test_split(english_sentences, italian_sentences, test_size=test_size,
                                                            random_state=random_state)
    en_test, en_val, it_test, it_val = train_test_split(en_test, it_test, test_size=val_size / (1 - test_size),
                                                        random_state=random_state)
    print("Tokenizing training sentence ...")
    en_train_token, it_train_token = get_tokenized_sentences(en_train, it_train)
    print("Tokenizing validation sentence ...")
    en_val_token, it_val_token = get_tokenized_sentences(en_val, it_val)
    print("Tokenizing test sentence ...")
    en_test_token, it_test_token = get_tokenized_sentences(en_test, it_val)
    return en_train_token, it_train_token, en_val_token, it_val_token, en_test_token, it_test_token
