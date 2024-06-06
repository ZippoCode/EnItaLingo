import spacy

en_nlp = spacy.load("en_core_web_sm")
it_nlp = spacy.load("it_core_news_sm")


def get_tokens(sentence, nlp):
    doc = nlp(sentence)
    tokens = [token.text for token in doc]
    return tokens


def tokenize_sentences(english_sentences, italian_sentences):
    en_tokenized_sentences = []
    it_tokenized_sentences = []
    for en_sentence, it_sentence in zip(english_sentences, italian_sentences):
        en_tokens = get_tokens(en_sentence, en_nlp)
        it_tokens = get_tokens(it_sentence, en_nlp)

        en_tokenized_sentences.append(en_tokens)
        it_tokenized_sentences.append(it_tokens)

    return en_tokenized_sentences, it_tokenized_sentences
