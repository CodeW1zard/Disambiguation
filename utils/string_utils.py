import nltk

punct = set(u''':!),.:;?.]}¢'"、。〉》」』〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､￠
々‖•·ˇˉ―′’”([{£¥'"‵〈《「『〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘_…/''')

stemmer = nltk.stem.PorterStemmer()


def stem(word):
    return stemmer.stem(word)


def clean_sentence(text, stemming=False, join_str=' '):
    for token in punct:
        text = text.replace(token, "")
    text = text.strip()
    words = text.split()
    if stemming:
        stemmed_words = []
        for w in words:
            stemmed_words.append(stem(w))
        words = stemmed_words
    return join_str.join(words).lower()


def clean_name(name):
    if name is None:
        return ""
    x = [k.strip() for k in name.lower().strip().replace(".", " ").replace("-", " ").split()]
    return "_".join(x)