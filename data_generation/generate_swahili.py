import json

import numpy as np


data_type = [
    ('word', 'U100'),
    ('category', 'U100'),
    ('noun_class', int),
    ('semantic_class', 'U100'),
    ('possible_connections', 'U100'),
]

vocab = np.genfromtxt('vocabulary/swahili.csv', delimiter=',', names=True, dtype=data_type)

all_semantic_classes = set(vocab['semantic_class']) - {'_'}

preposition = {
    1: 'wa',  2: 'wa',
    3: 'wa',  4: 'ya',
    5: 'la',  6: 'ya',
    7: 'cha', 8: 'vya',
    9: 'ya',  10: 'za',
    11: 'wa'
}

proximal = {
    1: 'huyu',  2: 'hawa',
    3: 'huu',   4: 'hii',
    5: 'hili',  6: 'haya',
    7: 'hiki',  8: 'hivi',
    9: 'hii',   10: 'hizi',
    11: 'huu'
}

medial = {
    1: 'huyo',  2: 'hao',
    3: 'huo',   4: 'hiyo',
    5: 'hilo',  6: 'hayo',
    7: 'hicho', 8: 'hivyo',
    9: 'hiyo',  10: 'hizo',
    11: 'huo'
}

distal = {
    1: 'yule',  2: 'wale',
    3: 'ule',   4: 'ile',
    5: 'lile',  6: 'yale',
    7: 'kile',  8: 'vile',
    9: 'ile',   10: 'zile',
    11: 'ule'
}

subject = {
    1: 'a',  2: 'wa',
    3: 'u',  4: 'i',
    5: 'li', 6: 'ya',
    7: 'ki', 8: 'vi',
    9: 'i',  10: 'zi',
    11: 'u'
}

tense_non_rel = ['li', 'na', 'ta']

tense_rel = ['li', 'na', 'taka']

rel = {
    1: 'ye',  2: 'o',
    3: 'o',   4: 'yo',
    5: 'lo',  6: 'yo',
    7: 'cho', 8: 'vyo',
    9: 'yo',  10: 'zo',
    11: 'o'
}


def get_all(f):
    return np.array(list(filter(f, vocab)), dtype=vocab.dtype)


def get_adjectives(semantic_class):
    return get_all(lambda x: x['category'] == 'a' and semantic_class in x['possible_connections'])


def choice(words):
    return np.random.choice(words)


def choose_adjective(semantic_class):
    return choice(get_adjectives(semantic_class))['word']


def choose_noun(f = lambda x: x['semantic_class'] in all_semantic_classes):
    result = choice(get_all(lambda x: x['category'] == 'n' and f(x)))
    return result, concord_class(result)


def choose_determiner(cls):
    return np.random.choice([proximal[cls], medial[cls], distal[cls]])


def choose_verb(semantic_class):
    return choice(get_all(lambda x: x['category'] == 'v' and semantic_class in x['possible_connections']))['word']


def form(adjective, cls):
    a_prefix = {
        1: 'mw', 2: 'wa',
        3: 'mw', 4: 'mi',
        5: '',   6: 'ma',
        7: 'ki', 8: 'vi',
        9: '',   10: '',
        11: 'mw'
    }
    other_vowel_prefix = {
        1: 'mw', 2: 'w',
        3: 'mw', 4: 'my',
        5: 'j',  6: 'm',
        7: 'ch', 8: 'vy',
        9: 'ny', 10: 'ny',
        11: 'mw'
    }
    consonant_prefix = {
        1: 'm',  2: 'wa',
        3: 'm',  4: 'mi',
        5: '',   6: 'ma',
        7: 'ki', 8: 'vi',
        11: 'm'
    }
    # exceptions
    if adjective == 'pya':
        if cls == 5:
            return 'jipya'
        if cls in {9, 10}:
            return 'mpya'
    if adjective == 'ema' and cls == 5:
        return 'njema'
    # regular
    if adjective[0] == 'a':
        return a_prefix[cls] + adjective
    if adjective[0] in 'eiou':
        return other_vowel_prefix[cls] + adjective
    if cls in {9, 10}:
        if adjective[0] in {'b', 'v'}:
            return 'm' + adjective
        if adjective[0] in {'l', 'r'}:
            return 'nd' + adjective[1:]
        if adjective[0] in {'d', 'j', 'g', 'z'}:
            return 'n' + adjective
        return adjective
    return consonant_prefix[cls] + adjective


def relative(verb, cls):
    return subject[cls] + np.random.choice(tense_rel) + rel[cls] + verb


def non_relative(verb, tense, cls):
    return subject[cls] + tense + verb


def animate(noun):
    return noun['semantic_class'] in {'human', 'animal'}


def concord_class(noun):
    if animate(noun):
        if noun['noun_class'] % 2 == 1:
            return 1
        return 2
    return noun['noun_class']


def random_possessor(N, cls_N, f):
    while True:
        Poss, cls_Poss = choose_noun()
        if animate(N) and N['noun_class'] in {9, 10} and (f(cls_Poss) == f(1) or f(cls_Poss) == f(2)):
            continue
        if f(cls_N) != f(cls_Poss):
            break
    return Poss, cls_Poss


def sample_N_of_Poss_ni_A():
    N, cls_N = choose_noun()
    A = choose_adjective(N['semantic_class'])
    Poss, cls_Poss = random_possessor(N, cls_N, lambda cls: form(A, cls))
    condition = f"{N['word'].capitalize()} {preposition[cls_N]} {Poss['word']} ni"
    grammatical_adj = form(A, cls_N)
    ungrammatical_adj = form(A, cls_Poss)
    return condition, grammatical_adj, ungrammatical_adj


def sample_N_of_Poss_D_ni_A():
    N, cls_N = choose_noun()
    A = choose_adjective(N['semantic_class'])
    Poss, cls_Poss = random_possessor(N, cls_N, lambda cls: form(A, cls))
    D = choose_determiner(cls_Poss)
    condition = f"{N['word'].capitalize()} {preposition[cls_N]} {Poss['word']} {D} ni"
    grammatical_adj = form(A, cls_N)
    ungrammatical_adj = form(A, cls_Poss)
    return condition, grammatical_adj, ungrammatical_adj


def sample_N_of_Poss_D_AP_ni_AN():
    N, cls_N = choose_noun()
    AN = choose_adjective(N['semantic_class'])
    Poss, cls_Poss = random_possessor(N, cls_N, lambda cls: form(AN, cls))
    D = choose_determiner(cls_Poss)
    AP = choose_adjective(Poss['semantic_class'])
    condition = f"{N['word'].capitalize()} {preposition[cls_N]} {Poss['word']} {D} {form(AP, cls_Poss)} ni"
    grammatical_adj = form(AN, cls_N)
    ungrammatical_adj = form(AN, cls_Poss)
    return condition, grammatical_adj, ungrammatical_adj


def sample_N_of_Poss_D_AP_V_ni_AN():
    N, cls_N = choose_noun()
    AN = choose_adjective(N['semantic_class'])
    Poss, cls_Poss = random_possessor(N, cls_N, lambda cls: form(AN, cls))
    D = choose_determiner(cls_Poss)
    AP = choose_adjective(Poss['semantic_class'])
    V = choose_verb(Poss['semantic_class'])
    condition = f"{N['word'].capitalize()} {preposition[cls_N]} {Poss['word']} {D} {form(AP, cls_Poss)} {relative(V, cls_Poss)} ni"
    grammatical_adj = form(AN, cls_N)
    ungrammatical_adj = form(AN, cls_Poss)
    return condition, grammatical_adj, ungrammatical_adj


def sample_N_of_Poss_V():
    N, cls_N = choose_noun()
    V = choose_verb(N['semantic_class'])
    T = np.random.choice(tense_non_rel)
    Poss, cls_Poss = random_possessor(N, cls_N, lambda cls: non_relative(V, T, cls))
    condition = f"{N['word'].capitalize()} {preposition[cls_N]} {Poss['word']}"
    grammatical_v = non_relative(V, T, cls_N)
    ungrammatical_v = non_relative(V, T, cls_Poss)
    return condition, grammatical_v, ungrammatical_v


def sample_N_of_Poss_D_V():
    N, cls_N = choose_noun()
    V = choose_verb(N['semantic_class'])
    T = np.random.choice(tense_non_rel)
    Poss, cls_Poss = random_possessor(N, cls_N, lambda cls: non_relative(V, T, cls))
    D = choose_determiner(cls_Poss)
    condition = f"{N['word'].capitalize()} {preposition[cls_N]} {Poss['word']} {D}"
    grammatical_v = non_relative(V, T, cls_N)
    ungrammatical_v = non_relative(V, T, cls_Poss)
    return condition, grammatical_v, ungrammatical_v


def sample_N_of_Poss_D_A_V():
    N, cls_N = choose_noun()
    V = choose_verb(N['semantic_class'])
    T = np.random.choice(tense_non_rel)
    Poss, cls_Poss = random_possessor(N, cls_N, lambda cls: non_relative(V, T, cls))
    D = choose_determiner(cls_Poss)
    AP = choose_adjective(Poss['semantic_class'])
    condition = f"{N['word'].capitalize()} {preposition[cls_N]} {Poss['word']} {D} {form(AP, cls_Poss)}"
    grammatical_v = non_relative(V, T, cls_N)
    ungrammatical_v = non_relative(V, T, cls_Poss)
    return condition, grammatical_v, ungrammatical_v


def sample_N_of_Poss_D_A_V1_V2():
    N, cls_N = choose_noun()
    V2 = choose_verb(N['semantic_class'])
    T = np.random.choice(tense_non_rel)
    Poss, cls_Poss = random_possessor(N, cls_N, lambda cls: non_relative(V2, T, cls))
    D = choose_determiner(cls_Poss)
    AP = choose_adjective(Poss['semantic_class'])
    V1 = choose_verb(Poss['semantic_class'])
    condition = f"{N['word'].capitalize()} {preposition[cls_N]} {Poss['word']} {D} {form(AP, cls_Poss)} {relative(V1, cls_Poss)}"
    grammatical_v = non_relative(V2, T, cls_N)
    ungrammatical_v = non_relative(V2, T, cls_Poss)
    return condition, grammatical_v, ungrammatical_v


def generate_suite(length, sample_function):
    minimal_pairs = set()
    for i in range(length):
        while True:
            minimal_pair = sample_function()
            if minimal_pair not in minimal_pairs:
                minimal_pairs.add(minimal_pair)
                break
    suite = []
    for condition, grammatical, ungrammatical in minimal_pairs:
        suite.append([2*[condition], [f"{grammatical}.", f"{ungrammatical}."]])
    with open(f"../suites/swahili-{sample_function.__name__[7:]}.json", 'w') as f:
        json.dump(suite, f)