import json

import numpy as np


data_type = [
    ('word', 'U100'),
    ('category', 'U100'),
    ('subcategory', 'U100'),
    ('tense', 'U100'),
    ('possible_S', 'U100'),
    ('possible_DO', 'U100'),
    ('possible_IO', 'U100'),
]

vocab = np.genfromtxt('vocabulary/basque.csv', delimiter=',', names=True, dtype=data_type)


def get_all(f):
    return np.array(list(filter(f, vocab)), dtype=vocab.dtype)


def get_verbs(paradigm):
    return get_all(lambda x: x['category'] == 'v' and x['subcategory'] == paradigm)


def get_nouns(semantic_classes):
    return get_all(lambda x: x['subcategory'] in semantic_classes)


def get_auxiliary(paradigm, tense, possible_S, possible_DO='_', possible_IO='_'):
    for x in vocab:
        if (
            x['category'] == 'aux'
            and x['subcategory'] == paradigm
            and tense in x['tense'].split(' ')
            and x['possible_S'] == possible_S
            and x['possible_DO'] == possible_DO
            and x['possible_IO'] == possible_IO
        ):
            return x['word']


def choice(words):
    return np.random.choice(words)


def choose_noun(V, role):
    return choice(get_nouns(V['possible_' + role].split(' ')))['word']


def choose_number():
    return np.random.choice(['sg', 'pl'])


def form(noun, case, number):
    stem = noun[:]
    if stem[-1] == 'a':
        stem = stem[:-1]
    elif stem[-1] == 'r':
        stem = stem + 'r'
    if case == 'erg' and number == 'pl':
        return stem + 'ek'
    if (case == 'erg' and number == 'sg') or (case == 'abs' and number == 'pl'):
        return stem + 'ak'
    if case == 'abs' and number == 'sg':
        return stem + 'a'
    if case == 'dat' and number == 'sg':
        return stem + 'ari'
    if case == 'dat' and number == 'pl':
        return stem + 'ei'


def opposite(number):
    if number == 'sg':
        return 'pl'
    return 'sg'


all_nor_verbs = get_verbs('nor')
all_nor_nork_verbs = get_verbs('nor-nork')
all_nor_nori_nork_verbs = get_verbs('nor-nori-nork')
all_nor_nori_verbs = get_verbs('nor-nori')


def sample_S_V_AUX(parameter):
    V = choice(all_nor_verbs)
    S = choose_noun(V, 'S')
    S_number = choose_number()
    condition = f"{form(S, 'abs', S_number).capitalize()} {V['word']}"
    grammatical_aux = get_auxiliary('nor', V['tense'], S_number)
    ungrammatical_aux = get_auxiliary('nor', V['tense'], opposite(S_number))
    return condition, grammatical_aux, ungrammatical_aux


def sample_S_DO_V_AUX(parameter):
    V = choice(all_nor_nork_verbs)
    while True:
        S, DO = choose_noun(V, 'S'), choose_noun(V, 'DO')
        if S != DO:
            break
    S_number = choose_number()
    DO_number = opposite(S_number)
    condition = f"{form(S, 'erg', S_number).capitalize()} {form(DO, 'abs', DO_number)} {V['word']}"
    grammatical_aux = get_auxiliary('nor-nork', V['tense'], S_number, DO_number)
    if parameter == 'S':
        ungrammatical_aux = get_auxiliary('nor-nork', V['tense'], opposite(S_number), DO_number)
    elif parameter == 'DO':
        ungrammatical_aux = get_auxiliary('nor-nork', V['tense'], S_number, opposite(DO_number))
    return condition, grammatical_aux, ungrammatical_aux


def sample_S_IO_DO_V_AUX(parameter):
    V = choice(all_nor_nori_nork_verbs)
    while True:
        S, IO, DO = choose_noun(V, 'S'), choose_noun(V, 'IO'), choose_noun(V, 'DO')
        if S != IO != DO:
            break
    if parameter == 'S':
        S_number = choose_number()
        IO_number = DO_number = opposite(S_number)
    elif parameter == 'IO':
        IO_number = choose_number()
        S_number = DO_number = opposite(IO_number)
    elif parameter == 'DO':
        DO_number = choose_number()
        S_number = IO_number = opposite(DO_number)
    condition = f"{form(S, 'erg', S_number).capitalize()} {form(IO, 'dat', IO_number)} {form(DO, 'abs', DO_number)} {V['word']}"
    grammatical_aux = get_auxiliary('nor-nori-nork', V['tense'], S_number, DO_number, IO_number)
    if parameter == 'S':
        ungrammatical_aux = get_auxiliary('nor-nori-nork', V['tense'], opposite(S_number), DO_number, IO_number)
    elif parameter == 'DO':
        ungrammatical_aux = get_auxiliary('nor-nori-nork', V['tense'], S_number, opposite(DO_number), IO_number)
    elif parameter == 'IO':
        ungrammatical_aux = get_auxiliary('nor-nori-nork', V['tense'], S_number, DO_number, opposite(IO_number))
    return condition, grammatical_aux, ungrammatical_aux


def sample_IO_S_V_AUX(parameter):
    V = choice(all_nor_nori_verbs)
    while True:
        IO, S = choose_noun(V, 'IO'), choose_noun(V, 'S')
        if IO != S:
            break
    IO_number = choose_number()
    S_number = opposite(IO_number)
    condition = f"{form(IO, 'dat', IO_number).capitalize()} {form(S, 'abs', S_number)} {V['word']}"
    grammatical_aux = get_auxiliary('nor-nori', V['tense'], S_number, possible_IO=IO_number)
    if parameter == 'S':
        ungrammatical_aux = get_auxiliary('nor-nori', V['tense'], opposite(S_number), possible_IO=IO_number)
    elif parameter == 'IO':
        ungrammatical_aux = get_auxiliary('nor-nori', V['tense'], S_number, possible_IO=opposite(IO_number))
    return condition, grammatical_aux, ungrammatical_aux


def generate_suite(length, sample_function, parameter):
    minimal_pairs = set()
    for i in range(length):
        while True:
            minimal_pair = sample_function(parameter)
            if minimal_pair not in minimal_pairs:
                minimal_pairs.add(minimal_pair)
                break
    suite = []
    for condition, grammatical_aux, ungrammatical_aux in minimal_pairs:
        suite.append([2*[condition], [f"{grammatical_aux}.", f"{ungrammatical_aux}."]])
    with open(f'../suites/basque-{parameter}-{sample_function.__name__[7:]}.json', 'w') as f:
        json.dump(suite, f)


if __name__ == "__main__":
    generate_suite(1000, sample_S_V_AUX, 'S')
    generate_suite(1000, sample_S_DO_V_AUX, 'S')
    generate_suite(1000, sample_S_IO_DO_V_AUX, 'S')
    generate_suite(1000, sample_IO_S_V_AUX, 'S')

    generate_suite(1000, sample_S_DO_V_AUX, 'DO')
    generate_suite(1000, sample_S_IO_DO_V_AUX, 'DO')

    generate_suite(1000, sample_S_IO_DO_V_AUX, 'IO')
    generate_suite(1000, sample_IO_S_V_AUX, 'IO')