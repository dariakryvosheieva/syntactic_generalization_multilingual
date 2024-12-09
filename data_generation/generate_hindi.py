import json

import numpy as np


data_type = [
    ('word', 'U100'),
    ('category', 'U100'),
    ('semantic_class', 'U100'),
    ('gender', 'U100'),
    ('nom_pl', 'U100'),
    ('obl_sg', 'U100'),
    ('obl_pl', 'U100'),
]

vocab = np.genfromtxt('vocabulary/hindi.csv', delimiter=',', names=True, dtype=data_type)


def get_all(f):
    return np.array(list(filter(f, vocab)), dtype=vocab.dtype)


def get_nouns(include, avoid=''):
    return get_all(lambda x: x['category'] == 'n' and x['semantic_class'] in include and x['semantic_class'] not in avoid)


def choice(words):
    return np.random.choice(words)


def choose_subject():
    return choice(get_nouns('person animal'))


def choose_object(verb, avoid=''):
    return choice(get_nouns(verb['semantic_class'], avoid))


def choose_number():
    return np.random.choice(['sg', 'pl'])


def opposite(number):
    if number == 'sg':
        return 'pl'
    return 'sg'


def choose_tense():
    return np.random.choice(['pst', 'prs'])


def form(verb, aspect, gender, number, tense):
    stem = (
        verb[0] +
        ('त' if aspect == 'hab' else 'य' if aspect == 'pfv' and verb[0][-1] == 'ा' else '') +
        ('ी' if gender == 'f' else 'ा' if number == 'sg' else 'े')
    )
    if tense == 'prs':
        aux = 'है' + ('ं' if number == 'pl' else '')
    else:
        aux = 'थ'
        if gender == 'f':
            aux += 'ी' + ('ं' if number == 'pl' else '')
        else:
            aux += ('ा' if number == 'sg' else 'े')
    return f'{stem} {aux}'


def possessive_pronoun(gender):
    return np.random.choice(['मेर', 'हमार', 'तुम्हार', 'आपक', 'उसक', 'उनक', 'इसक', 'इनक']) + ('े' if gender == 'm' else 'ी')


def genitive(gender, number):
    if gender == 'm':
        if number == 'sg':
            return 'का'
        return 'के'
    return 'की'


all_verbs = get_all(lambda x: x['category'] == 'v')


def sample_S_O_V():
    V = choice(all_verbs)
    while True:
        S = choose_subject()
        O = choose_object(V)
        if S != O:
            break
    S_number, O_number = choose_number(), choose_number()
    tense = choose_tense()
    if O['semantic_class'] in {'person', 'animal'}:
        use_ko = True
    else:
        use_ko = np.random.choice([True, False])
    if use_ko:
        condition = f"{S[0 if S_number == 'sg' else 'nom_pl']} {O[f'obl_{O_number}']} को"
    else:
        condition = f"{S[0 if S_number == 'sg' else 'nom_pl']} {O[0 if O_number == 'sg' else 'nom_pl']}"
    grammatical_verb = form(V, 'hab', S['gender'], S_number, tense)
    ungrammatical_verb = form(V, 'pfv', S['gender'], S_number, tense)
    return condition, grammatical_verb, ungrammatical_verb


def sample_S_ne_O_V():
    V = choice(all_verbs)
    while True:
        S = choose_subject()
        O = choose_object(V)
        if S != O:
        break
    S_number, O_number = choose_number(), choose_number()
    tense = choose_tense()
    if O['semantic_class'] in {'person', 'animal'}:
        use_ko = True
    else:
        use_ko = np.random.choice([True, False])
    if use_ko:
        condition = f"{S[f'obl_{S_number}']} ने {O[f'obl_{O_number}']} को"
        grammatical_verb = form(V, 'pfv', 'm', 'sg', tense)
        ungrammatical_verb = form(V, 'hab', 'm', 'sg', tense)
    else:
        condition = f"{S[f'obl_{S_number}']} ने {O[0 if O_number == 'sg' else 'nom_pl']}"
        grammatical_verb = form(V, 'pfv', O['gender'], O_number, tense)
        ungrammatical_verb = form(V, 'hab', O['gender'], O_number, tense)
    return condition, grammatical_verb, ungrammatical_verb


def sample_S_PossPRN_O_V():
    V = choice(all_verbs)
    while True:
        S = choose_subject()
        O = choose_object(V)
        if S != O:
            break
    S_number, O_number = choose_number(), choose_number()
    tense = choose_tense()
    if O['semantic_class'] in {'person', 'animal'}:
        use_ko = True
    else:
        use_ko = np.random.choice([True, False])
    if use_ko:
        condition = f"{S[0 if S_number == 'sg' else 'nom_pl']} {possessive_pronoun(O['gender'])} {O[f'obl_{O_number}']} को"
    else:
        condition = f"{S[0 if S_number == 'sg' else 'nom_pl']} {possessive_pronoun(O['gender'])} {O[0 if O_number == 'sg' else 'nom_pl']}"
    grammatical_verb = form(V, 'hab', S['gender'], S_number, tense)
    ungrammatical_verb = form(V, 'pfv', S['gender'], S_number, tense)
    return condition, grammatical_verb, ungrammatical_verb


def sample_S_ne_PossPRN_O_V():
    V = choice(all_verbs)
    while True:
        S = choose_subject()
        O = choose_object(V)
        if S != O:
            break
    S_number, O_number = choose_number(), choose_number()
    tense = choose_tense()
    if O['semantic_class'] in {'person', 'animal'}:
        use_ko = True
    else:
        use_ko = np.random.choice([True, False])
    if use_ko:
        condition = f"{S[f'obl_{S_number}']} ने {possessive_pronoun(O['gender'])} {O[f'obl_{O_number}']} को"
        grammatical_verb = form(V, 'pfv', 'm', 'sg', tense)
        ungrammatical_verb = form(V, 'hab', 'm', 'sg', tense)
    else:
        condition = f"{S[f'obl_{S_number}']} ने {possessive_pronoun(O['gender'])} {O[0 if O_number == 'sg' else 'nom_pl']}"
        grammatical_verb = form(V, 'pfv', O['gender'], O_number, tense)
        ungrammatical_verb = form(V, 'hab', O['gender'], O_number, tense)
    return condition, grammatical_verb, ungrammatical_verb


def sample_S_PossPRN_PossN_O_V():
    V = choice(all_verbs)
    while True:
        S = choose_subject()
        PossN = choose_subject()
        O = choose_object(V, 'person')
        if S != PossN != O:
            break
    S_number, PossN_number, O_number = choose_number(), choose_number(), choose_number()
    tense = choose_tense()
    if O['semantic_class'] == 'animal':
        use_ko = True
    else:
        use_ko = np.random.choice([True, False])
    if use_ko:
        condition = f"{S[0 if S_number == 'sg' else 'nom_pl']} {possessive_pronoun(PossN['gender'])} {PossN[f'obl_{PossN_number}']} {genitive(O['gender'], O_number)} {O[f'obl_{O_number}']} को"
    else:
        condition = f"{S[0 if S_number == 'sg' else 'nom_pl']} {possessive_pronoun(PossN['gender'])} {PossN[f'obl_{PossN_number}']} {genitive(O['gender'], O_number)} {O[0 if O_number == 'sg' else 'nom_pl']}"
    grammatical_verb = form(V, 'hab', S['gender'], S_number, tense)
    ungrammatical_verb = form(V, 'pfv', S['gender'], S_number, tense)
    return condition, grammatical_verb, ungrammatical_verb


def sample_S_ne_PossPRN_PossN_O_V():
    V = choice(all_verbs)
    while True:
        S = choose_subject()
        PossN = choose_subject()
        O = choose_object(V, 'person')
        if S != PossN != O:
            break
    S_number, PossN_number, O_number = choose_number(), choose_number(), choose_number()
    tense = choose_tense()
    if O['semantic_class'] == 'animal':
        use_ko = True
    else:
        use_ko = np.random.choice([True, False])
    if use_ko:
        condition = f"{S[f'obl_{S_number}']} ने {possessive_pronoun(PossN['gender'])} {PossN[f'obl_{PossN_number}']} {genitive(O['gender'], O_number)} {O[f'obl_{O_number}']} को"
        grammatical_verb = form(V, 'pfv', 'm', 'sg', tense)
        ungrammatical_verb = form(V, 'hab', 'm', 'sg', tense)
    else:
        condition = f"{S[f'obl_{S_number}']} ने {possessive_pronoun(PossN['gender'])} {PossN[f'obl_{PossN_number}']} {genitive(O['gender'], O_number)} {O[0 if O_number == 'sg' else 'nom_pl']}"
        grammatical_verb = form(V, 'pfv', O['gender'], O_number, tense)
        ungrammatical_verb = form(V, 'hab', O['gender'], O_number, tense)
    return condition, grammatical_verb, ungrammatical_verb


def generate_suite(length, sample_function):
    minimal_pairs = set()
    for i in range(length):
        while True:
            minimal_pair = sample_function()
            if minimal_pair not in minimal_pairs:
                minimal_pairs.add(minimal_pair)
                break
    suite = []
    for condition, grammatical_verb, ungrammatical_verb in minimal_pairs:
        suite.append([2*[condition], [f"{grammatical_verb}।", f"{ungrammatical_verb}।"]])
    with open(f'../suites/hindi-{sample_function.__name__[7:]}.json', 'w', encoding='utf-8') as f:
        json.dump(suite, f, ensure_ascii=False)


if __name__ == "__main__":
    generate_suite(1000, sample_S_O_V)
    generate_suite(1000, sample_S_PossPRN_O_V)
    generate_suite(1000, sample_S_PossPRN_PossN_O_V)

    generate_suite(1000, sample_S_ne_O_V)
    generate_suite(1000, sample_S_ne_PossPRN_O_V)
    generate_suite(1000, sample_S_ne_PossPRN_PossN_O_V)