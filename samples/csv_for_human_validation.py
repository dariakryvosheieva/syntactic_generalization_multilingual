import os
import csv
import json
import random

import pandas as pd
import numpy as np


def generate_csv(language, control):
    filename = f'{language}-samples.csv'
    with open(filename, 'w', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['suite', 'grammatical', 'ungrammatical'])
        for suite_name in os.listdir('../suites'):
            if language not in suite_name:
                continue
            with open(f'../suites/{suite_name}', 'r', encoding='utf-8') as json_file:
                suite = json.load(json_file)
                for minimal_pair in random.sample(suite, 5):
                    condition = minimal_pair[0][0]
                    grammatical_target, ungrammatical_target = minimal_pair[1]
                    csv_writer.writerow([suite_name, f'{condition} {grammatical_target}', f'{condition} {ungrammatical_target}'])
        for control_real, control_artificial in control:
            csv_writer.writerow(['control', control_real, control_artificial])
    df = pd.read_csv(filename)
    return df.reindex(np.random.permutation(df.index)).to_csv(filename)


if __name__ == "__main__":
    generate_csv(
        'basque',
        [
            ('Urak ibar osoa bete zuen.', 'Urak ibar osoa bete neuz.'),
            ('Goizeko trenean etorri ziren.', 'Goizeko trenean etorri neriz.')
        ]
    )

    generate_csv(
        'swahili',
        [
            ('Vyama vya kisiasa ni vingi.', 'Vyama vya kisiasa ni bingi.'),
            ('Lugha ya Kiswahili ni tamu.', 'Lugha ya Kiswahili ni datamu.')
        ]
    )

    generate_csv(
        'hindi',
        [
            ('मैं एक फ़िल्म देखना चाहती हूँ।', 'मैं एक फ़िल्म देखना चाहलू हूँ।'),
            ('हम अपना घर साफ़ करते हैं।', 'हम अपना घर साफ़ करसो हैं।')
        ]
    )