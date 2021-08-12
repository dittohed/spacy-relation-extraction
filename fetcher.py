# this script downloads plain .txt texts of specified articles to corresponding directories

import subprocess

# ./articles/<folder>: [<article ids>]
article_ids = {
    'diseases': ['PMC5363789'],
    'food': ['PMC6137369', 'PMC5561571', 'PMC6131755', 'PMC6542079', 'PMC6356196']
    }

# dictionary to a list of pairs e.g. ('food', 'PMC6137369'), ('food', 'PMC5561571') etc.
article_ids = [(key, val) for key, values in article_ids.items() for val in values]

for dir, article_id in article_ids:
    cmd = ['aws', 's3', 'cp', f's3://pmc-oa-opendata/oa_comm/txt/all/{article_id}.txt',
                    f'./articles/{dir}/{article_id}.txt', '--no-sign-request']

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        try:
            print('Trying oa_noncomm directory instead...')
            cmd[3] = f's3://pmc-oa-opendata/oa_noncomm/txt/all/{article_id}.txt'
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            print(f'{article_id} was not found, skipping to the next one...')
