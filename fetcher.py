# plan: downloads plain .txt texts of articles specified in article_ids
# TODO: save under specific path, differentiate between food and disease articles
import subprocess

article_ids = [
    'PMC5363789',
    'PMC6137369']

for article_id in article_ids:
    cmd = ['aws', 's3', 'cp', f's3://pmc-oa-opendata/oa_noncomm/txt/all/{article_id}.txt',
                    f'./{article_id}.txt', '--no-sign-request']

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print('Trying oa_comm directory...')
        cmd[3] = f's3://pmc-oa-opendata/oa_comm/txt/all/{article_id}.txt'
        subprocess.check_call(cmd)
