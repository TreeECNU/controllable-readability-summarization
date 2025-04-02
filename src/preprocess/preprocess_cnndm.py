# import json
# from readability import Readability
# from datasets import load_dataset

# # Download the CNNDM data (e.g., from https://huggingface.co/datasets/cnn_dailymail)
# dataset = ''

# def open_txt_file(file):
#     entities = []

#     for line in open(file).readlines():
#         entities.append(line)

#     return entities


# def open_file(file):
#     entities = []

#     for line in open(file).readlines():
#         entities.append(json.loads(line))

#     return entities

# def save_file(data, file):
#     file_writer = open(file, 'w')

#     for line in data:
#         file_writer.write(json.dumps(line) + "\n")

# def get_flesch_kincaid(text):
#     r = Readability(text)
#     fk = r.flesch_kincaid()
#     return fk.score


# def get_flesch(text):
#     r = Readability(text)
#     f = r.flesch()
#     return f.score


# def get_dale_chall(text):
#     r = Readability(text)
#     dc = r.dale_chall()
#     return dc.score


# def get_ari(text):
#     r = Readability(text)
#     ari = r.ari()
#     return ari.score


# def get_coleman_liau(text):
#     r = Readability(text)
#     cl = r.coleman_liau()
#     return cl.score


# def get_gunning_fog(text):
#     r = Readability(text)
#     gf = r.gunning_fog()
#     return gf.score


# def get_smog(text):
#     r = Readability(text)
#     s = r.smog()
#     return s.score


# def get_spache(text):
#     r = Readability(text)
#     s = r.spache()
#     return s.score

# def get_linsear_write(text):
#     r = Readability(text)
#     lw = r.linsear_write()
#     return lw.score


# def compute_metrics(text):
#     metrics = {}
#     flesch = get_flesch(text)
#     metrics['flesch'] = round(flesch, 4)

#     dale_chall = get_dale_chall(text)
#     metrics['dale_chall'] = round(dale_chall, 4)

#     coleman_liau = get_coleman_liau(text)
#     metrics['coleman_liau'] = round(coleman_liau, 4)

#     gunning_fog = get_gunning_fog(text)
#     metrics['gunning_fog'] = round(gunning_fog, 4)

#     return metrics


# def process_data(split):
#     data = []
#     for idx, (dial, sum, id_) in enumerate(zip(dataset[split]['article'], dataset[split]['highlights'], dataset[split]['id'])):
#         entry = {}
#         entry['input'] = dial
#         metrics = compute_metrics(entry["input"])
#         entry['input_metrics'] = metrics

#         entry['summary'] = sum
#         entry['id'] = str(id_)
#         metrics = compute_metrics(entry["summary"].replace("\n", " "))
#         entry['summary_metrics'] = metrics
#         data.append(entry)

#         save_file(data, 'data/' + split + '.json')


# process_data('train')
# process_data('validation')
# process_data('test')


# import json
# from readability import Readability
# from datasets import load_dataset

# # Load the CNNDM dataset in JSONL format
# dataset = load_dataset('json', data_files={'train': 'data/train.jsonl', 'validation': 'data/validation.jsonl', 'test': 'data/test.jsonl'})

# def save_file(data, file):
#     file_writer = open(file, 'w')

#     for line in data:
#         file_writer.write(json.dumps(line) + "\n")

# def get_flesch_kincaid(text):
#     r = Readability(text)
#     fk = r.flesch_kincaid()
#     return fk.score

# def get_flesch(text):
#     r = Readability(text)
#     f = r.flesch()
#     return f.score

# def get_dale_chall(text):
#     r = Readability(text)
#     dc = r.dale_chall()
#     return dc.score

# def get_ari(text):
#     r = Readability(text)
#     ari = r.ari()
#     return ari.score

# def get_coleman_liau(text):
#     r = Readability(text)
#     cl = r.coleman_liau()
#     return cl.score

# def get_gunning_fog(text):
#     r = Readability(text)
#     gf = r.gunning_fog()
#     return gf.score

# def get_smog(text):
#     r = Readability(text)
#     s = r.smog()
#     return s.score

# def get_spache(text):
#     r = Readability(text)
#     s = r.spache()
#     return s.score

# def get_linsear_write(text):
#     r = Readability(text)
#     lw = r.linsear_write()
#     return lw.score

# def compute_metrics(text):
#     metrics = {}
#     flesch = get_flesch(text)
#     metrics['flesch'] = round(flesch, 4)

#     dale_chall = get_dale_chall(text)
#     metrics['dale_chall'] = round(dale_chall, 4)

#     coleman_liau = get_coleman_liau(text)
#     metrics['coleman_liau'] = round(coleman_liau, 4)

#     gunning_fog = get_gunning_fog(text)
#     metrics['gunning_fog'] = round(gunning_fog, 4)

#     return metrics

# def process_data(split):
#     data = []
#     for idx, entry in enumerate(dataset[split]):
#         dial = entry['article']
#         sum = entry['highlights']
#         id_ = entry['id']

#         entry = {}
#         entry['input'] = dial
#         metrics = compute_metrics(entry["input"])
#         entry['input_metrics'] = metrics

#         entry['summary'] = sum
#         entry['id'] = str(id_)
#         metrics = compute_metrics(entry["summary"].replace("\n", " "))
#         entry['summary_metrics'] = metrics
#         data.append(entry)

#     save_file(data, 'data/' + split + '.json')

# process_data('train')
# process_data('validation')
# process_data('test')

# import json
# from readability import Readability
# from datasets import load_dataset

# # Load the CNNDM dataset in JSONL format
# dataset = load_dataset('json', data_files={'train': 'data/train.jsonl', 'validation': 'data/validation.jsonl', 'test': 'data/test.jsonl'})

# def save_file(data, file):
#     with open(file, 'w') as file_writer:
#         for line in data:
#             file_writer.write(json.dumps(line) + "\n")

# def compute_metrics(text):
#     metrics = {}
#     word_count = len(text.split())

#     r = Readability(text)

#     # Use suitable readability scores for short texts and set to None if text is too short
#     metrics['ari'] = round(r.ari().score, 4) if word_count >= 100 else None
#     metrics['coleman_liau'] = round(r.coleman_liau().score, 4) if word_count >= 100 else None
#     metrics['gunning_fog'] = round(r.gunning_fog().score, 4) if word_count >= 100 else None
    
#     # Flesch and Dale-Chall require at least 100 words
#     metrics['flesch'] = round(r.flesch().score, 4) if word_count >= 100 else None
#     metrics['dale_chall'] = round(r.dale_chall().score, 4) if word_count >= 100 else None

#     return metrics

# def process_data(split):
#     data = []
#     for idx, entry in enumerate(dataset[split]):
#         dial = entry['article']
#         sum = entry['highlights']
#         id_ = entry['id']

#         processed_entry = {
#             'input': dial,
#             'input_metrics': compute_metrics(dial),
#             'summary': sum,
#             'id': str(id_)
#         }
        
#         # Ensure summary is a string and replace newlines before computing metrics
#         processed_summary = sum.replace("\n", " ") if isinstance(sum, str) else ""
#         processed_entry['summary_metrics'] = compute_metrics(processed_summary)

#         data.append(processed_entry)

#     save_file(data, f'data/{split}.json')

# process_data('train')
# process_data('validation')
# process_data('test')

import json
from datasets import load_dataset
import textstat
from tqdm import tqdm  # 导入 tqdm

# Load the CNNDM dataset in JSONL format
dataset = load_dataset('json', data_files={'train': 'data/train.jsonl', 'validation': 'data/validation.jsonl', 'test': 'data/test.jsonl'})

def save_file(data, file):
    with open(file, 'w') as file_writer:
        for line in data:
            file_writer.write(json.dumps(line) + "\n")

def compute_metrics(text):
    metrics = {}

    metrics['ari'] = round(textstat.automated_readability_index(text), 4)
    metrics['coleman_liau'] = round(textstat.coleman_liau_index(text), 4)
    metrics['gunning_fog'] = round(textstat.gunning_fog(text), 4)
    metrics['flesch'] = round(textstat.flesch_reading_ease(text), 4)
    metrics['dale_chall'] = round(textstat.dale_chall_readability_score(text), 4)

    return metrics

def process_data(split):
    data = []
    for idx, entry in tqdm(enumerate(dataset[split]), total=len(dataset[split]), desc=f'Processing {split}'):
        dial = entry['article']
        sum = entry['highlights']
        id_ = entry['id']

        processed_entry = {
            'input': dial,
            'input_metrics': compute_metrics(dial),
            'summary': sum,
            'id': str(id_)
        }
        
        # Ensure summary is a string and replace newlines before computing metrics
        processed_summary = sum.replace("\n", " ") if isinstance(sum, str) else ""
        processed_entry['summary_metrics'] = compute_metrics(processed_summary)

        data.append(processed_entry)

    save_file(data, f'data/{split}.json')

process_data('train')
process_data('validation')
process_data('test')