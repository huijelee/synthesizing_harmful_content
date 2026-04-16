import os
import time
import json
import re
import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc
)
from openai import OpenAI
from googleapiclient import discovery
from private_keys import OPENAI_API_KEY, GoogleCloud_API_KEY
from dataset_classes import ELF22Dataset, QianDataset

# Moderation & Model Settings
MODERATION_PLATFORMS = {
    'PerspectiveAPI': 'get_perspective_api_score',
    'OpenAI': 'get_openai_moderation_score',
    'LlamaGuard-1': 'get_llamaguard_moderation_score',
    'LlamaGuard-2': 'get_llamaguard_moderation_score'
}

LLAMAGUARD1_ID = "meta-llama/LlamaGuard-7b"
LLAMAGUARD2_ID = "meta-llama/Meta-Llama-Guard-2-8B"
MODEL_CACHE_DIR = "cache/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
TROLL_THRESHOLD = 0.2

# API Call Settings
MAX_RETRIES = 3
SLEEP_SUCCESS = 1.0
SLEEP_FAIL = 10
DEALING_ERROR = 0

OPENAI_KEY = OPENAI_API_KEY[0]
GOOGLE_KEY = GoogleCloud_API_KEY
EVAL_DATASETS = ['qian_gab', 'qian_reddit', 'elf22', 'elf-hp', 'conan', 'mtconan', 'cadd', 'covid_hate', 'ours']
EVAL_PLATFORMS = ['OpenAI', 'LlamaGuard-1', 'LlamaGuard-2','PerspectiveAPI']


class BaseDataset:
    def __init__(self, **kwargs):
        self.name = "Base"

    def get_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError("This method should be implemented by subclasses.")


class ConanDataset(BaseDataset):
    def __init__(self, file_path, name, **kwargs):
        self.name = name
        self.file_path = file_path
    
    def get_dataframe(self) -> pd.DataFrame:
        print(f"Loading '{self.name}' dataset from {self.file_path}...")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        filtered_list = [item for item in data['conan'] if item.get("cn_id", "").startswith("EN")]
        df = pd.DataFrame(filtered_list)

        texts = df['hateSpeech'].dropna().astype(str)

        return pd.DataFrame({'text': texts, 'label': 1})


class MTConanDataset(BaseDataset):
    def __init__(self, file_path, name, **kwargs):
        self.name = name
        self.file_path = file_path
    
    def get_dataframe(self) -> pd.DataFrame:
        print(f"Loading '{self.name}' dataset from {self.file_path}...")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame.from_dict(data, orient='index')

        texts = df['HATE_SPEECH'].dropna().astype(str)

        return pd.DataFrame({'text': texts, 'label': 1})


class SyntheticDataset(BaseDataset):
    def __init__(self, file_path, name, **kwargs):
        self.name = name
        self.file_path = file_path

    def get_dataframe(self) -> pd.DataFrame:
        print(f"Loading '{self.name}' dataset from {self.file_path}...")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        text_subreddit = df['original_thread'].apply(lambda x: x['post']['subreddit'])
        text_title = df['original_thread'].apply(lambda x: x['post']['title'])
        text_post = df['original_thread'].apply(lambda x: x['post']['selftext'])
        text_troll = df['troll_comment'].apply(lambda x: x['body'])
        texts = 'Context: r/' + text_subreddit + '\nTitle: ' + text_title + '\nPost: ' + text_post + '\nComment: ' + text_troll
        
        return pd.DataFrame({'text': texts.dropna(), 'label': 1})


class ELFHPDataset(BaseDataset):
    def __init__(self, name="elf-hp", **kwargs):
        self.name = name

    def get_dataframe(self) -> pd.DataFrame:
        print(f"Loading '{self.name}' dataset from Hugging Face Hub...")
        
        dataset = load_dataset("huijelee/ELF-HP", split='train')
        df = dataset.to_pandas()

        # format: 'Context: r/... \nTitle: ... \nPost: ... \nComment: ...'
        texts = (
            'Context: r/' + df['Subreddit'].astype(str) +
            '\nTitle: ' + df['Title'].astype(str) +
            '\nPost: ' + df['Post'].astype(str) +
            '\nComment: ' + df['Comment'].astype(str)
        )
        
        # (0: non-troll, 1: troll).
        labels = df['TrollingStrategy'].apply(lambda x: 0 if x == 0 else 1)
        return pd.DataFrame({'text': texts, 'label': labels}).dropna(subset=['text'])


class CADDDataset(BaseDataset):
    def __init__(self, file_path, name, **kwargs):
        self.name = name
        self.file_path = file_path

    def get_dataframe(self) -> pd.DataFrame:
        print(f"Loading '{self.name}' dataset from {self.file_path}...")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        filtered_data = {k: v for k, v in data.items() if v.get("L2") == "1"}
        df = pd.DataFrame(list(filtered_data.values()))

        text_title = df['title']
        text_body = df['body']
        text_troll = df['comment']
        texts = 'Title: ' + text_title + '\nPost: ' + text_body + '\nComment: ' + text_troll
        
        return pd.DataFrame({'text': texts.dropna(), 'label': 1})


class CovidDataset(BaseDataset):
    def __init__(self, file_path, name, **kwargs):
        self.name = name
        self.file_path = file_path

    def get_dataframe(self) -> pd.DataFrame:
        print(f"Loading '{self.name}' dataset from {self.file_path}...")
        df = pd.read_csv(self.file_path)
        self.df = df

        all_hateful_turns = []
        for _, row in self.df.iterrows():
            if row['Label'] in ['Hate', 'Non-Asian Aggression']:
                text = row['Text']
                all_hateful_turns.append({'text': text, 'label': 1})

        return pd.DataFrame(all_hateful_turns)


def load_llamaguard_model(model_id):
    print(f"Loading {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=MODEL_CACHE_DIR, torch_dtype=DTYPE, device_map=DEVICE
        )
        print(f"{model_id} loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load {model_id}. Error: {e}")
        return None, None


def get_perspective_api_score(text: str, **kwargs) -> float:
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=GOOGLE_KEY,
        discoveryServiceUrl="<commentanalyzer.googleapis.com>",
        static_discovery=False,
    )

    if not text:
        return 0

    analyze_request = {
        'comment': {'text': text},
        'requestedAttributes': {'TOXICITY': {}},
        'languages': ['en']
    }

    for _ in range(MAX_RETRIES):
        try:
            response = client.comments().analyze(body=analyze_request).execute()
            time.sleep(SLEEP_SUCCESS)
            return response['attributeScores']['TOXICITY']['summaryScore']['value'] # (float) from 0 to 1
        except Exception as err:
            print(f"TOXICITY FUNCTION: {err}")
            time.sleep(SLEEP_FAIL)

    return 0


def get_openai_moderation_score(text: str, **kwargs) -> dict:
    client = OpenAI(api_key=OPENAI_KEY)

    try:
        response = client.moderations.create(input=text)
        output = response.results[0]
        time.sleep(SLEEP_SUCCESS / 30)
        openai_mod_keys = ["sexual", "sexual_minors", "harassment", "harassment_threatening", "hate", "hate_threatening", 
                           "violence", "violence_graphic"]
        
        # Get all category scores except self-harm related ones
        scores = {k: float(getattr(output.category_scores, k)) for k in dir(output.category_scores) if k in openai_mod_keys}
        
        # Find category with highest score
        max_category, max_category_score = max(scores.items(), key=lambda x: x[1])
        
        return {
            'categories': output.flagged,      # True or False
            'category_scores': max_category_score  # (float) from 0 to 1
        }
    except Exception as err:
        print(f"OPENAI MODERATION FUNCTION: {err}")
        time.sleep(SLEEP_FAIL / 30)
        return {
            'categories': DEALING_ERROR,
            'category_scores': DEALING_ERROR,
        }


def get_llamaguard_moderation_score(text: str, platform: str, llama_models={}, **kwargs) -> dict:

    if llama_models:
        model, tokenizer = llama_models[platform]
    else:
        model, tokenizer = LLAMA_MODELS[platform]
    

    def moderate(chat):
        # chat_template: https://huggingface.co/docs/transformers/main/en/chat_templating
        input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                pad_token_id=0,
                output_scores=True,
                return_dict_in_generate=True
            )
        return output, input_ids

    def check_top_predicted_tokens(first_token_logits):
        probs = F.softmax(first_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 5)
        top_tokens = tokenizer.convert_ids_to_tokens(top_indices)
        top_predictions = [(token, prob.item()) for token, prob in zip(top_tokens, top_probs)]
        safe_prob = probs[safe_token_id].item()
        unsafe_prob = probs[unsafe_token_id].item()
        safe_prob2 = probs[safe_token_id2].item()
        unsafe_prob2 = probs[unsafe_token_id2].item()

        # calculate specific token probability "safe" and "unsafe"
        # print(f"Top predicted_tokens: {top_predictions}")
        # print(f"▁safe: {safe_prob}")
        # print(f"▁unsafe: {unsafe_prob}")
        return top_predictions, safe_prob, unsafe_prob, safe_prob2, unsafe_prob2

    try:
        result, input_ids = moderate([
            {"role": "user", "content": text},
        ])

        # Extract logits for the first generated token
        first_token_logits = result.scores[0][0]

        # Get the indices of 'safe' and 'unsafe' tokens
        safe_token_id = tokenizer.encode('safe', add_special_tokens=False)[0]
        safe_token_id2 = tokenizer.encode('▁safe', add_special_tokens=False)[0]
        unsafe_token_id = tokenizer.encode('unsafe', add_special_tokens=False)[0]
        unsafe_token_id2 = tokenizer.encode('▁unsafe', add_special_tokens=False)[0]

        # Get logits for 'safe' and 'unsafe'
        safe_logit = first_token_logits[safe_token_id].item()
        safe_logit2 = first_token_logits[safe_token_id2].item()
        unsafe_logit = first_token_logits[unsafe_token_id].item()
        unsafe_logit2 = first_token_logits[unsafe_token_id2].item()
        top_predictions, safe_prob, unsafe_prob, safe_prob2, unsafe_prob2 = check_top_predicted_tokens(first_token_logits)
        generated_text = tokenizer.decode(result.sequences[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


        return {
            'categories': unsafe_prob > TROLL_THRESHOLD or unsafe_prob2 > TROLL_THRESHOLD,  # (bool) True or False
            'category_scores': max(unsafe_prob, unsafe_prob2),  # # (float) from 0 to 1
        }
    except Exception as err:
        print(f"LLAMAGUARD MODERATION FUNCTION: {err}")
        return {
            'categories': DEALING_ERROR,
            'category_scores': DEALING_ERROR,
        }


def evaluate_performance(df: pd.DataFrame) -> dict:
    results = {}
    y_true = df['label']

    for platform in MODERATION_PLATFORMS.keys():
        if platform not in EVAL_PLATFORMS: continue
        if f'{platform}_score' not in df.columns: continue

        y_score = df[f'{platform}_score']
        y_pred_binary = (y_score > TROLL_THRESHOLD).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        fpr, tpr, _ = roc_curve(y_true, y_score)

        results[platform] = {
            'Accuracy': accuracy_score(y_true, y_pred_binary),
            'Precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'Recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred_binary, zero_division=0),
            'AUC': auc(fpr, tpr),
            'TP': int(tp), 'FN': int(fn), 'FP': int(fp), 'TN': int(tn)
        }

    return results

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
def main():

    global LLAMA_MODELS
    LLAMA_MODELS = {
        'LlamaGuard-1': load_llamaguard_model(LLAMAGUARD1_ID) if 'LlamaGuard-1' in EVAL_PLATFORMS else None,
        'LlamaGuard-2': load_llamaguard_model(LLAMAGUARD2_ID) if 'LlamaGuard-2' in EVAL_PLATFORMS else None
    }

    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    RESULT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stats_paper'))
    DETAILED_RESULT_DIR = os.path.join(RESULT_DIR, 'evaluation_troll_detection_results_detail')
    os.makedirs(DETAILED_RESULT_DIR, exist_ok=True)

    all_results = {}
    for name in EVAL_DATASETS:
        print(f"\n{'='*20} Processing Dataset: {name.upper()} {'='*20}")
        
        if name == 'elf-hp':
            loader = ELFHPDataset(data_dir=DATA_DIR, task_name='ELF-HP')
            df = loader.get_dataframe()
            text_col, label_col = 'text', 'label'

        elif name == 'elf22':
            loader = ELF22Dataset(data_dir=DATA_DIR, task_name='counter_trollingy')
            df = loader.datasets['test'].to_pandas()
            text_col = 'InputContent'
            df['label'] = 1 
            label_col = 'label'

        elif name == 'qian_reddit':
            loader = QianDataset(file_path=os.path.join(DATA_DIR, "qian_reddit", "reddit.csv"), name='qian_reddit')
            df_full = loader.get_dataframe()
            df = df_full[df_full['label'] == 1].copy()
            text_col, label_col = 'text', 'label'

        elif name == 'qian_gab':
            loader = QianDataset(file_path=os.path.join(DATA_DIR, "qian_gab", "gab.csv"), name='qian_gab')
            df_full = loader.get_dataframe()
            df = df_full[df_full['label'] == 1].copy()
            text_col, label_col = 'text', 'label'

        elif name == 'conan':
            loader = ConanDataset(file_path=os.path.join(DATA_DIR, "conan", "CONAN_self_annotated.json"), name='conan')
            df = loader.get_dataframe()
            text_col, label_col = 'text', 'label'

        elif name == 'mtconan':
            loader = MTConanDataset(file_path=os.path.join(DATA_DIR, "mtconan", "Multitarget-CONAN.json"), name='mtconan')
            df = loader.get_dataframe()
            text_col, label_col = 'text', 'label'

        elif name == 'cadd':
            loader = CADDDataset(file_path=os.path.join(DATA_DIR, "cadd", "CADD_test.json"), name='cadd')
            df = loader.get_dataframe()
            text_col, label_col = 'text', 'label'

        elif name == 'covid_hate':
            loader = CovidDataset(file_path=os.path.join(DATA_DIR, "covid-hate", "covid-hate.csv"), name='covid_hate')
            df = loader.get_dataframe()
            text_col, label_col = 'text', 'label'

        elif name == 'ours':
            loader = SyntheticDataset(
                file_path=os.path.join(DATA_DIR, "data/simulation_outputs/troll_provocation_multiprofile/troll_provocation_multiprofile_sit_ours_openai_gpt-4o_results.json"),
                name='ours'
            )
            df = loader.get_dataframe()
            text_col, label_col = 'text', 'label'
        else:
            print(f"Skipping unknown dataset: {name}")
            continue

        df_eval = pd.DataFrame()
        df_eval['eval_text'] = df[text_col]
        df_eval['label'] = df[label_col].apply(lambda x: 0 if x == 0 else 1)
        
        print(f"Loaded {len(df_eval)} samples for evaluation.")
        if df_eval.empty:
            continue

        for platform, func_name in MODERATION_PLATFORMS.items():
            if platform not in EVAL_PLATFORMS: continue
            print(f"  - Getting scores from {platform}...")
            
            scores, categories = [], []
            func = globals()[func_name]
            for text in tqdm(df_eval['eval_text'], desc=f"  - {platform}", leave=False):
                result = func(text=str(text), platform=platform)
                if isinstance(result, dict):
                    scores.append(result.get('category_scores', DEALING_ERROR))
                    categories.append(result.get('categories', DEALING_ERROR))
                else: # PerspectiveAPI
                    scores.append(result)
            
            df_eval[f'{platform}_score'] = scores
            if categories:
                 df_eval[f'{platform}_categories'] = categories
        
        print(f"  - Generating predicted labels and saving detailed results for {name}...")
        
        for platform in MODERATION_PLATFORMS:
            if platform not in EVAL_PLATFORMS: continue
            if f'{platform}_categories' in df_eval.columns: # OpenAI, LlamaGuard
                df_eval[f'{platform}_label'] = df_eval[f'{platform}_categories'].astype(int)
            elif f'{platform}_score' in df_eval.columns: # PerspectiveAPI
                df_eval[f'{platform}_label'] = (df_eval[f'{platform}_score'] > TROLL_THRESHOLD).astype(int)
        
        json_path = os.path.join(DETAILED_RESULT_DIR, f"{name}_results.json")
        if os.path.exists(json_path):
            print(f"    -> Updating existing results file: {json_path}")
            existing_df = pd.read_json(json_path, orient='records', lines=True)
            merged_df = existing_df.combine_first(df_eval)
            merged_df.update(df_eval)
            final_df_to_save = merged_df
        else:
            print(f"    -> Creating new results file for {name}...")
            final_df_to_save = df_eval

        final_df_to_save.to_json(json_path, orient='records', lines=True, force_ascii=False)
        print(f"    -> Saved detailed JSON results to {json_path}")
        
        df_for_tsv = df_eval.copy()
        for col in df_for_tsv.columns:
            if df_for_tsv[col].dtype == 'object':
                df_for_tsv[col] = df_for_tsv[col].astype(str).str.replace(r'[\r\n\t]+', '  ', regex=True)
        
        tsv_path = os.path.join(DETAILED_RESULT_DIR, f"{name}_results.tsv")
        df_for_tsv.to_csv(tsv_path, sep='\t', index=False)
        print(f"    -> Saved detailed TSV results to {tsv_path}")

        all_results[name] = evaluate_performance(df_eval)
        print("\n\n" + "="*60)
        print("--- FINAL HARMFUL CONTENT DETECTION PERFORMANCE SUMMARY ---")
        print("="*60)
        
        summary_data = {}
        for dataset_name, platform_metrics in all_results.items():
            row = {}
            for platform_name, metrics in platform_metrics.items():
                row = summary_data.setdefault(platform_name, {})
                row[(dataset_name,'ACC')] = metrics.get('Accuracy', float('nan'))
                row[(dataset_name,'F1')] = metrics.get('F1-Score', float('nan'))
            summary_data[platform_name] = row
        
        summary_df = pd.DataFrame.from_dict(summary_data, orient='index')
        summary_df.index.name = 'platform'
        summary_df.columns = pd.MultiIndex.from_tuples(summary_df.columns)
        print(summary_df.to_markdown(floatfmt=".4f"))

        RESULT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
        result_path = os.path.join(RESULT_DIR, f'harmful_detection_result.tsv')

        if os.path.exists(result_path):
            existing_df = pd.read_csv(result_path, sep='\t', index_col=0, header=[0, 1])
            if existing_df.index.name != summary_df.index.name:
                existing_df.index.name = summary_df.index.name
            merged_df = existing_df.combine_first(summary_df)
            merged_df.update(summary_df) 
        else:
            merged_df = summary_df

        merged_df.to_csv(result_path, sep='\t')

if __name__ == '__main__':
    main()
    
