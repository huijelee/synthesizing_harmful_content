# input: reddit_policies/subreddit_list.json, reddit_threads
# output: user_profiles.json
import pandas as pd
import os
import re
import glob
import random
import numpy as np
import time
import json
import anthropic
from tqdm.std import tqdm
from argparse import ArgumentParser
from datetime import datetime
from typing import List, Dict
from collections import Counter, defaultdict

from dataset_classes import Thread, RedditPost, RedditComment, dataset_dict, FLAG_STS, FLAG_SRS, FLAG_TS, FLAG_ORDERED, FLAG_RS
from openai import OpenAI

from utils import generate_openai_model
from private_keys import OPENAI_API_KEY

# ------------------------- CONFIG ------------------------- #
MODEL = 'gpt-4o'
TEMPERATURE = 0.7
MAX_TOKENS = 1024
OUTPUT_DIR = 'synthesizing_harmful_content/data/synthetic_userprofiles'
IS_SHUFFLE = True

CHARACTER_TYPES = {
    "newcomer": {
        "experience_range": 'from 0 to 12 months',
        "policy_awareness_probs": {
            "low": 0.6,
            "moderate": 0.3,
            "high": 0.1
        }
    },
    "regular": {
        "experience_range": 'from 12 to 36 months',
        "policy_awareness_probs": {
            "low": 0.2,
            "moderate": 0.6,
            "high": 0.2
        }
    },
    "longtimeuser": {
        "experience_range": 'above 36 months',
        "policy_awareness_probs": {
            "low": 0.1,
            "moderate": 0.3,
            "high": 0.6
        }
    }
}


CATEGORIES = [
    "General Content", "Discussion", "Educational", "Entertainment", 
    "Hobbies and Occupations", "Lifestyle", "Technology", 
    "Humor", "Animals", "Other"
]

PERSONA_GEN_PROMPT = """You have been a Reddit user for nearly 20 years, making you highly specialized in predicting Reddit user profiles.

Generate a synthetic Reddit user profile based on the following parameters:

Character Type: {char_type}
Reddit Thread: {thread}
Top-visited Subreddits: {top_subs}
Recent Subreddits: {rec_subs}

The output should be a JSON object with the following predefined keys:
- basic_profile
- behavioral_pattern

### Explanation of Each Key and Sub-Key:

1. basic_profile (dict)
   - username (str): A plausible Reddit username (no PII).
   - account_age (str): How long they have been on Reddit. (e.g. "3 months")
   - bio (str): A detailed description of the user. Include background, interests, dislikes, location, typical online hours, job or occupation, relationship status, etc. Be as specific as possible.
   - top_subreddit_categories: At most 3 visited categories from the category set {{General Content, Discussion, Educational, Entertainment, Hobbies and Occupations, Lifestyle, Technology, Humor, Animals, Other}}
   
2. behavioral_pattern (dict)
   - knowledge_background (str): Detailed description of user’s knowledge or expertise. It can elaborate on how they acquired it (e.g., educational background, self-teaching, work).
   - typical_text_length (str): The usual length of their posts (e.g., "brief comments", "1-2 sentences", "short paragraph", "multiple paragraphs", "long-form content").

### Requirements:
1. Ensure all generated data is realistic for a Reddit environment.
2. Return the output strictly in JSON format with the predefined keys described above.
"""


def weighted_choice_from_dict(prob_dict):
    """
    Return a single key from a dictionary 
    based on weighted probabilities.
    """
    items = list(prob_dict.items())
    keys, weights = zip(*items)
    return random.choices(keys, weights=weights, k=1)[0]

def pick_sublist(original_list, min_k=1, max_k=3):
    """
    Pick a random sublist from an original list
    of length from min_k to max_k (capped by length).
    """
    size = random.randint(min_k, min(max_k, len(original_list)))
    return random.sample(original_list, size)


class SyntheticRedditUserGenerator:
    def __init__(self, openai_client, subreddit_dict):
        """
        openai_client: an openai API client instance
        subreddit_dict: list of subreddits to choose from
        """
        self.client = openai_client
        self.subreddit_dict = subreddit_dict
        self.subreddit_list = list(self.subreddit_dict.keys())

    def _find_last_userprofile_index(self, char_type):
        # Check existing files using glob to find the last number
        last_num = -1
        pattern = os.path.join(OUTPUT_DIR, f"{char_type}_*.json")
        existing_files = glob.glob(pattern)
        
        for filename in existing_files:
            base_name = os.path.basename(filename)
            try:
                file_num = int(base_name.split('_')[1].split('.')[0])
                last_num = max(last_num, file_num)
            except ValueError:
                continue
        
        return last_num + 1

    def run(self, thread_jsons, char_counts, logging=False):
        thread_generator = self.yield_thread(thread_jsons)

        for char_type, count in tqdm(char_counts.items(), desc="Character Types"):
            # Start generating from the next number after the last one found
            start_num = self._find_last_userprofile_index(char_type)
            for i in tqdm(range(start_num, start_num + count), desc=f"Generating {char_type}"):
                thread_data = next(thread_generator, None)
                if not thread_data:
                    print(f"No more threads. Partial characters generated ({i}/{count}) for {char_type}")
                    return
                character = self.generate_character_profile(char_type, thread_data)
                fname = f"{OUTPUT_DIR}/{char_type}_{i}.json"
                with open(fname, 'w') as f:
                    json.dump(character, f, indent=2)
        print(f"All characters are generated ({char_counts}) at {OUTPUT_DIR}")

    def yield_thread(self, thread_jsons):
        for json_file in thread_jsons:
            try:
                # Update batch number from filename
                batch_match = re.search(r'batch_(\d+)', json_file)
                if batch_match:
                    self.current_batch = int(batch_match.group(1))

                # Load thread from JSON
                threads = Thread.load_from_batch_json(json_file)

                for thread in threads:
                    # subreddit = thread.post.subreddit
                    yield thread    
            
            except Exception as e:
                print(f"\nError processing {json_file}:  {str(e)}")
                continue

    def parse_response(self, response_text):
        "output: user_profile (dict)"
        # Remove markdown formatting if present
        response_text = response_text.strip()
        if not response_text:
            print("No response from model.")
            return None
        if response_text.startswith("```json"):
            # Remove the starting ```
            response_text = re.sub(r"^```json\s*", "", response_text)
            response_text = re.sub(r"\s*```", "", response_text)

        try:
            user_profile = json.loads(response_text)
        except json.JSONDecodeError as e:
            match = re.search(r"``````", response_text, flags=re.DOTALL)
            if match:
                user_profile_text = match.group(1)
                user_profile = json.loads(user_profile_text)
            else:
                print("Failed to parse model output as JSON.")
                return None
        
        return user_profile
 

    def generate_character_profile(self, char_type, thread_data):
        """
        1) Determine or sample the user's top_subreddit_categories, top_subreddits, recent_subreddits
        2) Insert them into the prompt
        3) Call the LLM
        4) Parse the response, update JSON accordingly
        5) Return final JSON
        """
        # 1) Weighted picks for knowledge depth & policy awareness
        ctype_info = CHARACTER_TYPES[char_type]
        # knowledge_depth = weighted_choice_from_dict(ctype_info['knowledge_depth_probs'])
        policy_awareness = weighted_choice_from_dict(ctype_info['policy_awareness_probs'])

        # 2) Generate random picks for categories and subreddits
        # top_categories = pick_sublist(CATEGORIES, 1, 3)
        top_subs = pick_sublist(self.subreddit_list, 1, 3)
        rec_subs = pick_sublist(self.subreddit_list, 1, 3)

        # 3) Build the custom prompt 
        # We can pass these values to the large language model for reference.
        # e.g., We embed them in {char_type} plus some text in the 'thread' or appended instructions
        extended_prompt = PERSONA_GEN_PROMPT.format(
            char_type=f"{char_type}, experience_range: {ctype_info['experience_range']}, policy_awareness: {policy_awareness}",
            top_subs=top_subs,
            rec_subs=rec_subs,
            thread=thread_data,
        )

        prompt_for_model = extended_prompt
        
        # 4) Call the LLM
        response_text = generate_openai_model(
            data_id=f"{char_type}_{policy_awareness}",
            client=self.client,
            messages=prompt_for_model,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            model_name=MODEL,
            json_output=False,
        )

        user_profile = self.parse_response(response_text)
        if not user_profile:
            return None

        # 5) Overwrite the final JSON content with the chosen categories and subreddits
        # Ensure 'basic_profile' exists
        if 'basic_profile' not in user_profile:
            user_profile['basic_profile'] = {}

        # Insert or overwrite
        # user_profile['basic_profile']['top_subreddit_categories'] = ", ".join(top_categories)
        user_profile['basic_profile']['top_subreddits'] = ", ".join(top_subs)
        user_profile['basic_profile']['recent_subreddits'] = ", ".join(rec_subs)

        if 'behavioral_pattern' not in user_profile:
            user_profile['behavioral_pattern'] = {}

        return user_profile


def main():
    # load subreddit rules
    subreddit_rules_fname = 'synthesizing_harmful_content/data/reddit_policies/subreddit_rules.json'
    reddit_threads_dir = 'synthesizing_harmful_content/data/reddit_random_threads'
    char_counts = {
        "newcomer": 10,
        "regular": 10,
        "longtimeuser": 10,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(subreddit_rules_fname, 'r', encoding='utf-8') as f:
        subreddit_dict = json.load(f)

    openai_client = OpenAI(api_key=OPENAI_API_KEY[0])


    thread_jsons = sorted(glob.glob(os.path.join(reddit_threads_dir, 'batch_*.json')), key=lambda x: int(re.search(r'batch_(\d+)', x).group(1)))
    if IS_SHUFFLE:
        random.shuffle(thread_jsons)
    generator = SyntheticRedditUserGenerator(openai_client, subreddit_dict)
    generator.run(thread_jsons=thread_jsons, char_counts=char_counts)


if __name__ == "__main__":
    main()
