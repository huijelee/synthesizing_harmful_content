## main_simulation.py ##
import os

import sys
import re
import gc
import json
import logging
import random
from copy import deepcopy
from glob import glob

import torch
import numpy as np
import pandas as pd
import anthropic
from openai import OpenAI
from vllm import LLM, SamplingParams

from omegaconf import DictConfig, OmegaConf

from dataset_classes import (
    dataset_dict,
    Thread,
    RedditComment,
    FLAG_STS,
    FLAG_TS,
    FLAG_RS,
    FLAG_SRS
)
from utils import (
    TS_GUIDELINE,
    RS_GUIDELINE,
    generate_openai_model as util_generate_openai_model,
    generate_claude_model as util_generate_claude_model,
    generate_vllm_model as util_generate_vllm_model,
    NON_ENGLISH_SUBREDDIT_NAMES,
    calculate_shannon_entropy,
    sanitize_data
)

from evaluation import (
    evaluate_ts_fidelity,
    evaluate_rs_fidelity,
    calculate_hd_and_jsd,
    evaluate_perplexity,
    evaluate_conditional_perplexity,
    calculate_self_bleu,
    calculate_ttr,
    calculate_mattr,
    calculate_vocab_size,
    calculate_self_bleu_with_tiktoken,
    predict_trolling_types,
    calculate_refusal_rate,
    analyze_language_match_rate,
    save_evaluation_summary
)

from generate_userprofiles import CHARACTER_TYPES

from private_keys import OPENAI_API_KEY, CLAUDE_API_KEY, DEEPINFRA_API_KEY


DEFAULT_CFG_NAME = "APT.yaml"
SEED = 42
random.seed(SEED)
logger = logging.getLogger(__name__) 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S", 
    handlers=[logging.StreamHandler(sys.stdout)], 
)


def get_llm_client(client_type: str, model_name: str, cfg: DictConfig):
    """
    Factory function to get an LLM client.

    Args:
        client_type (str): Client type (e.g., "openai", "claude", "vllm", "deepinfra").
        model_name (str): Name of the model to use.
        cfg (DictConfig): OmegaConf configuration object.

    Returns:
        object: Initialized LLM client object.

    Raises:
        ValueError: If the client type is unsupported.
    """
    if client_type == 'openai':
        return OpenAI(api_key=OPENAI_API_KEY[0])
    elif client_type == 'deepinfra':
        return OpenAI(api_key=DEEPINFRA_API_KEY[0], base_url="https://api.deepinfra.com/v1/openai")
    elif client_type == 'claude':
        return anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    elif client_type == 'vllm':
        return LLM(model=model_name, download_dir=cfg.model_cache_dir, trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported client type: {client_type}")


def make_troll_provocation_filename_convention(output_subdir, task_cfg, troll_provocation_prompt_type, client_type, short_model_name):
    return f"{output_subdir}_{task_cfg.get('thread_data_type', 'unknown_threads')}_{troll_provocation_prompt_type}_{client_type}_{short_model_name}"

def make_elf_countering_filename_convention(output_subdir_elf, client_type, short_model_name, elf_countering_prompt_type, troll_prov_file_name_base):
    return f"{output_subdir_elf}_{client_type}_{short_model_name}_{elf_countering_prompt_type}_from_{troll_prov_file_name_base}"


def generate_troll_provocation_dataset(
    task_cfg: DictConfig, # Task-specific config (e.g., num_samples, strategies, model info)
    global_cfg: DictConfig,
    userprofiles: dict,
    threads: list,
    troll_examples: list,
):
    """
    Generates a troll provocation dataset.
    Refactored to use configuration objects.

    Args:
        task_cfg (DictConfig): Configuration for this specific generation task.
        global_cfg (DictConfig): Global application configuration.
        userprofiles (dict): Dictionary of user profiles.
        threads (list): List of Reddit thread objects.
        troll_examples (list): List of troll example data.

    Returns:
        tuple: A list of generated results and the number of generation errors.
    """
    logger.info(f"Starting troll provocation dataset generation: {task_cfg.get('name', 'Unnamed Task')}")

    client_type = task_cfg.client_type
    model_name = task_cfg.model_name
    client = get_llm_client(client_type, model_name, global_cfg)

    num_samples = task_cfg.get('num_samples', global_cfg.default_generation_params.num_samples_troll_provocation_default)
    temperature = task_cfg.get('temperature', global_cfg.default_generation_params.temperature)
    top_p = task_cfg.get('top_p', global_cfg.default_generation_params.top_p)
    max_tokens = task_cfg.get('max_tokens', global_cfg.default_generation_params.max_tokens)

    troll_provocation_prompt_type = task_cfg.troll_provocation_prompt_type
    troll_provocation_target = task_cfg.troll_provocation_target

    user_strategies = task_cfg.user_strategies
    thread_strategies = task_cfg.thread_strategies
    troll_strategies = task_cfg.troll_strategies
    current_user_strategy = random.choice(user_strategies) if isinstance(user_strategies, list) else user_strategies
    current_thread_strategy = random.choice(thread_strategies) if isinstance(thread_strategies, list) else thread_strategies
    current_troll_strategy = random.choice(troll_strategies) if isinstance(troll_strategies, list) else troll_strategies
    selected_subreddits = task_cfg.get('selected_subreddits', []) # Default to empty list
    non_english_subreddit_only = task_cfg.get('non_english_subreddit_only', False)

    output_subdir = OmegaConf.select(task_cfg, 'output_subdir', default="troll_provocation")
    output_dir = os.path.join(global_cfg.output_base_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    brief_model_name = re.split("/", model_name)[-1]
    file_name_base = make_troll_provocation_filename_convention(output_subdir, task_cfg, troll_provocation_prompt_type, client_type, brief_model_name)
    output_file_tsv = os.path.join(output_dir, f"{file_name_base}_results.tsv")
    output_file_json = os.path.join(output_dir, f"{file_name_base}_results.json")

    organized_profiles = {"newcomer": [], "regular": [], "longtimeuser": []}
    profile_keys = list(userprofiles.keys())
    for key, profile in userprofiles.items():
        user_type = key.split('_')[0] # Extract type from filename convention (e.g., newcomer_0)
        if user_type in organized_profiles:
            organized_profiles[user_type].append(profile)

    organized_troll_examples = {}
    for example in troll_examples:
        try:
            sts_label = FLAG_STS[example['TrollL'] - 1]
            if sts_label not in organized_troll_examples:
                organized_troll_examples[sts_label] = []
            organized_troll_examples[sts_label].append(example)
        except (KeyError, TypeError, IndexError) as e:
            sts_label = "uncategorized"
            if sts_label not in organized_troll_examples:
                organized_troll_examples[sts_label] = []
            organized_troll_examples[sts_label].append(example)
            logger.warning(f"Could not categorize troll example {example.get('id', 'N/A')} due to 'TrollL' issue: {e}")

    results_data = [] # List to store generated results
    generation_errors = 0 # Counter for errors during generation

    usable_threads_list = threads
    if non_english_subreddit_only:
        filtered_threads_list = [th for th in threads if hasattr(th, 'post') and th.post.subreddit in NON_ENGLISH_SUBREDDIT_NAMES]
        if filtered_threads_list: usable_threads_list = filtered_threads_list
        else: logger.warning("No non-English threads found, using all threads as fallback.")
    if selected_subreddits:
        filtered_threads_list = [th for th in threads if hasattr(th, 'post') and th.post.subreddit in selected_subreddits]
        if filtered_threads_list:
            usable_threads_list = filtered_threads_list
        else:
            logger.warning(f"No threads found for selected subreddits {selected_subreddits}. Using all threads.")

    if not usable_threads_list:
        logger.error("No usable threads available. Cannot proceed with sample generation.")
        generation_errors += 1
        return [], 0

    for i in range(num_samples):
        ### 1. Select user profile
        # two expected outputs
        selected_user_profile = None
        profile_key = "(usertype)_(index)" # Default key

        if current_user_strategy == "random":
            if profile_keys: # Check if there are available profile keys
                profile_key = random.choice(profile_keys)
                selected_user_profile = userprofiles[profile_key]
        elif current_user_strategy == "evenly_distributed":
            user_types_available = [ut for ut in CHARACTER_TYPES.keys() if organized_profiles.get(ut)]
            if user_types_available:
                selected_type = user_types_available[i % len(user_types_available)] # Select type by cycling through
                if organized_profiles[selected_type]: # Check if profiles of this type exist
                    selected_user_profile = random.choice(organized_profiles[selected_type])
                    # Generate profile key (e.g., "newcomer_0")
                    profile_key = f"{selected_type}_{organized_profiles[selected_type].index(selected_user_profile)}"
        elif current_user_strategy in CHARACTER_TYPES.keys(): # Specific user type directly specified
            if organized_profiles.get(current_user_strategy):
                selected_user_profile = random.choice(organized_profiles[current_user_strategy])
                profile_key = f"{current_user_strategy}_{organized_profiles[current_user_strategy].index(selected_user_profile)}"

        if selected_user_profile is None: # Use default if profile selection fails
             if profile_keys:
                profile_key = random.choice(profile_keys)
                selected_user_profile = userprofiles[profile_key]
             else: # If no profiles are available at all
                logger.warning(f"Sample {i}: No user profiles available for strategy {current_user_strategy}. Skipping user profile selection.")
                generation_errors += 1
                continue

        ### 2. Select thread
        # two expected outputs
        selected_thread_obj = None
        thread_key = -1 # Default thread identifier (index or original ID)

        if current_thread_strategy == "random":
            thread_key = random.randint(0, len(usable_threads_list) - 1)
            selected_thread_obj = usable_threads_list[thread_key]
        elif current_thread_strategy == "in_order":
            thread_key = i % len(usable_threads_list) # Select threads in order
            selected_thread_obj = usable_threads_list[thread_key]

        if selected_thread_obj is None: # If thread selection fails, use default (random)
            thread_key = random.randint(0, len(usable_threads_list) - 1)
            selected_thread_obj = usable_threads_list[thread_key]
            logger.warning(f"Sample {i+1}: Thread selection strategy {current_thread_strategy} failed or no results from subreddit filtering. Using random thread.")

        ### 3. Select troll example and determine trolling strategy
        # three expected outputs
        selected_troll_example_data = None
        troll_example_key = -1
        target_prompt_trolling_strategy = random.choice(FLAG_TS) # Use random TS by default, Actual trolling strategy to be used in the prompt (e.g., '[aggression]')

        if current_troll_strategy == "random":
            if troll_examples:
                troll_example_key = random.randint(0, len(troll_examples) - 1)
                selected_troll_example_data = troll_examples[troll_example_key]
        elif current_troll_strategy == "in_order":
            if troll_examples:
                troll_example_key = i % len(troll_examples) # Select examples in order
                selected_troll_example_data = troll_examples[troll_example_key]
                # Determine TS for prompt based on STS label ('TrollL') of the example
                if selected_troll_example_data.get('TrollL') == 1: # Overt (random from the first 3 of FLAG_TS)
                    target_prompt_trolling_strategy = random.choice(FLAG_TS[:3])
                elif selected_troll_example_data.get('TrollL') == 2: # Covert (random from the last 3 of FLAG_TS)
                    target_prompt_trolling_strategy = random.choice(FLAG_TS[3:])
        elif current_troll_strategy == "evenly_distributed":
            sts_types_available = [sts for sts in FLAG_STS if organized_troll_examples.get(sts)]
            if sts_types_available:
                selected_sts_type = sts_types_available[i % len(sts_types_available)]
                if organized_troll_examples[selected_sts_type]:
                    type_examples_list = organized_troll_examples[selected_sts_type]
                    organized_troll_example_key = (i // len(sts_types_available)) % len(type_examples_list)
                    selected_troll_example_data = type_examples_list[organized_troll_example_key]
                    troll_example_key = selected_troll_example_data['sample_index']

                    if selected_sts_type == FLAG_STS[0]: # Overt
                        target_prompt_trolling_strategy = random.choice(FLAG_TS[:3])
                    elif selected_sts_type == FLAG_STS[1]: # Covert
                        target_prompt_trolling_strategy = random.choice(FLAG_TS[3:])
        elif current_troll_strategy in FLAG_STS:
            if organized_troll_examples.get(current_troll_strategy):
                type_examples_list = organized_troll_examples[current_troll_strategy]
                organized_troll_example_key = random.randint(0, len(type_examples_list) - 1)
                selected_troll_example_data = type_examples_list[troll_example_key]
                troll_example_key = selected_troll_example_data['sample_index']

                if current_troll_strategy == FLAG_STS[0]: # Overt
                    target_prompt_trolling_strategy = random.choice(FLAG_TS[:3])
                elif current_troll_strategy == FLAG_STS[1]: # Covert
                    target_prompt_trolling_strategy = random.choice(FLAG_TS[3:])
            else: # If no examples for the specified STS type
                logger.warning(f"Troll example selection: No examples for STS strategy '{current_troll_strategy}'. Substituting with a random example.")
                if troll_examples: # Random selection from all available examples
                    troll_example_key = random.randint(0, len(troll_examples) - 1)
                    selected_troll_example_data = troll_examples[troll_example_key]

        if selected_troll_example_data is None: # If troll example selection fails
            if troll_examples: # Random selection from all available examples
                troll_example_key = random.randint(0, len(troll_examples) - 1)
                selected_troll_example_data = troll_examples[troll_example_key]
                logger.warning(f"Sample {i+1}: Troll example selection strategy {current_troll_strategy} failed. Using random example.")
            else: # If no examples are available at all
                logger.error(f"Sample {i+1}: No troll examples available. Skipping sample generation.")
                generation_errors += 1
                continue # Skip current sample generation

        # Brief troll example information to be used in the prompt
        brief_selected_troll_example = {
            'Title': selected_troll_example_data.get('Title', 'No Title'),
            'Post': selected_troll_example_data.get('Post', 'No Post Body'),
            'Comment': selected_troll_example_data.get('Troll', 'No Comment Body'), # Assuming the 'Troll' field in the original data is the example comment
        }

        prompt_template_key = troll_provocation_prompt_type
        prompt_template = OmegaConf.select(global_cfg, f"prompts.TrollAgentProvocationPrompt.{prompt_template_key}")
        if not prompt_template:
            logger.error(f"Prompt template '{prompt_template_key}' not found. Skipping sample {i+1}.")
            generation_errors += 1
            continue

        # Determine which response format prompt to use
        if troll_provocation_prompt_type == 'vanilla' or troll_provocation_prompt_type == 'in_only':
            current_response_format_prompt = global_cfg.prompts.RESPONSE_FORMAT_PROMPT
            target_prompt_trolling_strategy = "None"
        else:
            current_response_format_prompt = global_cfg.prompts.RESPONSE_FORMAT_PROMPT_WITH_TS

        # Prepare parameters for prompt formatting
        format_params = {
            'user_profile': json.dumps(selected_user_profile),
            'troll_example': json.dumps(brief_selected_troll_example),
            'thread': str(selected_thread_obj),
            'trolling_strategy': target_prompt_trolling_strategy,
            'target_comment': global_cfg.prompts.TrollingProvocationTarget[troll_provocation_target],
            'trolling_strategy_descriptions': TS_GUIDELINE,
            'response_format_prompt': current_response_format_prompt,
        }

        try:
            troll_provocation_message = prompt_template.format(**format_params)
        except KeyError as e:
            logger.error(f"Missing key during prompt formatting: {e}. Prompt type: {prompt_template_key}. Skipping sample {i+1}.")
            generation_errors += 1
            continue

        data_id = f"{task_cfg.get('name', 'TrollProvocation')}_{i}"
        troll_comment_json_output = None

        if client_type in ['openai', 'deepinfra']:
            troll_comment_json_output = util_generate_openai_model(
                data_id=data_id, client=client, messages=troll_provocation_message,
                temperature=temperature, max_tokens=max_tokens, model_name=model_name, json_output=True
            )
        elif client_type == 'claude':
            troll_comment_json_output = util_generate_claude_model(
                data_id=data_id, client=client, messages=troll_provocation_message,
                temperature=temperature, max_tokens=max_tokens, model_name=model_name, json_output=True
            )
        elif client_type == 'vllm':
             troll_comment_json_output = util_generate_vllm_model(
                data_id=data_id, client=client, messages=troll_provocation_message,
                temperature=temperature, top_p=top_p, max_tokens=max_tokens, json_output=True
            )
        else:
            logger.error(f"Unsupported client type '{client_type}' for sample {i+1}.")
            generation_errors += 1
            continue

        if troll_comment_json_output and isinstance(troll_comment_json_output, dict):
            try:
                troll_thread_obj = deepcopy(selected_thread_obj)
                troll_thread_obj.comments.append(RedditComment.from_dict(troll_comment_json_output))
                troll_thread_obj.renew_max_depth()

                result_entry = {
                    "id": data_id,
                    "user_profile_key": profile_key,
                    "user_profile": selected_user_profile,
                    "thread_key": thread_key,
                    "original_thread": selected_thread_obj.to_dict(),
                    "troll_example_key": troll_example_key,
                    "troll_example": selected_troll_example_data, 
                    "strategy_user_selection": current_user_strategy,
                    "trolling_strategy": target_prompt_trolling_strategy, 
                    "thread_strategy": current_thread_strategy,
                    "troll_strategy": current_troll_strategy,
                    "troll_comment": troll_comment_json_output,
                    "troll_thread": troll_thread_obj.to_dict(),
                }
                results_data.append(result_entry)
                logger.info(f"Sample {i+1}/{num_samples} generation complete: Target TS: {target_prompt_trolling_strategy}")

            except Exception as e:
                logger.error(f"Error processing generated comment for sample {i+1}/{num_samples}: {e}. JSON: {troll_comment_json_output}")
                generation_errors += 1
        else:
            logger.warning(f"Failed to generate or parse valid JSON for sample {i+1}/{num_samples}. Output: {troll_comment_json_output}")
            generation_errors += 1

    if results_data:
        results_data = sanitize_data(results_data)
        try:
            with open(output_file_json, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x), ensure_ascii=False)
            df = pd.DataFrame(results_data)
            df.to_csv(output_file_tsv, sep='\t', index=False, encoding='utf-8')
            logger.info(f"{len(results_data)} troll provocation samples generated. Saved to: {output_file_tsv} and {output_file_json}")
        except Exception as e:
            logger.error(f"Error saving results to CSV/JSON: {e}. JSON may have been saved at {output_file_json}")
    else:
        logger.info("No results were generated to save.")

    # Clean up memory if using vLLM
    if client_type == 'vllm' and client is not None:
        del client
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results_data, generation_errors


def generate_multiple_profiles_troll_provocation_dataset(
    task_cfg: DictConfig,
    global_cfg: DictConfig,
    userprofiles: dict,
    threads: list,
    troll_examples: list, # Changed from elf22_examples
):
    """
    Generates a troll provocation dataset where each thread is responded to by multiple user profiles.
    """
    logger.info(f"Starting MULTI-PROFILE troll provocation dataset generation: {task_cfg.get('name', 'Unnamed MultiProfile Task')}")

    client_type = task_cfg.client_type
    model_name = task_cfg.model_name
    client = get_llm_client(client_type, model_name, global_cfg)

    num_threads_to_process = task_cfg.get('num_samples', 10) # Renamed from num_samples for clarity
    num_profiles_per_thread = task_cfg.get('num_profiles_per_thread', 300)
    temperature = task_cfg.get('temperature', global_cfg.default_generation_params.temperature)
    top_p = task_cfg.get('top_p', global_cfg.default_generation_params.top_p)
    max_tokens = task_cfg.get('max_tokens', global_cfg.default_generation_params.max_tokens)

    troll_provocation_prompt_type = task_cfg.troll_provocation_prompt_type
    troll_provocation_target = task_cfg.troll_provocation_target

    user_strategies = task_cfg.user_strategies
    thread_strategies = task_cfg.thread_strategies
    troll_strategies = task_cfg.troll_strategies
    current_user_strategy = random.choice(user_strategies) if isinstance(user_strategies, list) else user_strategies
    current_thread_strategy = random.choice(thread_strategies) if isinstance(thread_strategies, list) else thread_strategies
    current_troll_strategy = random.choice(troll_strategies) if isinstance(troll_strategies, list) else troll_strategies
    selected_subreddits = task_cfg.get('selected_subreddits', [])
    non_english_subreddit_only = task_cfg.get('non_english_subreddit_only', False)

    output_subdir = OmegaConf.select(task_cfg, 'output_subdir', default="troll_provocation_multiprofile")
    output_dir = os.path.join(global_cfg.output_base_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    brief_model_name = re.split("/", model_name)[-1]
    file_name_base = make_troll_provocation_filename_convention(output_subdir, task_cfg, troll_provocation_prompt_type, client_type, brief_model_name)
    output_file_tsv = os.path.join(output_dir, f"{file_name_base}_results.tsv")
    output_file_json = os.path.join(output_dir, f"{file_name_base}_results.json")

    organized_profiles = {"newcomer": [], "regular": [], "longtimeuser": []}
    profile_keys = list(userprofiles.keys())
    for key, profile in userprofiles.items():
        user_type = key.split('_')[0]
        if user_type in organized_profiles:
            organized_profiles[user_type].append({"key": key, "data": profile}) # Store key along with data

    organized_troll_examples = {}
    for example in troll_examples:
        try:
            sts_label = FLAG_STS[example['TrollL'] - 1]
            if sts_label not in organized_troll_examples:
                organized_troll_examples[sts_label] = []
            organized_troll_examples[sts_label].append(example)
        except (KeyError, TypeError, IndexError) as e:
            sts_label = "uncategorized"
            if sts_label not in organized_troll_examples:
                organized_troll_examples[sts_label] = []
            organized_troll_examples[sts_label].append(example)
            logger.warning(f"Could not categorize troll example {example.get('id', 'N/A')} due to 'TrollL' issue: {e}")

    results_data = []
    generation_errors = 0

    usable_threads_list = threads
    if non_english_subreddit_only:
        filtered_threads_list = [th for th in threads if hasattr(th, 'post') and th.post.subreddit in NON_ENGLISH_SUBREDDIT_NAMES]
        if filtered_threads_list: usable_threads_list = filtered_threads_list
        else: logger.warning("No non-English threads found, using all threads as fallback.")
    elif selected_subreddits:
        filtered_threads_list = [th for th in threads if hasattr(th, 'post') and th.post.subreddit in selected_subreddits]
        if filtered_threads_list:
            usable_threads_list = filtered_threads_list
        else: 
            logger.warning(f"No threads found for selected subreddits {selected_subreddits}. Using all threads.")

    if not usable_threads_list:
        logger.error("No usable threads for multi-profile generation.")
        return [], 0

    for thread_iter_idx in range(min(num_threads_to_process, len(usable_threads_list))):
        ### 1. Select thread
        # two expected outputs
        selected_thread_obj = None
        thread_key = -1

        if current_thread_strategy == "random":
            thread_key = random.randint(0, len(usable_threads_list) - 1)
            selected_thread_obj = usable_threads_list[thread_key]
        elif current_thread_strategy == "in_order":
            thread_key = thread_iter_idx % len(usable_threads_list) # Select threads in order
            selected_thread_obj = usable_threads_list[thread_key]

        if selected_thread_obj is None: # If thread selection fails, use default (random)
            thread_key = random.randint(0, len(usable_threads_list) - 1)
            selected_thread_obj = usable_threads_list[thread_key]
            logger.warning(f"Sample {thread_iter_idx+1}: Thread selection strategy {current_thread_strategy} failed or no results from subreddit filtering. Using random thread.")

        logger.info(f"Processing thread {thread_iter_idx + 1}/{num_threads_to_process} (ID: {selected_thread_obj.post.id if selected_thread_obj else 'N/A'}) "
                    f"with {num_profiles_per_thread} profiles.")

        # For each thread, generate responses from multiple profiles
        for profile_iter_idx in range(num_profiles_per_thread):
            ### 2. Select user profile
            # two expected outputs
            selected_user_profile = None
            profile_key = "(usertype)_(index)" 

            if current_user_strategy == "random":
                if profile_keys: # Check if there are available profile keys
                    profile_key = random.choice(profile_keys)
                    selected_user_profile = userprofiles[profile_key]
            elif current_user_strategy == "evenly_distributed":
                user_types_available = [ut for ut in CHARACTER_TYPES.keys() if organized_profiles.get(ut)]
                if user_types_available:
                    selected_type = user_types_available[profile_iter_idx % len(user_types_available)] # Select type by cycling through
                    if organized_profiles[selected_type]: # Check if profiles of this type exist
                        selected_user_profile = random.choice(organized_profiles[selected_type])
                        # Generate profile key (e.g., "newcomer_0")
                        profile_key = f"{selected_type}_{organized_profiles[selected_type].index(selected_user_profile)}"
            elif current_user_strategy in CHARACTER_TYPES.keys(): # Specific user type directly specified
                if organized_profiles.get(current_user_strategy):
                    selected_user_profile = random.choice(organized_profiles[current_user_strategy])
                    profile_key = f"{current_user_strategy}_{organized_profiles[current_user_strategy].index(selected_user_profile)}"

            if selected_user_profile is None: # Use default if profile selection fails
                if profile_keys:
                    profile_key = random.choice(profile_keys)
                    selected_user_profile = userprofiles[profile_key]
                else: # If no profiles are available at all
                    logger.warning(f"Thread {thread_iter_idx+1}, Profile attempt {profile_iter_idx+1}: No user profiles available for strategy {current_user_strategy}. Skipping user profile selection.")
                    generation_errors += 1
                    continue

            ### 3. Select troll example and determine trolling strategy
            # three expected outputs
            selected_troll_example_data = None
            troll_example_key = -1
            target_prompt_trolling_strategy = random.choice(FLAG_TS)

            if current_troll_strategy == "random":
                if troll_examples:
                    troll_example_key = random.randint(0, len(troll_examples) - 1)
                    selected_troll_example_data = troll_examples[troll_example_key]
            elif current_troll_strategy == "in_order":
                if troll_examples:
                    troll_example_key = profile_iter_idx % len(troll_examples) # Select examples in order
                    selected_troll_example_data = troll_examples[troll_example_key]
                    # Determine TS for prompt based on STS label ('TrollL') of the example
                    if selected_troll_example_data.get('TrollL') == 1: # Overt (random from the first 3 of FLAG_TS)
                        target_prompt_trolling_strategy = random.choice(FLAG_TS[:3])
                    elif selected_troll_example_data.get('TrollL') == 2: # Covert (random from the last 3 of FLAG_TS)
                        target_prompt_trolling_strategy = random.choice(FLAG_TS[3:])
            elif current_troll_strategy == "evenly_distributed":
                sts_types_available = [sts for sts in FLAG_STS if organized_troll_examples.get(sts)]
                if sts_types_available:
                    selected_sts_type = sts_types_available[profile_iter_idx % len(sts_types_available)]
                    if organized_troll_examples[selected_sts_type]:
                        type_examples_list = organized_troll_examples[selected_sts_type]
                        organized_troll_example_key = (profile_iter_idx // len(sts_types_available)) % len(type_examples_list)
                        selected_troll_example_data = type_examples_list[organized_troll_example_key]
                        troll_example_key = selected_troll_example_data['sample_index']

                        if selected_sts_type == FLAG_STS[0]: # Overt
                            target_prompt_trolling_strategy = random.choice(FLAG_TS[:3])
                        elif selected_sts_type == FLAG_STS[1]: # Covert
                            target_prompt_trolling_strategy = random.choice(FLAG_TS[3:])
            elif current_troll_strategy in FLAG_STS:
                if organized_troll_examples.get(current_troll_strategy):
                    type_examples_list = organized_troll_examples[current_troll_strategy]
                    organized_troll_example_key = random.randint(0, len(type_examples_list) - 1)
                    selected_troll_example_data = type_examples_list[troll_example_key]
                    troll_example_key = selected_troll_example_data['sample_index']

                    if current_troll_strategy == FLAG_STS[0]: # Overt
                        target_prompt_trolling_strategy = random.choice(FLAG_TS[:3])
                    elif current_troll_strategy == FLAG_STS[1]: # Covert
                        target_prompt_trolling_strategy = random.choice(FLAG_TS[3:])
                else: # If no examples for the specified STS type
                    logger.warning(f"Troll example selection: No examples for STS strategy '{current_troll_strategy}'. Substituting with a random example.")
                    if troll_examples: # Random selection from all available examples
                        troll_example_key = random.randint(0, len(troll_examples) - 1)
                        selected_troll_example_data = troll_examples[troll_example_key]

            if selected_troll_example_data is None: # If troll example selection fails
                if troll_examples: # Random selection from all available examples
                    troll_example_key = random.randint(0, len(troll_examples) - 1)
                    selected_troll_example_data = troll_examples[troll_example_key]
                    logger.warning(f"Thread {thread_iter_idx+1}, Profile attempt {profile_iter_idx+1}: Troll example selection strategy {current_troll_strategy} failed. Using random example.")
                else: # If no examples are available at all
                    logger.error(f"Thread {thread_iter_idx+1}, Profile attempt {profile_iter_idx+1}: No troll examples available. Skipping sample generation.")
                    generation_errors += 1
                    continue # Skip current sample generation

            brief_selected_troll_example = {
                'Title': selected_troll_example_data.get('Title', 'No Title'),
                'Post': selected_troll_example_data.get('Post', 'No Post Body'),
                'Comment': selected_troll_example_data.get('Troll', 'No Comment Body'), # Assuming the 'Troll' field in the original data is the example comment
            }

            # Prompt Formatting and LLM Call (similar to single profile version)
            prompt_template_key = troll_provocation_prompt_type
            prompt_template = OmegaConf.select(global_cfg, f"prompts.TrollAgentProvocationPrompt.{prompt_template_key}")
            if not prompt_template:
                logger.error(f"Prompt template '{prompt_template_key}' not found. Skipping sample {thread_iter_idx+1}, {profile_iter_idx+1}.")
                generation_errors += 1
                continue

            # Determine which response format prompt to use
            if troll_provocation_prompt_type == 'vanilla' or troll_provocation_prompt_type == 'in_only':
                current_response_format_prompt = global_cfg.prompts.RESPONSE_FORMAT_PROMPT
                target_prompt_trolling_strategy = "None"
            else:
                current_response_format_prompt = global_cfg.prompts.RESPONSE_FORMAT_PROMPT_WITH_TS

            # Prepare parameters for prompt formatting
            format_params = {
                'user_profile': json.dumps(selected_user_profile),
                'troll_example': json.dumps(brief_selected_troll_example),
                'thread': str(selected_thread_obj),
                'trolling_strategy': target_prompt_trolling_strategy,
                'target_comment': global_cfg.prompts.TrollingProvocationTarget[troll_provocation_target],
                'trolling_strategy_descriptions': TS_GUIDELINE,
                'response_format_prompt': current_response_format_prompt,
            }
            try:
                troll_provocation_message = prompt_template.format(**format_params)
            except KeyError as e:
                logger.error(f"Missing key for prompt formatting: {e}. Prompt type: {prompt_template_key}. Skipping sample {thread_iter_idx}, {profile_iter_idx}.")
                generation_errors += 1
                continue

            data_id = f"{task_cfg.get('name', 'TrollProvocation')}_{thread_iter_idx}_{profile_iter_idx}"
            troll_comment_json_output = None

            if client_type in ['openai', 'deepinfra']:
                troll_comment_json_output = util_generate_openai_model(
                    data_id, client, troll_provocation_message, temperature, max_tokens, model_name, json_output=True)
            elif client_type == 'claude':
                troll_comment_json_output = util_generate_claude_model(
                    data_id, client, troll_provocation_message, temperature, max_tokens, model_name, json_output=True)
            elif client_type == 'vllm':
                troll_comment_json_output = util_generate_vllm_model(
                    data_id, client, troll_provocation_message, temperature, top_p, max_tokens, json_output=True)
            else:
                logger.error(f"Unsupported client type '{client_type}' for sample {thread_iter_idx+1}, {profile_iter_idx+1}.")
                generation_errors += 1
                continue

            if troll_comment_json_output and isinstance(troll_comment_json_output, dict):
                try:
                    troll_thread_obj = deepcopy(selected_thread_obj)
                    troll_thread_obj.comments.append(RedditComment.from_dict(troll_comment_json_output))
                    troll_thread_obj.renew_max_depth()

                    result_entry = {
                        "id": data_id,
                        "user_profile_key": profile_key,
                        "user_profile": selected_user_profile,
                        "thread_key": thread_key,
                        "original_thread": selected_thread_obj.to_dict(),
                        "troll_example_key": troll_example_key,
                        "troll_example": selected_troll_example_data, 
                        "strategy_user_selection": current_user_strategy,
                        "trolling_strategy": target_prompt_trolling_strategy, 
                        "thread_strategy": current_thread_strategy,
                        "troll_strategy": current_troll_strategy,
                        "troll_comment": troll_comment_json_output,
                        "troll_thread": troll_thread_obj.to_dict(),
                    }
                    results_data.append(result_entry)
                    logger.info(f"Thread {thread_iter_idx+1}, Profile {profile_iter_idx+1}/{num_profiles_per_thread} generation complete: Target TS: {target_prompt_trolling_strategy}")
                except Exception as e:
                    logger.error(f"Error processing generated comment for Thread {thread_iter_idx+1}, Profile {profile_iter_idx+1}/{num_profiles_per_thread}: {e}. JSON: {troll_comment_json_output}")
                    generation_errors += 1
            else:
                logger.warning(f"Failed to generate or parse valid JSON for Thread {thread_iter_idx+1}, Profile {profile_iter_idx+1}/{num_profiles_per_thread}. Output: {troll_comment_json_output}")
                generation_errors += 1

    if results_data:
        results_data = sanitize_data(results_data)
        try:
            with open(output_file_json, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x), ensure_ascii=False)
            df = pd.DataFrame(results_data)
            df.to_csv(output_file_tsv, sep='\t', index=False, encoding='utf-8')
            logger.info(f"Generated {len(results_data)} multi-profile troll provocations. Saved to {output_file_tsv} and {output_file_json}")
        except Exception as e:
            logger.error(f"Error saving multi-profile results: {e}. JSON may have been saved at {output_file_json}")
    else:
        logger.info("No multi-profile results generated to save.")

    if client_type == 'vllm' and client is not None:
        del client
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results_data, generation_errors


def run_static_evaluation_task(task_cfg: DictConfig, global_cfg: DictConfig):
    logger.info(f"Starting Static Dataset Evaluation task: {task_cfg.get('dataset_name', 'Unnamed')}")

    dataset_name = task_cfg.dataset_name
    agent_cfg = task_cfg.counter_response_agent
    output_subdir = task_cfg.get('output_subdir', 'elf_countering_static')
    
    counter_client = get_llm_client(agent_cfg.client_type, agent_cfg.model_name, global_cfg)
    eval_client = get_llm_client('openai', 'gpt-4o', global_cfg)

    output_dir = os.path.join(global_cfg.output_base_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    output_file_base = make_elf_countering_filename_convention(output_subdir, agent_cfg.client_type, agent_cfg.short_model_name, agent_cfg.elf_countering_prompt_type, dataset_name)
    output_tsv_path = os.path.join(output_dir, f"{output_file_base}_results.tsv")
    output_json_path = os.path.join(output_dir, f"{output_file_base}_results.json")
    output_eval_path = os.path.join(output_dir, f"{output_file_base}_evaluation.tsv")
    output_eval_json_path = os.path.join(output_dir, f"{output_file_base}_evaluation.json")
    output_eval_table_path = os.path.join(output_dir, f"{output_file_base}_table.csv")


    df = pd.DataFrame()
    troll_col_name = ''
    context_col_name = 'Context'

    if dataset_name == 'counter_trollingy':
        elf22_loader = dataset_dict[dataset_name](data_dir=global_cfg.base_data_dir, task_name=dataset_name)
        df = pd.DataFrame(elf22_loader.datasets['test'])
        troll_col_name = 'Response'
    elif dataset_name == 'ELF-HP':
        df = dataset_dict[dataset_name](data_dir=global_cfg.base_data_dir, task_name=dataset_name).df
        troll_col_name = 'Troll'
    elif dataset_name == 'qian_gab':
        qian_loader = dataset_dict[dataset_name](os.path.join(global_cfg.base_data_dir, "qian_gab/gab.csv"), "qian_gab")
        df = qian_loader.get_dataframe()
        troll_col_name = 'troll_text'
        
    elif dataset_name == 'qian_reddit':
        qian_loader = dataset_dict[dataset_name](os.path.join(global_cfg.base_data_dir, "qian_reddit/reddit.csv"), "qian_reddit")
        df = qian_loader.get_dataframe()
        troll_col_name = 'troll_text'
    else:
        logger.error(f"Unsupported dataset_name for static evaluation: {dataset_name}")
        return

    logger.info(f"Loaded {len(df)} samples from '{dataset_name}'. Troll comment column: '{troll_col_name}'")

    results_to_evaluate = []
    num_to_generate = min(task_cfg.get('max_num_samples', len(df)), len(df))
    
    for i, row in df.head(num_to_generate).iterrows():
        troll_text = row[troll_col_name]
        context_text = row[context_col_name]

        if not troll_text or not isinstance(troll_text, str):
            logger.warning(f"Skipping sample {i} due to empty or invalid troll text.")
            continue

        prompt_template = global_cfg.prompts.ElfAgentCounteringPrompt.vanilla
        format_params = {
            'thread': context_text,
            'trolling_comment': troll_text,
            'response_format_prompt': global_cfg.prompts.RESPONSE_FORMAT_PROMPT
        }
        message = prompt_template.format(**format_params)
        

        counter_json = util_generate_openai_model(
            f"static_{dataset_name}_{i}", counter_client, message, 
            global_cfg.default_generation_params.temperature, 
            global_cfg.default_generation_params.max_tokens, 
            agent_cfg.model_name, json_output=True
        )

        if counter_json and isinstance(counter_json, dict) and 'body' in counter_json:
            result_entry = {
                "id": f"static_{dataset_name}_{i}",
                "original_data": row.to_dict(),
                "troll_comment": {"body": troll_text},
                "elf_comment": counter_json,
                "original_thread": {"post": {"selftext": context_text}, "comments": []},
                "troll_thread": {"post": {"selftext": context_text}, "comments": [{"body": troll_text}]},
                "elf_thread": {"post": {"selftext": context_text}, "comments": [{"body": troll_text}, counter_json]},
            }
            results_to_evaluate.append(result_entry)
            logger.info(f"Generated counter-response for sample {i+1}/{num_to_generate}")
        else:
            logger.warning(f"Failed to generate counter-response for sample {i+1}")

    all_eval_results = {}
    logger.info("Starting evaluations with GPT-4o...")

    logger.info("Evaluating Trolling Strategy (TS) Fidelity...")
    ts_results = evaluate_ts_fidelity(
        troll_provocation_results=results_to_evaluate, client=eval_client, client_type='openai',
        model_name='gpt-4o', subreddit_policy_dict={}, temperature=0.0, max_tokens=100
    )
    all_eval_results.update(ts_results)
    
    logger.info("Evaluating Response Strategy (RS) Fidelity...")
    rs_results = evaluate_rs_fidelity(
        elf_countering_results=results_to_evaluate, client=eval_client, client_type='openai',
        model_name='gpt-4o', subreddit_policy_dict={}, temperature=0.0, max_tokens=100
    )
    all_eval_results.update(rs_results)

    save_evaluation_summary(all_eval_results, output_file=output_eval_path, target='elf')

    with open(output_eval_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_eval_results, f, indent=2, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x))
    logger.info(f"Saved evaluation summary to {output_eval_path}")

    df = pd.DataFrame(results_to_evaluate)
    df.to_csv(output_tsv_path, sep='\t', index=False, encoding='utf-8')
    with open(output_json_path, 'w') as f:
        json.dump(results_to_evaluate, f, indent=2, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x))
    logger.info(f"Saved generated results with annotations to {output_json_path}")


def run_evaluation_task(task_cfg: DictConfig, global_cfg: DictConfig, loaded_data: dict):
    """
    Runs a selective evaluation task based on the provided configuration.

    Args:
        task_cfg (DictConfig): The configuration specific to this evaluation task.
        global_cfg (DictConfig): The global application configuration.
        loaded_data (dict): A dictionary containing pre-loaded data like policies.
    """
    logger.info(f"Starting evaluation task: {task_cfg.get('evaluation_target.output_subdir', 'Unnamed Evaluation')}")

    # 1. Identify and load the target generated data
    eval_target_cfg = task_cfg.get('evaluation_target')
    if not eval_target_cfg:
        logger.error("Evaluation task config must contain 'evaluation_target' section. Skipping.")
        return

    # Construct the filename of the dataset to be evaluated
    target_subdir = eval_target_cfg.get('output_subdir')

    # Determine the file name base based on the target task type
    target_task_type = eval_target_cfg.get('output_subdir', 'troll_provocation')
    
    file_name_base = ""
    if re.findall("troll_provocation", target_task_type):
        file_name_base = make_troll_provocation_filename_convention(eval_target_cfg.output_subdir, eval_target_cfg, eval_target_cfg.troll_provocation_prompt_type, eval_target_cfg.client_type, eval_target_cfg.short_model_name)
    else:
        logger.error(f"Unsupported target task type for evaluation: {target_task_type}")
        return

    target_json_path = os.path.join(global_cfg.output_base_dir, target_subdir, f"{file_name_base}_results.json")
    elf_output_file_tsv = os.path.join(global_cfg.output_base_dir, target_subdir, f"{file_name_base}_results.tsv")
    elf_output_eval_file = os.path.join(global_cfg.output_base_dir, target_subdir, f"{file_name_base}_evaluation.tsv")
    elf_output_eval_json_file = os.path.join(global_cfg.output_base_dir, target_subdir, f"{file_name_base}_evaluation.json")
    elf_output_eval_table_file = os.path.join(global_cfg.output_base_dir, target_subdir, f"{file_name_base}_table.csv")


    if not os.path.exists(target_json_path):
        logger.error(f"Target data file for evaluation not found at: {target_json_path}. Skipping task.")
        return

    try:
        with open(target_json_path, 'r', encoding='utf-8') as f:
            results_to_evaluate = json.load(f)
        logger.info(f"Successfully loaded {len(results_to_evaluate)} results from {target_json_path}")
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load or parse target data file {target_json_path}: {e}. Skipping task.")
        return

    # 2. Setup for metric calculation
    metrics_to_run = task_cfg.get('metrics_to_run', [])
    all_eval_results = {}
    
    # Get just the text of the generated comments for metrics that need it
    generated_texts = []
    target_thread = ''
    target_comment = ''

    generated_texts = [
        res.get('troll_comment', {}).get('body', '')
        for res in results_to_evaluate
        if res.get('troll_comment', {}).get('body')
    ]
    target_thread = 'original_thread'
    target_comment = 'troll_comment'


    # 3. Run selected metrics
    if not metrics_to_run:
        logger.warning("No metrics specified in 'metrics_to_run'. Skipping evaluation.")
        return

    client_dict = {
        'openai': OpenAI(api_key=OPENAI_API_KEY[0]),
        'deepinfra': OpenAI(
            api_key=DEEPINFRA_API_KEY[0],
            base_url="https://api.deepinfra.com/v1/openai",
        ),
        'claude': anthropic.Anthropic(api_key=CLAUDE_API_KEY),
        'vllm': LLM,
    }

    logger.info(f"Running the following metrics: {metrics_to_run}")

    if "refusal_rate" in metrics_to_run:
        total_tried_samples = eval_target_cfg.get('num_samples', global_cfg.default_generation_params.num_samples_troll_provocation_default)
        generation_errors = total_tried_samples - len(results_to_evaluate)
        all_eval_results.update(calculate_refusal_rate(total_tried_samples, generation_errors))
        
    if "language_match" in metrics_to_run:
        target_comment_key = 'troll_comment' if re.findall("troll_provocation", target_task_type) else 'elf_comment'
        target_thread = 'original_thread' if re.findall("troll_provocation", target_task_type) else 'troll_thread'
        all_eval_results.update(analyze_language_match_rate(results_to_evaluate, target_thread=target_thread, target_comment=target_comment_key))

    if "self_bleu" in metrics_to_run:
        all_eval_results.update(calculate_self_bleu(generated_texts))
        all_eval_results.update(calculate_self_bleu_with_tiktoken(generated_texts))

    if "ttr" in metrics_to_run:
        all_eval_results.update(calculate_ttr(generated_texts))
        all_eval_results.update(calculate_mattr(generated_texts))
        all_eval_results.update(calculate_vocab_size(generated_texts))

    if "perplexity" in metrics_to_run:
        logger.info("Running perplexity calculations. This may take a while...")
        # all_eval_results.update(evaluate_perplexity(generated_texts, model_name=global_cfg.default_models.perplexity_eval_model, model_cache_dir=global_cfg.model_cache_dir))
        all_eval_results.update(evaluate_perplexity(results_to_evaluate, target_comment=target_comment, model_name=global_cfg.default_models.perplexity_eval_model, model_cache_dir=global_cfg.model_cache_dir, device_num=global_cfg.default_evaluation_params.perplexity_device_num))
        all_eval_results.update(evaluate_conditional_perplexity(results_to_evaluate, target_thread=target_thread, target_comment=target_comment, model_name=global_cfg.default_models.perplexity_eval_model, model_cache_dir=global_cfg.model_cache_dir, device_num=global_cfg.default_evaluation_params.perplexity_device_num))

    if "trolling_type" in metrics_to_run:
        evaluation_client_type = 'openai'
        evaluation_client = client_dict[evaluation_client_type]
        evaluation_model_name = 'gpt-4o'
        all_eval_results.update(predict_trolling_types(
            troll_provocation_results=results_to_evaluate, 
            client=evaluation_client, 
            client_type=evaluation_client_type, 
            model_name=evaluation_model_name,
            temperature=global_cfg.default_evaluation_params.temperature,
            max_tokens=global_cfg.default_evaluation_params.max_tokens,
            )
        )
    
    if "ts_fidelity" in metrics_to_run and re.findall("troll_provocation", target_task_type):
        evaluation_client_type = 'openai'
        evaluation_client = client_dict[evaluation_client_type]
        evaluation_model_name = 'gpt-4o'
        all_eval_results.update(evaluate_ts_fidelity(
                troll_provocation_results=results_to_evaluate, 
                client=evaluation_client, 
                client_type=evaluation_client_type, 
                model_name=evaluation_model_name,
                subreddit_policy_dict=loaded_data.get('subreddit_policy_dict', {}), 
                temperature=global_cfg.default_evaluation_params.temperature,
                max_tokens=global_cfg.default_evaluation_params.max_tokens,
            )
        )


    if re.findall("troll_provocation", target_task_type):
        is_trolling_openai = [res.get("IsTrolling_gpt-4o") for res in results_to_evaluate if res.get("IsTrolling_gpt-4o")]
        is_trolling_claude = [res.get("IsTrolling_claude-3-5-sonnet-20240620") for res in results_to_evaluate if res.get("IsTrolling_claude-3-5-sonnet-20240620")]
        
        # Get total number of samples that were attempted from the config
        total_tried_samples = eval_target_cfg.get('num_samples', global_cfg.default_generation_params.num_samples_troll_provocation_default)
        if re.findall("troll_provocation_multiprofile", target_task_type):
            total_tried_samples = total_tried_samples * eval_target_cfg.get('num_profiles_per_thread', 1)

        total_generated_samples = len(results_to_evaluate)
        generation_errors = total_tried_samples - total_generated_samples

        if total_tried_samples > 0:
            # Troll Success Rates
            successful_trolls_gpt4o_count = is_trolling_openai.count('Yes')
            all_eval_results['troll_success_rate_gpt-4o'] = (successful_trolls_gpt4o_count / total_generated_samples) * 100 if total_generated_samples > 0 else 0
            
            successful_trolls_claude_count = is_trolling_claude.count('Yes')
            all_eval_results['troll_success_rate_claude'] = (successful_trolls_claude_count / total_generated_samples) * 100 if total_generated_samples > 0 else 0

            non_troll_numerator = total_tried_samples - successful_trolls_gpt4o_count - generation_errors
            all_eval_results['non_troll_rate'] = (non_troll_numerator / total_tried_samples) * 100
            
            # Shannon Entropies
            trolling_strategies_openai = [res.get("EvalTS_gpt-4o") for res in results_to_evaluate if res.get("EvalTS_gpt-4o")]
            trolling_strategies_claude = [res.get("EvalTS_claude-3-5-sonnet-20240620") for res in results_to_evaluate if res.get("EvalTS_claude-3-5-sonnet-20240620")]
            trolling_types_openai = [res.get("predicted_trolling_type_gpt-4o") for res in results_to_evaluate if res.get("predicted_trolling_type_gpt-4o")]
            
            all_eval_results['shannon_entropy_ts_gpt-4o'] = calculate_shannon_entropy(trolling_strategies_openai)
            all_eval_results['shannon_entropy_ts_claude'] = calculate_shannon_entropy(trolling_strategies_claude)
            all_eval_results['shannon_entropy_trolling_type'] = calculate_shannon_entropy(trolling_types_openai)
            

    # 4. Save Evaluation Summary
    summary_target_type = 'troll'
    save_evaluation_summary(all_eval_results, output_file=elf_output_eval_file, target=summary_target_type)

    with open(elf_output_eval_json_file, 'w') as f:
        json.dump(all_eval_results, f, indent=2, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x))
    logging.info(f"Saved evaluation results to {elf_output_eval_json_file}")
    
    df = pd.DataFrame(results_to_evaluate)
    df.to_csv(elf_output_file_tsv, sep='\t', index=False, encoding='utf-8')
    with open(target_json_path, 'w') as f:
        json.dump(results_to_evaluate, f, indent=2, default=lambda x: x.__dict__ if hasattr(x, '__dict__') else str(x))


def run_configured_task(task_cfg: DictConfig, global_cfg: DictConfig, loaded_data: dict):
    is_enabled = task_cfg.get("enabled", False)
    task_type = task_cfg.get("task_type")
    output_subdir = task_cfg.get("output_subdir")
    client_type = task_cfg.get("client_type")
    short_model_name = task_cfg.get("short_model_name")
    troll_provocation_prompt_type = task_cfg.get("troll_provocation_prompt_type")
    thread_data_type = task_cfg.get("thread_data_type")
    is_enabled = task_cfg.get("enabled", False)
    task_name = ''

    # 1. Multiprofile conditions
    num_samples = task_cfg.get("num_samples")
    num_profiles_per_thread = task_cfg.get("num_profiles_per_thread")

    input_details = task_cfg.get("input_troll_provocation_details")

    if num_samples is not None and num_profiles_per_thread is not None:
        base_name_part = f"{output_subdir}_{client_type}_{short_model_name}_{troll_provocation_prompt_type}_{thread_data_type}"
        task_name = f"{base_name_part}_{num_samples}x{num_profiles_per_thread}"
    elif input_details is not None:
        elf_countering_prompt_type = task_cfg.get("elf_countering_prompt_type")
        input_tp_output_subdir = input_details.get("output_subdir")
        input_tp_client_type = input_details.get("client_type")
        input_tp_short_model_name = input_details.get("short_model_name")
        input_tp_prompt_type = input_details.get("troll_provocation_prompt_type")
        input_tp_thread_data_type = input_details.get("thread_data_type")
        
        input_troll_provocation_details_name = f"{input_tp_output_subdir}_{input_tp_client_type}_{input_tp_short_model_name}_{input_tp_prompt_type}_{input_tp_thread_data_type}"       
        task_name = f"{output_subdir}_{client_type}_{short_model_name}_{elf_countering_prompt_type}_from_{input_troll_provocation_details_name}"
    else:
        task_name = f"{output_subdir}_{client_type}_{short_model_name}_{troll_provocation_prompt_type}_{thread_data_type}"

    if not is_enabled:
        logger.info(f"Task '{task_name}' is disabled and will be skipped.")
        return

    logger.info(f"Starting task execution: {task_name} (Type: {task_type})")

    if task_type == "troll_provocation" or task_type == "nontroll_engagement":
        troll_provocation_results, generation_errors = generate_troll_provocation_dataset(
            task_cfg=task_cfg,
            global_cfg=global_cfg,
            userprofiles=loaded_data['userprofiles'],
            threads=loaded_data['threads_map'][task_cfg.get('thread_data_type', 'sit')],
            troll_examples=loaded_data['elf22_examples']
        )
    elif task_type == "troll_provocation_multiprofile" or task_type == "nontroll_engagement_multiprofile":
        troll_provocation_results, generation_errors = generate_multiple_profiles_troll_provocation_dataset(
            task_cfg=task_cfg,
            global_cfg=global_cfg,
            userprofiles=loaded_data['userprofiles'],
            threads=loaded_data['threads_map'][task_cfg.get('thread_data_type', 'sit')],
            troll_examples=loaded_data['elf22_examples']
        )
    elif task_type == "evaluation":
        run_evaluation_task(
            task_cfg=task_cfg,
            global_cfg=global_cfg,
            loaded_data=loaded_data
        )
    elif task_type == "static_evaluation":
        run_static_evaluation_task(
            task_cfg=task_cfg,
            global_cfg=global_cfg,
        )
    else:
        logger.error(f"Unknown task type: {task_type} (Name: {task_name})")


def check_job_status(gen_task_cfg: DictConfig, eval_task_cfg: DictConfig, global_cfg: DictConfig):
    task_type = gen_task_cfg.task_type
    eval_target_cfg = eval_task_cfg.evaluation_target
    output_base_dir = global_cfg.output_base_dir
    file_name_base = ""

    if task_type == "troll_provocation" or task_type == "troll_provocation_multiprofile":
        file_name_base = make_troll_provocation_filename_convention(
            eval_target_cfg.output_subdir, eval_target_cfg, eval_target_cfg.troll_provocation_prompt_type,
            eval_target_cfg.client_type, eval_target_cfg.short_model_name
        )
    else:
        logger.error(f"Unknown task_type for status check: {task_type}")
        return 'ERROR'

    output_dir = os.path.join(output_base_dir, eval_target_cfg.output_subdir)
    results_json_path = os.path.join(output_dir, f"{file_name_base}_results.json")
    evaluation_json_path = os.path.join(output_dir, f"{file_name_base}_evaluation.json")

    if not os.path.exists(results_json_path):
        logger.warning(f"Status Check: Generation result missing -> {results_json_path}")
        return 'NEEDS_GENERATION'

    try:
        with open(results_json_path, 'r', encoding='utf-8') as f:
            if f.read().strip() == "":
                logger.warning(f"Status Check: Generation result is empty -> {results_json_path}")
                return 'NEEDS_GENERATION'
            f.seek(0)
            json.load(f)
    except json.JSONDecodeError:
        logger.warning(f"Status Check: Generation result is a broken JSON -> {results_json_path}")
        return 'NEEDS_GENERATION'
    except Exception as e:
        logger.error(f"Status Check: Error reading file {results_json_path}: {e}")
        return 'NEEDS_GENERATION'

    if not os.path.exists(evaluation_json_path):
        logger.warning(f"Status Check: Evaluation result missing -> {evaluation_json_path}")
        return 'NEEDS_EVALUATION'

    return 'SUCCESS'


def main_with_config(config_path: str):
    # 0. Load configuration file
    try:
        cfg = OmegaConf.load(config_path)
        logger.info(f"Configuration loaded successfully from '{config_path}'.")
        # Additional configuration validation logic can be added here.
    except Exception as e:
        logger.error(f"Failed to load configuration file '{config_path}': {e}")
        return

    # 1. Load Policies (Subreddit Rules)
    subreddit_policy_dict = {}
    try:
        rules_path = OmegaConf.select(cfg, "datasets.subreddit_rules_fname", default=None)
        if rules_path and os.path.exists(rules_path):
            with open(rules_path, 'r', encoding='utf-8') as f:
                subreddit_policy_dict = json.load(f)
            logger.info(f"Subreddit policies loaded successfully: {rules_path}")
        else:
            logger.warning(f"Subreddit rules file path not found or invalid: {rules_path}")
    except Exception as e:
        logger.error(f"Failed to load subreddit policies: {e}")

    # 2. Load User Profiles
    userprofiles = {}
    try:
        profiles_dir = OmegaConf.select(cfg, "datasets.synthetic_userprofiles_dir", default=None)
        if profiles_dir and os.path.isdir(profiles_dir):
            userprofile_fnames = glob(os.path.join(profiles_dir, "*.json"))
            for fname in userprofile_fnames:
                key = os.path.splitext(os.path.basename(fname))[0]
                with open(fname, 'r', encoding='utf-8') as f:
                    userprofiles[key] = json.load(f)
            logger.info(f"{len(userprofiles)} user profiles loaded successfully from: {profiles_dir}")
        else:
            logger.warning(f"User profiles directory path not found or invalid: {profiles_dir}")
    except Exception as e:
        logger.error(f"Failed to load user profiles: {e}")

    # 3. Load Reddit Threads
    # Load various thread types and store them in a dictionary (e.g., 'sit', 'ft', 'set')
    threads_map = {}
    try:
        reddit_threads_loader_key = OmegaConf.select(cfg, "datasets.reddit_threads_task_name", default="reddit_random_threads")
        if reddit_threads_loader_key in dataset_dict:
            thread_dataset_loader = dataset_dict[reddit_threads_loader_key](
                data_dir=cfg.base_data_dir,
                task_name=reddit_threads_loader_key,
                cache_dir=cfg.base_cache_dir
            )
            for thread_type_key in ['sit', 'ft', 'set']: # Thread types to load
                threads_map[thread_type_key] = thread_dataset_loader.thread_dict.get(thread_type_key, [])
                if not threads_map[thread_type_key] and thread_type_key == 'sit': # If 'sit' is empty, replace with default threads
                    threads_map[thread_type_key] = thread_dataset_loader.threads
                logger.info(f"Successfully loaded {len(threads_map[thread_type_key])} threads of type '{thread_type_key}'.")
        else:
            logger.error(f"No loader for '{reddit_threads_loader_key}' in dataset_dict.")
    except Exception as e:
        logger.error(f"Failed to load Reddit threads: {e}")


    # 4. Load ELF22 Dataset (for troll examples)
    elf22_examples = []
    elf22_examples_df = pd.DataFrame()
    try:
        elf22_loader_key = OmegaConf.select(cfg, "datasets.elf22_task_name", default="counter_trollingy")
        if elf22_loader_key in dataset_dict:
            elf22_loader = dataset_dict[elf22_loader_key](
                data_dir=cfg.base_data_dir,
                task_name=elf22_loader_key
            )
            elf22_examples = list(elf22_loader.datasets['train']) # Convert to list
            elf22_examples_df = pd.DataFrame(elf22_examples)
            logger.info(f"Successfully loaded {len(elf22_examples)} troll examples from ELF22 training set.")
        else:
            logger.error(f"No loader for '{elf22_loader_key}' in dataset_dict.")
    except Exception as e:
        logger.error(f"Failed to load ELF22 dataset: {e}")

    # Data set required for task execution
    loaded_data_for_tasks = {
        "subreddit_policy_dict": subreddit_policy_dict,
        "userprofiles": userprofiles,
        "threads_map": threads_map, 
        "elf22_examples": elf22_examples, 
        "elf22_examples_df": elf22_examples_df 
    }

    if re.findall('(Template|template)', cfg.project_name):
        logger.info(f"Use Configuration '{config_path} as template.")

        MODELS_TO_TEST = [
            {'model_name': "meta-llama/Meta-Llama-3.1-70B-Instruct", 'short_model_name': "Meta-Llama-3.1-70B-Instruct", 'client_type': "deepinfra"},
            {'model_name': "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 'short_model_name': "DeepSeek-R1-Distill-Llama-70B", 'client_type': "deepinfra"},
            {'model_name': "gpt-4o", 'short_model_name': "gpt-4o", 'client_type': "openai"},
        ]
        PROMPT_TYPES = ["ours"]
        MAX_RETRIES = 2

        gen_task_template = cfg.tasks[0]
        eval_task_template = cfg.tasks[1]

        permanently_failed_jobs = []
        total_combinations = len(MODELS_TO_TEST) * len(PROMPT_TYPES)
        logger.info(f"--- Starting automated run for {total_combinations} combinations (Max retries: {MAX_RETRIES}) ---")

        for i, model_info in enumerate(MODELS_TO_TEST):
            for j, prompt_type in enumerate(PROMPT_TYPES):
                random.seed(SEED)
                job_key = f"{model_info['short_model_name']}_{prompt_type}"
                current_job_num = i * len(PROMPT_TYPES) + j + 1
                
                logger.info(f"===== Processing Job {current_job_num}/{total_combinations}: {job_key} =====")

                is_job_successful = False
                for attempt in range(MAX_RETRIES + 1):
                    gen_cfg = deepcopy(gen_task_template)
                    gen_cfg.enabled = True
                    gen_cfg.model_name = model_info['model_name']
                    gen_cfg.short_model_name = model_info['short_model_name']
                    gen_cfg.client_type = model_info['client_type']
                    if 'troll_provocation_prompt_type' in gen_cfg:
                        gen_cfg.troll_provocation_prompt_type = prompt_type
                    elif 'elf_countering_prompt_type' in gen_cfg:
                        gen_cfg.elf_countering_prompt_type = prompt_type
                    
                    eval_cfg = deepcopy(eval_task_template)
                    eval_cfg.enabled = True
                    target = eval_cfg.evaluation_target
                    target.model_name = model_info['model_name']
                    target.short_model_name = model_info['short_model_name']
                    target.client_type = model_info['client_type']
                    if 'troll_provocation_prompt_type' in target:
                        target.troll_provocation_prompt_type = prompt_type

                    status = check_job_status(gen_cfg, eval_cfg, cfg)

                    if status == 'SUCCESS':
                        logger.info(f"Job {job_key} is already complete.")
                        is_job_successful = True
                        break

                    logger.warning(f"Job {job_key} status: {status}. Starting attempt {attempt + 1}/{MAX_RETRIES + 1}.")
                    
                    if status == 'NEEDS_GENERATION':
                        logger.info(f"--- Running GENERATION for {job_key} ---")
                        run_configured_task(gen_cfg, cfg, loaded_data_for_tasks)
                        logger.info(f"--- Running EVALUATION for {job_key} after generation ---")
                        run_configured_task(eval_cfg, cfg, loaded_data_for_tasks)
                    
                    elif status == 'NEEDS_EVALUATION':
                        logger.info(f"--- Running EVALUATION ONLY for {job_key} ---")
                        run_configured_task(eval_cfg, cfg, loaded_data_for_tasks)

                if not is_job_successful:
                    final_status = check_job_status(gen_cfg, eval_cfg, cfg)
                    if final_status == 'SUCCESS':
                        logger.info(f"Job {job_key} successfully completed after {attempt + 1} attempts.")
                    else:
                        logger.error(f"Job {job_key} FAILED permanently after all retries.")
                        permanently_failed_jobs.append(job_key)
                
                logger.info(f"===== Finished processing job {job_key} =====\n")

        logger.info("--- Automated run finished. ---")
        if permanently_failed_jobs:
            logger.error("The following jobs failed after all retries:")
            for job in permanently_failed_jobs:
                logger.error(f"  - {job}")
        else:
            logger.info("All jobs completed successfully.")

    else:
        if OmegaConf.is_list(cfg.get('tasks')):
            for task_config_node in cfg.tasks:
                random.seed(SEED)
                run_configured_task(task_config_node, cfg, loaded_data_for_tasks)
        else:
            logger.warning("The list of 'tasks' to execute is not defined in the configuration file.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].endswith(('.yaml', '.yml')):
        config_file_path = sys.argv[1]
        if not os.path.exists(config_file_path):
            logger.error(f"Configuration file not found: {config_file_path}")
            sys.exit(1)
        
        logger.info(f"Loading configuration from command line: {config_file_path}")
        main_with_config(config_file_path)

    else:
        logger.warning("No config file provided via command line. Falling back to default dev config.")
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(script_dir)
        except NameError:
            project_dir = os.getcwd()

        default_config_for_dev = os.path.join(project_dir, "experiments", DEFAULT_CFG_NAME)

        if os.path.exists(default_config_for_dev):
            logger.info(f"Loading default development configuration file: {default_config_for_dev}")
            main_with_config(default_config_for_dev)
        else:
            logger.error(f"Default development configuration file not found: {default_config_for_dev}")


    ########### (example) subreddit_policy_dict ############
    # subreddit_policy_dict = {
    #     "suberddit_name": {
    #         "rules": [
    #             {
    #                 "priority": 1,
    #                 "short_name": "All posts must ...",
    #                 "description": "Humor is subjective, but all ...",
    #             },
    #             ...
    #         ],
    #         "instruction_text": "1. All posts must make ...",
    #         "status": "{success,error,no_rules}",
    #     },
    #     ...
    # }
    ########################################################

    ########### (example) user profiles ############
    # user_profile = {
    #   "basic_profile": {
    #     "username": "GalacticNewbie2021",
    #     "account_age": "2 weeks",
    #     "bio": "I'm a 28-year-old graphic designer from Portland, Oregon, who recently got into the Star Wars universe. I enjoy exploring new fandoms and diving deep into their lore. When I'm not working on design projects, I love hiking in the Pacific Northwest and trying out new coffee blends. I'm usually online in the evenings after work, around 7-10 PM PST. I'm single and enjoy spending my free time learning about new things and connecting with like-minded people online.",
    #     "top_subreddit_categories": "Humor, Technology",
    #     "top_subreddits": "leanfire",
    #     "recent_subreddits": "Traeger, simplerockets"
    #   },
    #   "behavioral_pattern": {
    #     "knowledge_depth": "Basic. I recently started watching the Star Wars movies and series, and I'm slowly getting into the extended universe. Most of my knowledge comes from watching the films and reading fan discussions online.",
    #     "typical_text_length": "1-2 sentences",
    # }
    #############################################

    ############## (example) elf22 examples #############   
    # troll_examples = troll_examples.select(range(80, len(troll_examples)))  # for debugging
    # troll_examples = troll_examples.filter(lambda example: example['TrollL'] == 1) # for only selecting overt trolls
    # troll_examples[0]['Title'] = 'Why no achievements on PC?'
    # troll_examples[0]['Post'] = 'Does anyone else wish there was achievements on PC? Feels like there...s game outside of getting your tryhard on.'
    # troll_examples[0]['Troll'] = 'Have you tried playing for fun?'
    # troll_examples[0]['InputThread']          # We use this as troll_example
    # '{"post": {"subreddit": "Chivalry2", "title": "Why no achievements on PC?", "selftext": "Does anyone else wish there was achievements on PC? Feels like there\'s not really anything to work for in this game outside of getting your tryhard on."}, "comments": [{"body": "Have you tried playing for fun?", "id": "r8bbjmu", "path": ["r8bbjmu"]}]}'
    # troll_examples[0]['ChosenComment']
    # '{"body": "Only lasts for a few hours at most. Game goes beyond stale at about the 40hr mark....", "id": "hwvyu6s", "path": ["r8bbjmu", "hwvyu6s"], "ResponseStrategy": "[engage]"}'
    # troll_examples[0]['ChosenThread']
    # '{"post": {"subreddit": "Chivalry2", "title": "Why no achievements on PC?", "selftext": "Does anyone else wish there was achievements on PC? Feels like there\'s not really anything to work for in this game outside of getting your tryhard on."}, "comments": [{"body": "Have you tried playing for fun?", "id": "r8bbjmu", "path": ["r8bbjmu"]}, {"body": "Only lasts for a few hours at most. Game goes beyond stale at about the 40hr mark....", "id": "hwvyu6s", "path": ["r8bbjmu", "hwvyu6s"], "ResponseStrategy": "[engage]"}]}'
    ######################################################

