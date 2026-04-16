import sys, json
import string
from lingua import LanguageDetectorBuilder, Language
import torch
from tqdm import tqdm 
import re, gc
import numpy as np
import pandas as pd
import nltk
import sacrebleu
import tiktoken
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import logging


from scipy.spatial.distance import jensenshannon

from evaluation_prompts import TROLLING_TYPES, TrollingTypePredictionPrompt, TSFidelityEvaluationPrompt, HarmfulContentDetectionPrompt, HarmfulContentTypePredictionPrompt, HARMFUL_CONTENT_DESCRIPTIONS, HARMFUL_CONTENT_TYPES
from utils import (TS_GUIDELINE, RS_GUIDELINE,
                   generate_openai_model, generate_claude_model,
                   NON_ENGLISH_SUBREDDIT_NAMES, calculate_metrics, safe_json_loads)

from dataset_classes import (
    dataset_dict,
    Thread,
    RedditComment,
    FLAG_STS,
    FLAG_TS,
    FLAG_RS,
    FLAG_SRS
)


def predict_harmful_types(results, client, client_type, model_name, temperature=0.0, max_tokens=50):
    """
    Uses an LLM to predict the harmful type using HarmfulContentTypePredictionPrompt.
    """
    print(f"\n--- Harmful Type Prediction using {model_name} ---")
    predictions = []
    eval_errors = 0

    if not results:
        print("No generation results found to evaluate.")
        return {
            f"harmful_type_counts_{model_name}": {},
            f"evaluation_details_{model_name}": {"notes": "No results found."}
        }

    print(f"Predicting harmful types for {len(results)} valid comments...")
    for result in tqdm(results, desc="Predicting Harmful Types"):
        # Handle Thread Object
        thread_data = result.get('original_thread', {})
        if isinstance(thread_data, str):
            try:
                 thread_obj = json.loads(thread_data)
                 thread_str = str(Thread.from_dict(thread_obj))
            except:
                 thread_str = thread_data
        else:
            thread_str = str(Thread.from_dict(thread_data))

        # Handle Comment Text
        comment_text = result.get('harmful_comment', {}).get('body', '')
        if not comment_text:
             comment_text = result.get('troll_comment', {}).get('body', '')

        # Format Prompt
        prompt = HarmfulContentTypePredictionPrompt.format(
            thread=thread_str,
            comment=comment_text,
            definitions=HARMFUL_CONTENT_DESCRIPTIONS
        )
        
        predicted_type = None
        error_msg = None

        try:
            if client_type in ['openai', 'deepinfra']:
                response = generate_openai_model(
                    data_id=f"eval_type_{result['id']}", client=client, messages=prompt,
                    temperature=temperature, max_tokens=max_tokens, model_name=model_name, json_output=False
                )
                predicted_type = response.strip() if response else "None"
            elif client_type == 'claude':
                 response = generate_claude_model(
                    data_id=f"eval_type_{result['id']}", client=client, messages=prompt,
                    temperature=temperature, max_tokens=max_tokens, model_name=model_name, json_output=False
                )
                 predicted_type = response.strip() if response else "None"
            else:
                error_msg = f"Unsupported client type: {client_type}"

        except Exception as e:
            error_msg = f"Error during harmful type prediction: {e}"
            logging.exception(f"Error for {result['id']}: {e}")

        if error_msg:
            predictions.append({"id": result['id'], f"predicted_harmful_type_{model_name}": None, "error": error_msg})
            eval_errors += 1
        else:
            # Clean up response
            predicted_type = predicted_type.strip().strip('"').strip("'")
            predictions.append({"id": result['id'], f"predicted_harmful_type_{model_name}": predicted_type, "error": None})

    print(f"Finished predicting types. Encountered {eval_errors} errors.")

    # Analysis
    type_counts = Counter(pred[f'predicted_harmful_type_{model_name}'] for pred in predictions)
    print("\nPredicted Harmful Type Distribution:")
    print(type_counts)

    # Attach to results
    for result in results:
        matching_pred = next((p for p in predictions if p['id'] == result['id']), None)
        if matching_pred:
            result[f'predicted_harmful_type_{model_name}'] = matching_pred[f'predicted_harmful_type_{model_name}']

    return {
        f'harmful_type_counts_{model_name}': dict(type_counts),
        f'harmful_types_details_{model_name}': {
            'predictions': predictions,
            'error_count': eval_errors
        }
    }


# IsHarmful
def evaluate_harmfulness_fidelity(results, client, client_type, model_name, temperature=0.0, max_tokens=100):
    """
    Evaluates IsHarmful using HarmfulnessFidelityPrompt.
    """
    print(f"\n--- Harmfulness Fidelity Evaluation using {model_name} ---")
    evaluations = []
    eval_errors = 0

    if not results:
        return {
            f"harmfulness_rate_{model_name}": 0.0,
            f"harmfulness_fidelity_details_{model_name}": {"notes": "No results."}
        }

    print(f"Evaluating harmfulness for {len(results)} valid comments...")
    for result in tqdm(results, desc="Evaluating Harmfulness"):
        # Handle Context
        thread_data = result.get('original_thread', {})
        if isinstance(thread_data, str):
             thread_str = thread_data
        else:
            thread_str = str(Thread.from_dict(thread_data))
        
        # Handle Comment
        comment_text = result.get('harmful_comment', {}).get('body', '')
        if not comment_text:
             comment_text = result.get('troll_comment', {}).get('body', '')

        if not comment_text:
            evaluations.append({"id": result['id'], "IsHarmful": None, "error": "Missing comment body"})
            eval_errors += 1
            continue

        prompt = HarmfulContentDetectionPrompt.format(
            context=thread_str,
            comment=comment_text
        )

        fidelity_result = None
        error_msg = None

        try:
            if client_type in ['openai', 'deepinfra']:
                response_str = generate_openai_model(
                    data_id=f"eval_fid_{result['id']}", client=client, messages=prompt,
                    temperature=temperature, max_tokens=max_tokens, model_name=model_name, json_output=True
                )
            elif client_type == 'claude':
                 response_str = generate_claude_model(
                     data_id=f"eval_fid_{result['id']}", client=client, messages=prompt,
                     temperature=temperature, max_tokens=max_tokens, model_name=model_name, json_output=True
                 )
            else:
                error_msg = f"Unsupported client type: {client_type}"
                response_str = None

            if response_str and not error_msg:
                fidelity_result, parse_error = safe_json_loads(response_str)
                if not fidelity_result or "IsHarmful" not in fidelity_result:
                    error_msg = f"Invalid JSON. Got: {fidelity_result}"
                else:
                    pass 
            elif not error_msg:
                 error_msg = "Empty response."

        except Exception as e:
            error_msg = f"Error: {e}"

        if error_msg:
            evaluations.append({"id": result['id'], "IsHarmful": None, "error": error_msg})
            eval_errors += 1
        else:
            evaluations.append({
                "id": result['id'],
                "IsHarmful": fidelity_result.get("IsHarmful"),
                "error": None
            })

    # Analysis
    eval_df = pd.DataFrame(evaluations)
    isharmful_counts = eval_df['IsHarmful'].value_counts(dropna=False)
    
    harmfulness_rate = (isharmful_counts.get("Yes", 0) / len(results)) * 100 if len(results) > 0 else 0
    print(f"\nHarmfulness Rate: {harmfulness_rate:.2f}%")

    # Attach back
    for result in results:
        matching = next((e for e in evaluations if e['id'] == result['id']), None)
        if matching:
            result[f'IsHarmful_{model_name}'] = matching['IsHarmful']

    return {
        f"harmfulness_rate_{model_name}": harmfulness_rate,
        f"harmfulness_fidelity_details_{model_name}": {
            "distribution": isharmful_counts.to_dict(),
            "error_count": eval_errors,
            "predictions": evaluations
        }
    }



def evaluate_ts_fidelity(troll_provocation_results, client, client_type, model_name, subreddit_policy_dict, temperature=0.0, max_tokens=100, not_include_ts=False):
    """
    Uses an LLM (e.g., GPT-4o) to evaluate if the generated comment is trolling and matches the intended Trolling Strategy (TS).
    """
    print(f"\n--- Trolling Strategy (TS) Fidelity Evaluation using {model_name} ---")
    evaluations = []
    eval_errors = 0

    if not troll_provocation_results:
        print("No provocation generation results found to evaluate.")
        return {
            f"is_trolling_{model_name}": 0.0,
            f"ts_fidelity_{model_name}": 0.0, 
            f"ts_fidelity_details_{model_name}": {
                 "notes": "No provocation generation results found."
            },
        } 
    
    if not_include_ts:
        eval_metrics = {
            f"is_trolling_{model_name}": 0.0,
            f"ts_fidelity_{model_name}": 0.0,
            f"ts_fidelity_details_{model_name}": {
                "note": "not using ts for this troll provocation prompt.",
            }
        }
        return eval_metrics

    print(f"Evaluating TS fidelity for {len(troll_provocation_results)} valid comments...")
    for result in tqdm(troll_provocation_results, desc="Evaluating TS Fidelity"):
        troll_comment = result['troll_comment'].get('body', None) if isinstance(result['troll_comment'], dict) else None
        original_thread_data = result.get('original_thread', None)
        thread_context = Thread.from_dict(original_thread_data)
        subreddit_name = result.get('post', {}).get('subreddit', '')
        subreddit_rules = subreddit_policy_dict.get(subreddit_name, {}).get('instruction_text', 'No policies found')
        intended_ts = 'Unknown' if not_include_ts else result.get('trolling_strategy', 'Unknown') 

        if not troll_comment:
            evaluations.append({"id": result['id'], "IsTrolling": None, "TSFidelity": None, "EvalTS": None, "error": "Missing comment body"})
            eval_errors += 1
            continue

        prompt = TSFidelityEvaluationPrompt.format(
            thread_context=str(thread_context),
            subreddit_rules=subreddit_rules[:1000], # Limit rules length
            intended_ts=intended_ts,
            troll_comment=troll_comment,
            ts_guideline=TS_GUIDELINE
        )

        fidelity_result = None
        error_msg = None

        try:
            # Use appropriate generator function based on client_type
            if client_type in ['openai', 'deepinfra']:
                response_str = generate_openai_model(
                    data_id=f"eval_fid_{result['id']}", client=client, messages=prompt,
                    temperature=temperature, max_tokens=max_tokens, model_name=model_name, json_output=True # Expecting JSON
                )
            elif client_type == 'claude':
                 response_str = generate_claude_model(
                     data_id=f"eval_fid_{result['id']}", client=client, messages=prompt,
                     temperature=temperature, max_tokens=max_tokens, model_name=model_name, json_output=True
                 )
            else:
                error_msg = f"Unsupported client type for evaluation: {client_type}"
                response_str = None

            # logging
            logging.debug(f"Response from TS fidelity evaluator: {response_str}")

            if response_str and not error_msg:
                fidelity_result, parse_error = safe_json_loads(response_str)
                if not fidelity_result:
                    error_msg = f"Failed to parse fidelity JSON: {parse_error}. Raw: '{response_str}'"
                elif not isinstance(fidelity_result, dict) or "TSFidelity" not in fidelity_result or "EvalTS" not in fidelity_result:
                    error_msg = f"Invalid JSON structure from fidelity LLM: {fidelity_result}"
                    fidelity_result = None # Reset on invalid structure

            elif not error_msg:
                 error_msg = "LLM returned empty response for fidelity evaluation."


        except Exception as e:
            error_msg = f"Error during TS fidelity API call: {e}"
            logging.exception(f"Error evaluating fidelity for {result['id']}: {e}")

        if error_msg:
            evaluations.append({"id": result['id'], "IsTrolling": None, "TSFidelity": None, "EvalTS": None, "error": error_msg})
            eval_errors += 1
        else:
            # Ensure EvalTS is one of the known flags or None
            eval_ts = fidelity_result.get("EvalTS")
            istrolling = fidelity_result.get("IsTrolling")
            if eval_ts not in FLAG_TS and eval_ts != "None":
                 logging.warning(f"EvalTS '{eval_ts}' for {result['id']} not in defined FLAG_TS or 'None'. Setting to 'Unknown'.")
                 eval_ts = "Unknown"

            evaluations.append({
                "id": result['id'],
                "IsTrolling": istrolling,
                "TSFidelity": fidelity_result.get("TSFidelity"), # Should be "Yes" or "No"
                "EvalTS": eval_ts,
                "error": None
            })

    print(f"Finished evaluating TS fidelity. Encountered {eval_errors} errors during evaluation.")

    # Analysis of Fidelity
    eval_df = pd.DataFrame(evaluations)
    istrolling_counts = eval_df['IsTrolling'].value_counts(dropna=False)
    fidelity_counts = eval_df['TSFidelity'].value_counts(dropna=False)
    eval_ts_counts = eval_df['EvalTS'].value_counts(dropna=False)

    print("\nThe number of Trolling ")
    print(istrolling_counts)

    print("\nTS Fidelity (Intended vs. Generated):")
    print(fidelity_counts)

    print("\nEvaluated Trolling Strategy Distribution (LLM-eval):")
    print(eval_ts_counts)

    accuracy = (fidelity_counts.get("Yes", 0) / len(troll_provocation_results)) * 100 if len(troll_provocation_results) > 0 else 0
    trolling_accuracy = (istrolling_counts.get("Yes", 0) / len(troll_provocation_results)) * 100 if len(troll_provocation_results) > 0 else 0
    print(f"\nTS Fidelity Accuracy (Intended TS achieved): {accuracy:.2f}%")

    # Add evaluations back to original results
    for result in troll_provocation_results:
        matching_eval = next((e for e in evaluations if e['id'] == result['id']), None)
        if matching_eval:
            result[f'IsTrolling_{model_name}'] = matching_eval['IsTrolling']
            result[f'TSFidelity_{model_name}'] = matching_eval['TSFidelity']
            result[f'EvalTS_{model_name}'] = matching_eval['EvalTS']

    eval_metrics = {
        f"is_trolling_{model_name}": trolling_accuracy,
        f"ts_fidelity_{model_name}": accuracy,
        f"ts_fidelity_details_{model_name}": {
            "fidelity_distribution": fidelity_counts.to_dict(),
            "eval_ts_distribution": eval_ts_counts.to_dict(),
            "eval_errors": eval_errors,
            "predictions": evaluations
        }
    }
    return eval_metrics



def evaluate_perplexity(results, target_comment='troll_comment', model_name="google/gemma-2b", model_cache_dir='', batch_size=8, device_num=0):
    """
    Calculates internal perplexity for a target comment in each result.
    The results are processed in batches for efficiency. The calculated perplexity
    is added back to each result dictionary.

    Args:
        results (list): List of result dictionaries.
        target_comment (str): Key for the target comment object.
        model_name (str): Name of the model for evaluation.
        model_cache_dir (str): Directory to cache the model.
        batch_size (int): Batch size for processing.
        device_num (int): The GPU device number to use.

    Returns:
        dict: A dictionary containing the overall perplexity and detailed results.
    """
    if not results:
        print("No results found to evaluate.")
        return {"internal_perplexity": 0.0, "internal_perplexity_details": {"notes": "Input results list is empty"}}

    try:
        print(f"\n--- Perplexity Evaluation using {model_name} ---")
        device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_cache_dir).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return {"internal_perplexity": 0.0, "internal_perplexity_details": {"notes": f"Model loading failed: {e}"}}

    # Perplexity Calculation
    total_loss = 0
    total_tokens = 0
    individual_perplexities = []
    
    try:
        with torch.no_grad():
            for i in tqdm(range(0, len(results), batch_size), desc="Evaluating Internal Perplexity"):
                batch_results = results[i:i+batch_size]
                
                texts = [res.get(target_comment, {}).get('body', '') for res in batch_results]
                valid_texts = [text for text in texts if text and text.strip()]
                
                if not valid_texts:
                    continue

                # Tokenize batch
                inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
                
                # Calculate loss
                outputs = model(**inputs, labels=inputs.input_ids)
                
                # To calculate perplexity per sentence, we need to un-reduce the loss
                logits = outputs.logits
                labels = inputs.input_ids
                attention_mask = inputs.attention_mask

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_attention_mask = attention_mask[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(shift_labels.size(0), -1)
                
                # Calculate loss and tokens per sentence
                per_sentence_loss = (loss * shift_attention_mask).sum(dim=1)
                per_sentence_tokens = shift_attention_mask.sum(dim=1)
                
                # Update total loss and tokens for overall PPL
                total_loss += per_sentence_loss.sum().item()
                total_tokens += per_sentence_tokens.sum().item()

                # Calculate and assign individual PPL back to results
                res_idx = 0
                for j, text in enumerate(texts):
                    if text and text.strip():
                        # Avoid division by zero
                        if per_sentence_tokens[res_idx].item() > 0:
                            avg_loss = per_sentence_loss[res_idx] / per_sentence_tokens[res_idx]
                            individual_ppl = torch.exp(avg_loss).item()
                            batch_results[j]['internal_perplexity'] = individual_ppl
                        else:
                            batch_results[j]['internal_perplexity'] = None
                        res_idx += 1
                        individual_perplexities.append(individual_ppl)
                    else:
                        batch_results[j]['internal_perplexity'] = None

    except Exception as e:
        print(f"Error during perplexity calculation: {e}")
        # Fallback return
    finally:
        # Memory Cleanup
        model = None
        tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    # Final Metrics Calculation
    if total_tokens > 0:
        overall_avg_loss = total_loss / total_tokens
        overall_perplexity = np.exp(overall_avg_loss)
        print(f"Overall Internal Perplexity: {overall_perplexity:.4f}")
        
        return {
            "internal_perplexity": overall_perplexity,
            "internal_perplexity_details": {
                "individual_perplexities": individual_perplexities,
                "total_tokens": total_tokens,
                "notes": "Calculated using token-weighted average loss."
            }
        }
    else:
        return {
            "internal_perplexity": 0.0,
            "internal_perplexity_details": {"notes": "No valid tokens to evaluate."}
        }


def evaluate_conditional_perplexity(results, target_thread='original_thread', target_comment='troll_comment', model_name="google/gemma-2b", model_cache_dir='', batch_size=8, device_num=0):
    """
    Calculates conditional perplexity of a response given a context for each result.
    The calculation is batched for efficiency, and the result is added back
    to each result dictionary.

    Args:
        results (list): List of result dictionaries.
        target_thread (str): Key for the context thread object.
        target_comment (str): Key for the response comment object.
        model_name (str): Name of the model for evaluation.
        model_cache_dir (str): Directory to cache the model.
        batch_size (int): Batch size for processing. Should be smaller for C-PPL.
        device_num (int): The GPU device number to use.

    Returns:
        dict: A dictionary containing the overall conditional perplexity and detailed results.
    """
    if not results:
        print("No results found to evaluate.")
        return {"conditional_perplexity": 0.0, "conditional_perplexity_details": {"notes": "Input results list is empty"}}

    # Model Loading
    try:
        print(f"\n--- Conditional Perplexity Evaluation using {model_name} ---")
        device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_cache_dir).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return {"conditional_perplexity": 0.0, "conditional_perplexity_details": {"notes": f"Model loading failed: {e}"}}

    # Perplexity Calculation
    total_loss = 0
    total_response_tokens = 0
    individual_perplexities = []
    diversity_scores = {}

    try:
        with torch.no_grad():
            for i in tqdm(range(0, len(results), batch_size), desc="Evaluating Conditional Perplexity"):
                batch_results = results[i:i+batch_size]
                
                # Prepare batch data
                context_texts, response_texts = [], []
                valid_indices = [] # To track which results in the batch are valid
                for idx, res in enumerate(batch_results):
                    # Extract context
                    original_thread = res.get(target_thread, {})
                    post_text = original_thread.get('post', {}).get('selftext', '')
                    comment_bodies = [c.get('body', '') for c in original_thread.get('comments', [])]
                    context = " ".join([post_text] + comment_bodies).strip()
                    
                    # Extract response
                    response = res.get(target_comment, {}).get('body', '')

                    if context and response:
                        context_texts.append(context)
                        response_texts.append(response)
                        valid_indices.append(idx)
                
                if not valid_indices:
                    continue

                # Tokenization
                combined_texts = [c + r for c, r in zip(context_texts, response_texts)]
                
                context_encodings = tokenizer(context_texts, return_tensors='pt', padding=True, truncation=True, max_length=768)
                combined_encodings = tokenizer(combined_texts, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(device)
                
                # Loss Calculation
                outputs = model(**combined_encodings, labels=combined_encodings.input_ids)
                
                logits = outputs.logits
                labels = combined_encodings.input_ids
                
                # Create a mask to calculate loss only on the response part
                loss_mask = torch.ones_like(labels)
                context_lengths = context_encodings.attention_mask.sum(dim=1)
                
                for j, context_len in enumerate(context_lengths):
                    # For each item in the batch, mask out the context tokens
                    loss_mask[j, :context_len] = 0
                
                # Mask out padding tokens as well
                loss_mask[labels == tokenizer.pad_token_id] = 0

                # Calculate Per-Token Loss and Mask
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_mask = loss_mask[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(shift_labels.size(0), -1)
                
                # Calculate loss and tokens for the response part of each sentence
                response_loss = (loss * shift_mask).sum(dim=1)
                response_tokens = shift_mask.sum(dim=1)
                
                # Update total loss and tokens for overall C-PPL
                total_loss += response_loss.sum().item()
                total_response_tokens += response_tokens.sum().item()

                # Assign individual PPL back to results
                for j, original_idx in enumerate(valid_indices):
                    if response_tokens[j].item() > 0:
                        avg_loss = response_loss[j] / response_tokens[j]
                        individual_ppl = torch.exp(avg_loss).item()
                        batch_results[original_idx]['conditional_perplexity'] = individual_ppl
                        individual_perplexities.append(individual_ppl)
                    else:
                        batch_results[original_idx]['conditional_perplexity'] = None

    except Exception as e:
        print(f"Error during conditional perplexity calculation: {e}")
    finally:
        # Memory Cleanup
        model = None
        tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    # Calculate additional diversity metrics
    if len(individual_perplexities) > 1:
        diversity_scores["perplexity_variance"] = np.var(individual_perplexities)
        diversity_scores["perplexity_range"] = max(individual_perplexities) - min(individual_perplexities)
            
    # Final Metrics Calculation
    if total_response_tokens > 0:
        overall_avg_loss = total_loss / total_response_tokens
        overall_perplexity = np.exp(overall_avg_loss)
        print(f"Overall Conditional Perplexity: {overall_perplexity:.4f}")
        
        return {
            "conditional_perplexity": overall_perplexity,
            "conditional_perplexity_details": {
                "individual_perplexities": individual_perplexities,
                "total_tokens": total_response_tokens,
                "diversity_metrics": diversity_scores,
                "notes": "Calculated using conditional perplexity of responses given context."
            }
        }
    else:
        return {
            "conditional_perplexity": 0.0,
            "conditional_perplexity_details": {"notes": "No valid response tokens to evaluate."}
        }


def calculate_self_bleu(texts):
    """
    Calculate Self-BLEU scores using sacrebleu's default tokenizer
    
    Args:
        texts (list): List of text strings
        
    Returns:
        dict: Dictionary with Self-BLEU scores and details
    """
    print("\n--- Self-BLEU Calculation using sacrebleu default tokenizer ---")
    
    if not texts or len(texts) < 2:
        print("Need at least two texts to calculate Self-BLEU.")
        return {
            "self_bleu": 0.0,
            "self_bleu_details": {
                "note": "Not enough texts provided."
            }
        }
    
    # Calculate Self-BLEU scores using sacrebleu
    score_list = []
    pair_count = 0
    
    print(f"Calculating pairwise BLEU for {len(texts)} texts...")
    for i in tqdm(range(len(texts)), desc="Calculating Self-BLEU"):
        hypothesis = [texts[i]] # Single hypothesis
        references = [texts[:i] + texts[i+1:]] # All other texts as references
        
        if references[0]: # Check if we have references
            bleu = sacrebleu.corpus_bleu(
                hypotheses=hypothesis,
                references=references,
                smooth_method='exp', # Exponential smoothing
                smooth_value=0.0,
                force=True, # Force score even with few sentences
                lowercase=True,
                use_effective_order=True # Adaptive n-gram order
            )
            score_list.append(bleu.score)
            pair_count += 1
    
    mean_self_bleu = np.mean(score_list) if score_list else 0
    print(f"Self-BLEU: {mean_self_bleu:.4f}")
    
    # Prepare results
    results = {
        "self_bleu": mean_self_bleu,
        "self_bleu_details": {
            "pair_count": pair_count,
            "tokenizer": "sacrebleu default"
        }
    }
    
    return results


def calculate_ttr(texts):
    """
    Calculate Type-Token Ratio (TTR) for a list of texts.
    TTR is the ratio of unique words to total words, measuring lexical diversity.
    
    Args:
        texts (list): List of text strings
        
    Returns:
        dict: Dictionary with TTR scores and details
    """
    print("\n--- Calculating Type-Token Ratio (TTR) ---")
    
    if not texts:
        print("No texts provided for TTR calculation.")
        return {
        "ttr": {
            "corpus-level": 0.0,
            "mean": 0.0,
            "std": 0.0,
        },
        "ttr_details": {
            "notes": "No texts provided"
        }
    }
        
    # Calculate TTR for each text
    ttrs = []
    total_types = set()  # Unique words across all texts
    total_tokens = 0     # Total words across all texts
    
    print(f"Calculating TTR for {len(texts)} texts...")
    for text in tqdm(texts, desc="Calculating TTR"):
        # Tokenize into words (split on whitespace and remove punctuation)
        words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        
        if words:
            # Calculate unique words (types) and total words (tokens)
            types = len(set(words))
            tokens = len(words)
            ttr = types / tokens if tokens > 0 else 0
            ttrs.append(ttr)
            
            # Update corpus-level statistics
            total_types.update(words)
            total_tokens += tokens
    
    # Calculate statistics
    mean_ttr = np.mean(ttrs) if ttrs else 0
    std_ttr = np.std(ttrs) if ttrs else 0
    corpus_ttr = len(total_types) / total_tokens if total_tokens > 0 else 0
    
    print(f"Mean TTR: {mean_ttr:.4f}")
    print(f"Corpus-level TTR: {corpus_ttr:.4f}")
    
    results = {
        "ttr": {
            "corpus-level": corpus_ttr,
            "mean": mean_ttr,
            "std": std_ttr,
        },
        "ttr_details": {
            "total_unique_words": len(total_types),
            "total_words": total_tokens,
            "num_texts": len(texts)
        }
    }
    
    return results


def calculate_mattr(texts, window_size=40):
    """
    Calculate Moving-Average Type-Token Ratio (MATTR) for a list of texts.
    MATTR addresses the text-length sensitivity of traditional TTR by using a sliding window.
    
    Args:
        texts (list): List of text strings
        window_size (int): Size of the sliding window in words (default: 100)
        
    Returns:
        dict: Dictionary with MATTR scores and details
    """
    print(f"\n--- Calculating Moving-Average TTR (MATTR) with window size {window_size} ---")
    
    if not texts:
        print("No texts provided for MATTR calculation.")
        return {
        "mattr": {
            "corpus-level": 0.0,
            "mean": 0.0,
            "std": 0.0,
        },
        "mattr_details": {
            "notes": "No texts provided"
        }
    }
    
    # Calculate MATTR for each text
    mattrs = []
    word_counts = []
    
    print(f"Calculating MATTR for {len(texts)} texts...")
    for text in tqdm(texts, desc="Calculating MATTR"):
        # Tokenize into words
        words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        word_counts.append(len(words))
        
        # If text is shorter than window size, calculate regular TTR
        if len(words) < window_size:
            if words:
                ttr = len(set(words)) / len(words)
                mattrs.append(ttr)
            continue
            
        # Calculate TTR for each window and take the average
        window_ttrs = []
        for i in range(len(words) - window_size + 1):
            window = words[i:i + window_size]
            window_ttr = len(set(window)) / window_size
            window_ttrs.append(window_ttr)
            
        # Average TTR across all windows
        mattr = np.mean(window_ttrs) if window_ttrs else 0
        mattrs.append(mattr)
    
    # Calculate statistics
    mean_mattr = np.mean(mattrs) if mattrs else 0
    std_mattr = np.std(mattrs) if mattrs else 0
    
    print(f"Mean MATTR: {mean_mattr:.4f}")
    
    results = {
        "mattr": {
            "corpus-level": mean_mattr,
            "mean": mean_mattr,
            "std": std_mattr,
        },
        "mattr_details": {
            "window_size": window_size,
            "num_texts": len(texts),
            "texts_below_window_size": sum(1 for count in word_counts if count < window_size)
        }
    }
    
    return results


def calculate_vocab_size(texts):
    """
    Calculate vocabulary size metrics at sentence and corpus level.
    
    Args:
        texts (list): List of text strings
        
    Returns:
        dict: Dictionary containing vocab size metrics at sentence and corpus levels
    """
    print("\n--- Calculating Vocabulary Size Metrics ---")
    
    if not texts:
        print("No texts provided for vocab size calculation.")
        return {
            "vocab_size": {
                "sentence-level": {"mean": 0, "std": 0},
                "corpus-level": 0
            }
        }

    # Calculate sentence-level vocab sizes
    sentence_vocab_sizes = []
    for text in texts:
        # Split into words and get unique vocab
        words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        vocab = set(words)
        sentence_vocab_sizes.append(len(vocab))
    
    # Calculate sentence-level statistics
    mean_sentence_vocab = np.mean(sentence_vocab_sizes)
    std_sentence_vocab = np.std(sentence_vocab_sizes)
    
    # Calculate corpus-level vocab size
    all_words = []
    for text in texts:
        words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        all_words.extend(words)
    corpus_vocab_size = len(set(all_words))
    
    print(f"Mean sentence-level vocab size: {mean_sentence_vocab:.2f}")
    print(f"Corpus-level vocab size: {corpus_vocab_size}")
    
    return {
        "vocab_size": {
            "sentence-level": {
                "mean": mean_sentence_vocab,
                "std": std_sentence_vocab
            },
            "corpus-level": corpus_vocab_size
        }
    }


def calculate_self_bleu_with_tiktoken(texts, encoding_name="cl100k_base"):
    """
    Calculate Self-BLEU scores using OpenAI's tiktoken
    
    Args:
        texts (list): List of text strings
        encoding_name (str): Name of the encoding to use (e.g., "cl100k_base", "p50k_base")
        use_subwords (bool): Whether to use subword tokens or convert to words
        
    Returns:
        dict: Dictionary with Self-BLEU scores and details
    """
    print(f"\n--- Self-BLEU Calculation using tiktoken ({encoding_name}) ---")
    
    if not texts or len(texts) < 2:
        print("Need at least two texts to calculate Self-BLEU.")
        return {
            "self_bleu_tik": 0.0,
            "self_bleu_tik_details": {
                "note": "Not enough texts provided."
            }
        }
    
    # Load the encoder
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        print(f"Error loading tiktoken encoding: {e}")
        return {"mean_self_bleu": None, "notes": f"Failed to load encoding: {str(e)}"}
    
    # Tokenize texts
    tokenized_texts = []
    token_counts = []
    
    print("Tokenizing texts...")
    for text in tqdm(texts):
        # Tokenize the text
        tokens = encoding.encode(text)
        
        words = []
        token_bytes = [encoding.decode_single_token_bytes(token) for token in tokens]
        current_word = b""
        
        for token_byte in token_bytes:
            # If token starts with a space or is a punctuation byte, it might indicate a word boundary
            if token_byte.startswith(b" ") or token_byte in [b".", b",", b"!", b"?", b":", b";", b")"]:
                if current_word:
                    words.append(current_word.decode("utf-8", errors="replace").strip())
                    current_word = token_byte
                else:
                    current_word += token_byte
            else:
                current_word += token_byte
        
        # Add the last word if there is one
        if current_word:
            words.append(current_word.decode("utf-8", errors="replace").strip())
        
        tokenized_texts.append([word for word in words if word])  # Filter out empty strings
                
        token_counts.append(len(tokenized_texts[-1]))
    
    # Calculate Self-BLEU scores using sacrebleu
    score_list = []
    pair_count = 0
    
    print(f"Calculating pairwise BLEU for {len(texts)} texts...")
    for i in tqdm(range(len(texts)), desc="Calculating Self-BLEU"):
        hypothesis = [tokenized_texts[i]] # Single hypothesis
        references = [tokenized_texts[:i] + tokenized_texts[i+1:]] # List of reference lists
        
        if references[0]: # Check if we have references
            # Calculate BLEU using sacrebleu with smoothing
            # Convert hypothesis from list of tokens to list of joined strings
            hypothesis_str = [" ".join(hypothesis[0])]
            
            # Convert references from list of list of tokens to list of list of joined strings
            references_str = [[" ".join(ref)] for ref in references[0]]
            
            bleu = sacrebleu.corpus_bleu(
                hypotheses=hypothesis_str,
                references=references_str,
                smooth_method='exp', # Exponential smoothing
                smooth_value=0.0,
                force=True, # Force score even with few sentences
                lowercase=True,
                tokenize='none', # Already tokenized
                use_effective_order=True # Adaptive n-gram order
            )
            score_list.append(bleu.score)
            pair_count += 1
    
    mean_self_bleu = np.mean(score_list) if score_list else 0
    print(f"Self-BLEU: {mean_self_bleu:.4f}")
    
    # Prepare results
    results = {
        "self_bleu_tik": mean_self_bleu,
        "self_bleu_tik_details": {
            "pair_count": pair_count,
            "tokenizer": f"tiktoken ({encoding_name})",
            "avg_token_count": np.mean(token_counts),
            "min_token_count": min(token_counts),
            "max_token_count": max(token_counts),
        }
    }
    
    return results


def predict_trolling_types(troll_provocation_results, client, client_type, model_name, temperature=0.0, max_tokens=50):
    """
    Uses an LLM (e.g., GPT-4o) to predict the trolling type for each generated comment.
    """
    print(f"\n--- Trolling Type Prediction using {model_name} ---")
    predictions = []
    eval_errors = 0

    if not troll_provocation_results:
        print("No provocation generation results found to evaluate.")
        return {
            f"trolling_type_counts_{model_name}": {type: 0 for type in TROLLING_TYPES},
            f"evaluation_details_{model_name}": {
                "notes": "No provocation generation results found."
            }
        }

    print(f"Predicting trolling types for {len(troll_provocation_results)} valid comments...")
    for result in tqdm(troll_provocation_results, desc="Predicting Trolling Types"):
        # Convert thread dict to Thread object
        thread = Thread.from_dict(result['original_thread'])
        
        prompt = TrollingTypePredictionPrompt.format(thread=str(thread), troll_comment=result['troll_comment'])
        predicted_type = None
        error_msg = None

        try:
            if client_type in ['openai', 'deepinfra']:
                response = generate_openai_model(
                    data_id=f"eval_type_{result['id']}", client=client, messages=prompt,
                    temperature=temperature, max_tokens=max_tokens, model_name=model_name, json_output=False # Expecting plain text type name
                )
                predicted_type = response.strip() if response else None
            elif client_type == 'claude':
                 response = generate_claude_model(
                    data_id=f"eval_type_{result['id']}", client=client, messages=prompt,
                    temperature=temperature, max_tokens=max_tokens, model_name=model_name, json_output=False
                )
                 predicted_type = response.strip() if response else None
            else:
                error_msg = f"Unsupported client type for evaluation: {client_type}"

            if not predicted_type and not error_msg:
                 error_msg = "LLM returned empty response for type prediction."

        except Exception as e:
            error_msg = f"Error during trolling type prediction API call: {e}"
            logging.exception(f"Error predicting type for {result['id']}: {e}")

        if error_msg:
            predictions.append({"id": result['id'], f"predicted_trolling_type_{model_name}": None, "error": error_msg})
            eval_errors += 1
        else:
            predictions.append({"id": result['id'], f"predicted_trolling_type_{model_name}": predicted_type, "error": None})

    print(f"Finished predicting types. Encountered {eval_errors} errors during evaluation.")

    # Analysis of Predictions
    type_counts = Counter(pred[f'predicted_trolling_type_{model_name}'] for pred in predictions)
    print("\nPredicted Trolling Type Distribution:")
    print(type_counts)

    # Add predictions back to original results
    for result in troll_provocation_results:
        matching_pred = next((p for p in predictions if p['id'] == result['id']), None)
        if matching_pred:
            result[f'predicted_trolling_type_{model_name}'] = matching_pred[f'predicted_trolling_type_{model_name}']

    eval_metrics = {
        f'trolling_type_counts_{model_name}': dict(type_counts),
        f'trolling_types_details_{model_name}': {
            'predictions': predictions,
            'error_count': eval_errors
        }
    }
    return eval_metrics


def calculate_refusal_rate(num_samples, generation_errors):
    """Calculates the refusal rate based on generation errors."""
    print(f"\n--- Refusal Rate Calculation ---")
    if num_samples == 0:
        print("No samples attempted.")
        return {"refusal_rate": None, "notes": "No samples attempted."}

    refusal_rate = (generation_errors / num_samples) * 100
    print(f"Total samples attempted: {num_samples}")
    print(f"Generation errors/refusals: {generation_errors}")
    print(f"Refusal Rate: {refusal_rate:.2f}%")
    return {"refusal_rate": refusal_rate}


def detect_language(text, detector, chars_to_check=1000, default_lang_code="unknown"):
    """
    Detect the language of the given text.
    
    Args:
        text (str): Text to detect language from
        detector: Language detector instance
        chars_to_check (int): Number of characters to check
        default_lang_code (str): Default language code if detection fails
    
    Returns:
        str: Detected language ISO code in lowercase
    """
    if not text or not isinstance(text, str):
        return default_lang_code
    
    # Get a sample of the text for language detection
    text_sample = text[:min(chars_to_check, len(text))]
    
    try:
        detected_language = detector.detect_language_of(text_sample)
        if detected_language:
            # Return language code in lowercase as per ISO standard
            return detected_language.iso_code_639_1.name.lower()
        else:
            return default_lang_code
    except Exception as e:
        print(f"Error detecting language: {e}")
        return "error"
    

def analyze_language_match_rate(results, max_tokens=1000, target_thread='original_thread', target_comment='troll_comment'):
    detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
    
    true_labels = []
    predicted_labels = []
    detailed_results = []
    
    # Process each result
    for result in tqdm(results, desc="Analyzing language match rate"):
        # Extract original thread content
        original_thread = result.get(target_thread)
        troll_comment = result.get(target_comment)
        
        # Extract text from original post and comments
        context_text = []
        
        # Add post text
        if original_thread and original_thread.get('post', {}).get('selftext', ''):
            post_text = original_thread['post']['selftext']
            if post_text and isinstance(post_text, str):
                context_text.append(post_text)
        
        # Add comment texts
        if original_thread and original_thread.get('comments', []):
            for comment in original_thread['comments']:
                if comment.get('body', ''):
                    comment_text = comment['body']
                    if comment_text and isinstance(comment_text, str):
                        context_text.append(comment_text)
    
        # Extract response text
        response_text = troll_comment.get('body', '')
        
        # Skip if no meaningful text to analyze
        if not context_text or not response_text:
            continue
            
        # Combine context texts for language detection
        combined_context = " ".join(context_text)
        
        # Detect languages
        context_lang = detect_language(combined_context, detector, chars_to_check=max_tokens)
        response_lang = detect_language(response_text, detector, chars_to_check=max_tokens)
        
        # Add language info to original result
        result['context_language'] = context_lang
        result['response_language'] = response_lang
        
        # Determine if languages match
        language_match = context_lang == response_lang
        
        # Store results
        true_labels.append(1 if language_match else 0)  # 1 for match, 0 for mismatch
        predicted_labels.append(1)  # Always predicted to match (ideal case)
        
        # Store detailed information
        detailed_results.append({
            'id': result.get('id', ''),
            'context_language': context_lang,
            'response_language': response_lang,
            'language_match': language_match,
            'context_sample': combined_context[:max_tokens] + "..." if len(combined_context) > max_tokens else combined_context,
            'response_sample': response_text[:max_tokens] + "..." if len(response_text) > max_tokens else response_text
        })
    
    # Calculate metrics
    metrics = {}
    language_matches_details = {}
    
    if true_labels:
        metrics = calculate_metrics(true_labels, predicted_labels, labels=[0,1], task_type='binary')
        
        # Count language statistics
        language_counts = Counter([result['context_language'] for result in detailed_results])
        language_match_counts = Counter([result['language_match'] for result in detailed_results])
        
        language_matches_details['language_counts'] = dict(language_counts)
        language_matches_details['language_match_counts'] = dict(language_match_counts)
    
    language_matches_details['predictions'] = detailed_results
    return {
        'language_matches': metrics,
        'language_matches_details': language_matches_details
    }


def display_harmfulness_evaluation_summary(eval_results):
    print("Metric\tValue")
    print("-" * 50)
    # 1. Refusal Rate
    if eval_results.get('refusal_rate') is not None:
        print(f"refusal_rate\t{eval_results['refusal_rate']:.2f}%")
    else:
        print("refusal_rate\tN/A")
    
    # --- Harmfulness Rate ---
    # Check for GPT-4o
    if eval_results.get('harmfulness_rate_gpt-4o') is not None:
        print(f"harmfulness_rate_gpt-4o\t{eval_results['harmfulness_rate_gpt-4o']:.2f}%")
    # Check for Claude or generic
    elif eval_results.get('harmfulness_rate') is not None:
         print(f"harmfulness_rate\t{eval_results['harmfulness_rate']:.2f}%")
    else:
        print("harmfulness_rate\tN/A")

    # --- Shannon Entropies ---
    if eval_results.get('shannon_entropy_harmful_type') is not None:
        print(f"shannon_entropy_harmful_type\t{eval_results['shannon_entropy_harmful_type']:.4f}")
    else:
        print("shannon_entropy_harmful_type\tN/A")

    # 2. Language Match Rate
    language_matches = eval_results.get('language_matches', {})
    if language_matches:
        print(f"language_match_accuracy\t{language_matches.get('accuracy', 'N/A'):.4f}")
        print(f"language_match_f1\t{language_matches.get('f1', 'N/A'):.4f}")
    else:
        print("language_match_accuracy\tN/A")
    
    # 3. Self-BLEU
    if eval_results.get('self_bleu') is not None:
        print(f"self_bleu\t{eval_results['self_bleu']:.4f}")
    else:
        print("self_bleu\tN/A")
    
    # 4. Self-BLEU tik
    if eval_results.get('self_bleu_tik') is not None:
        print(f"self_bleu_tik\t{eval_results['self_bleu_tik']:.4f}")
    else:
        print("self_bleu_tik\tN/A")
    
    # 4. Harmful Type Prediction
    # Check keys starting with harmful_type_counts
    type_keys = [k for k in eval_results.keys() if k.startswith('harmful_type_counts')]
    if type_keys:
        for key in type_keys:
            print(f"--- {key} ---")
            for h_type, count in eval_results[key].items():
                print(f"type_{h_type}\t{count}")
    else:
        print("harmful_type_prediction\tN/A")
    
    # 5. Harmfulness Fidelity Details
    fid_keys = [k for k in eval_results.keys() if k.startswith('harmfulness_fidelity_details')]
    if fid_keys:
        for key in fid_keys:
            dist = eval_results[key].get('distribution', {})
            for resp, count in dist.items():
                print(f"harmfulness_eval_{resp}\t{count}")

    # 6. Perplexity Metrics
    if 'internal_perplexity' in eval_results and eval_results['internal_perplexity']:
        print(f"internal_perplexity\t{eval_results['internal_perplexity']:.4f}")
    else:
        print("internal_perplexity\tN/A")
        
    if 'conditional_perplexity' in eval_results and eval_results['conditional_perplexity']:
        print(f"conditional_perplexity\t{eval_results['conditional_perplexity']:.4f}")
    else:
        print("conditional_perplexity\tN/A")
        
    # 9. Type-Token Ratio (TTR)
    if 'ttr' in eval_results:
        print(f"ttr\t{eval_results['ttr']['corpus-level']:.4f}")
        print(f"ttr_mean\t{eval_results['ttr']['mean']:.4f}")
        print(f"ttr_std\t{eval_results['ttr']['std']:.4f}")
    else:
        print("ttr\tN/A")
        
    # 10. Moving Average Type-Token Ratio (MATTR)
    if 'mattr' in eval_results:
        print(f"mattr\t{eval_results['mattr']['corpus-level']:.4f}")
        print(f"mattr_mean\t{eval_results['mattr']['mean']:.4f}")
        print(f"mattr_std\t{eval_results['mattr']['std']:.4f}")
    else:
        print("mattr\tN/A")
        
    # 11. Vocabulary Size
    if 'vocab_size' in eval_results:
        print(f"vocab_size_sentence_mean\t{eval_results['vocab_size']['sentence-level']['mean']:.4f}")
        print(f"vocab_size_sentence_std\t{eval_results['vocab_size']['sentence-level']['std']:.4f}")
        print(f"vocab_size_corpus\t{eval_results['vocab_size']['corpus-level']}")
    else:
        print("vocab_size\tN/A")



def display_evaluation_summary(eval_results):
    print("Metric\tValue")
    print("-" * 50)
    
    # 1. Refusal Rate
    if eval_results.get('refusal_rate') is not None:
        print(f"refusal_rate\t{eval_results['refusal_rate']:.2f}%")
    else:
        print("refusal_rate\tN/A")
    
    # --- Non-Troll Rate and Troll Success Rate ---
    if eval_results.get('non_troll_rate') is not None:
        print(f"non_troll_rate\t{eval_results['non_troll_rate']:.2f}%")
    else:
        print("non_troll_rate\tN/A")

    if eval_results.get('troll_success_rate_gpt-4o') is not None:
        print(f"troll_success_rate_gpt-4o\t{eval_results['troll_success_rate_gpt-4o']:.2f}%")
    else:
        print("troll_success_rate_gpt-4o\tN/A")

    if eval_results.get('troll_success_rate_claude') is not None:
        print(f"troll_success_rate_claude\t{eval_results['troll_success_rate_claude']:.2f}%")
    else:
        print("troll_success_rate_claude\tN/A")

    # --- Shannon Entropies ---
    if eval_results.get('shannon_entropy_ts_gpt-4o') is not None:
        print(f"shannon_entropy_ts_gpt-4o\t{eval_results['shannon_entropy_ts_gpt-4o']:.4f}")
    else:
        print("shannon_entropy_ts_gpt-4o\tN/A")

    if eval_results.get('shannon_entropy_ts_claude') is not None:
        print(f"shannon_entropy_ts_claude\t{eval_results['shannon_entropy_ts_claude']:.4f}")
    else:
        print("shannon_entropy_ts_claude\tN/A")
    
    if eval_results.get('shannon_entropy_trolling_type') is not None:
        print(f"shannon_entropy_trolling_type\t{eval_results['shannon_entropy_trolling_type']:.4f}")
    else:
        print("shannon_entropy_trolling_type\tN/A")

    # 2. Language Match Rate
    language_matches = eval_results.get('language_matches', {})
    if language_matches:
        print(f"language_match_accuracy\t{language_matches.get('accuracy', 'N/A'):.4f}")
        print(f"language_match_f1\t{language_matches.get('f1', 'N/A'):.4f}")
        print(f"language_match_auc\t{language_matches.get('auc', 'N/A'):.4f}")
        
        # Display language counts as separate entries
        language_matches_details = eval_results.get('language_matches_details', {})
        if 'language_counts' in language_matches_details:
            for lang, count in language_matches_details['language_counts'].items():
                print(f"language_count_{lang}\t{count}")
        
        # Display language match counts
        if 'language_match_counts' in language_matches_details:
            for match_type, count in language_matches_details['language_match_counts'].items():
                print(f"language_match_{match_type}\t{count}")
    else:
        print("language_match_accuracy\tN/A")
        print("language_match_f1\tN/A")
        print("language_match_auc\tN/A")
    
    # 3. Self-BLEU
    if eval_results.get('self_bleu') is not None:
        print(f"self_bleu\t{eval_results['self_bleu']:.4f}")
    else:
        print("self_bleu\tN/A")
    
    # 4. Self-BLEU tik
    if eval_results.get('self_bleu_tik') is not None:
        print(f"self_bleu_tik\t{eval_results['self_bleu_tik']:.4f}")
    else:
        print("self_bleu_tik\tN/A")
    
    # 5. Trolling Type Prediction
    if 'trolling_type_counts' in eval_results:
        for troll_type, count in eval_results['trolling_type_counts'].items():
            print(f"trolling_type_{troll_type}\t{count}")
    else:
        print("trolling_type_prediction\tN/A")
    
    # 6. Trolling Strategy Fidelity
    # Find all ts_fidelity metrics for different models
    ts_fidelity_metrics = {k:v for k,v in eval_results.items() if k.startswith('ts_fidelity_')}
    
    if ts_fidelity_metrics:
        for metric_name, value in ts_fidelity_metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric_name}\t{value:.2f}%")
            elif isinstance(value, dict) and 'fidelity_distribution' in value:
                # Print detailed distributions
                for response, count in value['fidelity_distribution'].items():
                    print(f"{metric_name}_fidelity_{response}\t{count}")
                
                if 'eval_ts_distribution' in value:
                    for ts_type, count in value['eval_ts_distribution'].items():
                        ts_type_safe = ts_type.replace(" ", "_").lower() if ts_type else "none"
                        print(f"{metric_name}_eval_ts_{ts_type_safe}\t{count}")
    else:
        print("ts_fidelity\tN/A")
    
    # 7. Perplexity Metrics
    # Internal perplexity
    if 'internal_perplexity' in eval_results and eval_results['internal_perplexity']:
        print(f"internal_perplexity\t{eval_results['internal_perplexity']:.4f}")
    else:
        print("internal_perplexity\tN/A")
        
    # 8. Conditional perplexity
    if 'conditional_perplexity' in eval_results and eval_results['conditional_perplexity']:
        print(f"conditional_perplexity\t{eval_results['conditional_perplexity']:.4f}")
    else:
        print("conditional_perplexity\tN/A")
    # 9. Type-Token Ratio (TTR)
    if 'ttr' in eval_results:
        print(f"ttr\t{eval_results['ttr']['corpus-level']:.4f}")
        print(f"ttr_mean\t{eval_results['ttr']['mean']:.4f}")
        print(f"ttr_std\t{eval_results['ttr']['std']:.4f}")
    else:
        print("ttr\tN/A")
        
    # 10. Moving Average Type-Token Ratio (MATTR)
    if 'mattr' in eval_results:
        print(f"mattr\t{eval_results['mattr']['corpus-level']:.4f}")
        print(f"mattr_mean\t{eval_results['mattr']['mean']:.4f}")
        print(f"mattr_std\t{eval_results['mattr']['std']:.4f}")
    else:
        print("mattr\tN/A")
        
    # 11. Vocabulary Size
    if 'vocab_size' in eval_results:
        print(f"vocab_size_sentence_mean\t{eval_results['vocab_size']['sentence-level']['mean']:.4f}")
        print(f"vocab_size_sentence_std\t{eval_results['vocab_size']['sentence-level']['std']:.4f}")
        print(f"vocab_size_corpus\t{eval_results['vocab_size']['corpus-level']}")
    else:
        print("vocab_size\tN/A")


def save_evaluation_summary(eval_results, output_file="evaluation_summary.tsv", target='troll'):
    """Save the evaluation summary to a tab-separated file"""
    original_stdout = sys.stdout
    with open(output_file, 'w') as f:
        sys.stdout = f
        if target == 'troll':
            display_evaluation_summary(eval_results)
        elif target == 'harmfulness':
            display_harmfulness_evaluation_summary(eval_results)
        sys.stdout = original_stdout
    print(f"Evaluation summary saved to {output_file}")

