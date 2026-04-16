import re
import ast
import json
import math
import time

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_curve, auc, accuracy_score

from transformers import (
    AutoModel,
    AutoConfig,
    EncoderDecoderConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    EncoderDecoderModel,
    PreTrainedTokenizerFast,
)

from vllm import SamplingParams
from collections import Counter
from peft import LoraConfig, TaskType


TS_GUIDELINE = """There are six trolling strategies from overt to covert strategies: 
  Aggression (Engages in direct, promote violence, and unwarranted hostility without any apparent reason), 
  Shocking (exploits sensitive or contentious topics to provoke emotional reaction), 
  Endangering (Pretends to offer help or advice but actually causes harm), 
  Antipathy (Proactively and subtly introduces controversial or provocative topics), 
  Hypocriticism (Targets someone with criticism for a fault or a flaw to undermine the critic's position), 
  Digression (Deviates from the main topic or purpose of the discussion to derail or disrupt the conversation flow).
"""

RS_GUIDELINE = """There are seven response strategies: 
  Engage (sincerely engage with the troll, treating the troll's comment as genuine while subtly addressing the troll's true motives. Generally agree with or accept the troll's opinion.), 
  Ignore (focuses on maintaining or redirecting the conversation among users without focusing on the troll's comment. Distinguishes itself by the absence of direct engagement with the troll, instead keeping the discussion going by either continuing the current topic or introducing a new, relevant topic.), 
  Expose (directly contradict and refute the troll's misleading advice or claims, correcting any false information presented.), 
  Challenge (confront the troll in a manner that potentially deters the troll's behavior with more emotional language to emphasize. Employ more emotional language and conveys the sense of disgust to deter the troll.), 
  Critique (assess the quality and cleverness of the troll's attempt. Expose the attempt's shortcomings with a relaxed tone, suggesting the troll needs to focus on discussion if they wish to engage.), 
  Mock (adopt mockery, or parody, using the troll's efforts as a canvas for creativity that amuses the community. Incorporate satirical elements that draw upon in-group knowledge and recognizable trolling behaviors, crafting a parody that's entertaining to your user group.), 
  Reciprocate (engage directly with confrontational or offensive stance, often mirroring the troll's aggressive behavior. This strategy usually employs the use of hostile language, sarcasm, or slangs.).
"""

CADD_GUIDELINE = """The 'Harmful Strategy' is defined by a combination of three components: [Type] - [Target].

1. Type:
  - Hate Speech: Language that attacks people with a particular identity (e.g., race, gender)
  - Derogatory: Language that attacks a group or an individual but is not based on specific identity attributes.
  - Profanity: Language that contains sexual remarks or slurs (can be targeted or non-targeted).

2. Target:
  - Targeted: The comment is directed at a specific individual or group.
  - Non-targeted: The comment uses harmful language generally without a specific target.
"""

NON_ENGLISH_SUBREDDIT_NAMES = ['ontario', 'ottawa', 'argentina', 'greece', 'China_irl', 'hungary', 'de', 'mexico', 'FragReddit', 'france', 'Austria', 'Finanzen', 'Quebec', 'mauerstrassenwetten', 'chile', 'Monterrey', 'uruguay', 'askspain', 'wien', 'Spielstopp', 'PERU', 'CanadaPublicServants', 'preguntaleareddit', 'montreal', 'Panama', 'GriseldaxFR', 'AskRedditespanol', 'Ticos', 'de_IAmA', 'Argaming', 'MexicoCity', 'MexicoFinanciero', 'rocketbeans', 'paris', 'Colombia', 'geegees', 'BocaJuniors', 'PuertoRico', 'GermanRap', 'einfach_posten', 'tijuana', 'merval', 'AskFrance', 'Paraguay', 'ukraina', 'rbtv_cj', 'Aktien', 'de_EDV', 'hamburg', 'Pikabu', 'Cordoba', 'Gatineau', 'ElSalvador', 'Guadalajara', 'pfennigfuchser', 'bundeswehr', 'DoubanGoosegroup', 'SpainPolitics', 'Republica_Argentina', 'vegetarischDE', 'RepublicadeChile', 'es', 'graz', 'espanol', 'Fahrrad', 'vosfinances', 'VeganDE', 'RepublicaArgentina', 'TroChuyenLinhTinh', 'DerechoGenial', 'bisbille', 'climateskeptics', 'Rosario', 'BUENZLI', 'queretaro', 'germantrees', 'ArAutos', 'etsmtl', 'Dominican', 'kiszamolo', 'arbeitsleben', 'FocusST', 'LegaladviceGerman', 'PietSmiet', 'sqdc', 'YahooQR', 'Mercadoreddit', 'podemos', 'spain', 'PSA', 'preguntaReddit', 'CallofDutyMobileES', 'drogen', 'germantrans', 'RetroBowl', 'ameisenstrassenwetten', 'Lyon', 'valencia', 'greececirclejerk', 'wallstreetbetsGER', 'VtuberV8', 'Kryptostrassenwetten', 'fussball', 'UACommunity', 'ArgEntos', 'Leipzig', 'MAAU', 'DragRace_Canada', 'lehrerzimmer', 'real_China_irl', 'Andalucia', 'beziehungen', 'mannheim', 'Lille', 'rance', 'conseiljuridique', 'DescuentosArgentina', 'Asi_va_Espana', 'Yucatan', 'PCBaumeister', 'Laesterschwestern', 'QuebecFinance', 'BOLIVIA', 'DragRace_Espana', 'orslokx', 'wuerzburg', 'dresden', 'hanguk', 'iwanttorun', 'blaulicht', 'Puebla', 'Muenster', 'ROU', 'FizzMobile', 'Strasbourg', 'bremen', 'toulouse', 'medellin', 'CoronavirusDACH', 'Chinatown_irl', 'SpainFIRE', 'memezuela', 'Epicentr', 'buecher', 'Formel1', 'GME_Mexico', 'Burises', 'programacion', 'BinIchDasArschloch', 'quebeccity', 'Nicaragua', 'QuebecLibre', 'metaquebec', 'aixmarseille', 'Sprechstunde', 'libros', 'wortwitzkasse', 'bielefeld', 'biotechnology', 'augsburg', 'Metallmaimais', 'wasletztepreis', 'QuebecPorn', 'KaIT', 'ani_lgbtq', 'kriptovaluta', 'fiat124', 'egenbogen', 'asozialesnetzwerk', 'Kyiv', 'PhysikBonnMemes', 'Lviv', 'zocken', 'RedditPregunta', 'Nurnberg', 'ParentingFR', 'BahiaBlanca', 'ColombiaReddit', 'Argnime', 'Laval', 'FragNeFrau', 'BajaCalifornia', 'famoseworte', 'spacefrogs', 'AsiaTripper', 'france6', 'DylanteroYT', 'ArgentinaBenderStyle', 'chivas', 'karlsruhe', 'Gronkh', 'recht', 'Wallonia', 'FranceDigeste', 'TfwYouLiveInMexico', 'Steuern', 'RanguGamer', 'montrealhousing', '600euro', 'inversionESP', 'TecDeMonterrey', 'Tenerife', 'Antidigitalisten', 'venezuela', 'PikabuPolitics', 'LoLDE', 'besoindeparler', 'exzj', 'Dachschaden', 'Elektroautos', 'Feminisme', 'Guanajuato', 'XPatriados', 'SonnyLoops', 'maudadomememittwoch', 'Regensburg', 'WriteStreak', 'Bestagons', 'saludmentaluruguay', '2X_INTJ', 'TechoNegro', 'ArgentinaManga', 'iLuTV', 'vozforums', 'Kochen', 'SexualiteFR', 'Mogong', 'brosaemvmeste', 'Jolygolf', 'BiereQc', 'CuartetoDeNos', 'crete', 'ich_iel', 'CadizCF', 'BrainpainPartner', 'WriteStreakGerman', 'KalaRedditFTPCH', 'SpainReps', 'GemischtesHack', 'SchnitzelVerbrechen', 'pajas_mentales', 'Filme', 'jwd', 'InfiniteJest', 'HistoriasdeTerror', 'Miedo', 'depression_de', 'Snacksss', 'CapitolVersicherungAG', 'youngalpha_kingoli', 'notArgentina', 'Dragracelatam', 'filosofia_en_espanol', 'de_netflix', 'Huebi', 'hunnofap', 'GayGermany', 'cosmere_es', 'dearbrother', 'transgenre', 'FormosaProvincia', 'Desahogo', 'Kopiernudeln', 'FestundFlauschig', 'DSA_RPG', 'Stoizismus', 'aweonasogang']

trainerStage2datasetStage = {
    'tr': 'train',
    'val': 'validation',
    'test': 'test',
}


def calculate_metrics(true, pred, labels, task_type='binary'):
    """Calculate comprehensive metrics for different task types"""
    metrics = {
        'accuracy': accuracy_score(true, pred),
        'classification_report': classification_report(true, pred, labels=labels, output_dict=True, zero_division=0)
    }
    
    if task_type == 'binary':
        fpr, tpr, _ = roc_curve(true, pred)
        metrics.update({
            'f1': f1_score(true, pred),
            'precision': precision_score(true, pred),
            'recall': recall_score(true, pred),
            'auc': auc(fpr, tpr)
        })
    else:
        metrics.update({
            'f1_micro': f1_score(true, pred, average='micro', zero_division=0),
            'precision_micro': precision_score(true, pred, average='micro', zero_division=0),
            'recall_micro': recall_score(true, pred, average='micro', zero_division=0),
            'f1_macro': f1_score(true, pred, average='macro', zero_division=0),
            'precision_macro': precision_score(true, pred, average='macro', zero_division=0),
            'recall_macro': recall_score(true, pred, average='macro', zero_division=0),
        })
    
    return metrics


def extract_dictionary_manually(dict_str):
    try:
        # Remove outer braces
        content = dict_str.strip()[1:-1].strip()
        
        result = {}
        # Simple key-value extraction (this is basic and won't handle complex nested structures)
        key_value_pattern = r"'([^']*)':\s*'([^']*)'"
        for match in re.finditer(key_value_pattern, content):
            key, value = match.groups()
            result[key] = value
            
        # Extract list values (basic implementation)
        list_pattern = r"'([^']*)':\s*\[(.*?)\]"
        for match in re.finditer(list_pattern, content):
            key, value_str = match.groups()
            # Split by comma and clean up
            values = [v.strip().strip("'\"") for v in value_str.split(",")]
            result[key] = values
            
        return result
    except Exception as e:
        print(f"Manual extraction failed: {e}")
        return {}


def extract_json_from_text(text):
    "output: user_profile (dict)"
    # Remove markdown formatting if present
    if not text:
        print("No response from model.")
        return None
    response_text = text.strip()
    if not response_text:
        print("No response from model.")
        return None
    if response_text.startswith("```json"):
        # Remove the starting ```
        response_text = re.sub(r"^```json\s*", "", response_text)
        response_text = re.sub(r"\s*```$", "", response_text)

    # Remove doubled braces at start and end if they exist
    response_text = re.sub(r'^\{\{', '{', response_text)
    response_text = re.sub(r'\}\}$', '}', response_text)

    # match = re.search(r'\{.*?\}', text, re.DOTALL)  # Use non-greedy match
    match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', response_text, re.DOTALL)
    if match:
        dict_str = match.group().strip()
        try:
            # First attempt: Clean special characters that cause problems
            cleaned_str = dict_str.replace("'", "'").replace("'", "'").replace(""", "\"").replace(""", "\"")
            return ast.literal_eval(cleaned_str)
        except (SyntaxError, ValueError):
            # Second attempt: Convert to proper JSON format
            try:
                # Replace single quotes with double quotes for keys and string values
                # This is a simplified approach and may not work for all cases
                json_str = re.sub(r"'([^']*)':", r'"\1":', dict_str)  # Replace for keys
                json_str = re.sub(r':\s*\'([^\']*)\'', r': "\1"', json_str)  # Replace for values
                json_str = json_str.replace("'", '"')  # Replace remaining single quotes
                
                # Clean up any special quotes/apostrophes
                json_str = json_str.replace("'", "'").replace("'", "'").replace(""", "\"").replace(""", "\"")
                
                return json.loads(json_str)
            except (json.JSONDecodeError, Exception) as e:
                print(f"All parsing methods failed: {e}")
                # Manual extraction as a last resort
                return extract_dictionary_manually(dict_str)
    
    print(f"Failed to find any dictionary-like object in the response")
    return None


def safe_json_loads(json_string: str):
    """
    Safely loads a JSON string, handling potential errors and LLM-specific formatting issues.

    Args:
        json_string (str): The JSON string to load.

    Returns:
        tuple: A tuple containing the parsed JSON object and an error message (None if successful).
    """
    if not isinstance(json_string, str):
        if isinstance(json_string, dict): # If already a dict (e.g., Claude API response)
            return json_string, None
        try:
            json_string = str(json_string) # Attempt to convert to string
        except Exception as e:
            return None, f"Could not convert input to string: {e}"

    # Clean the string: remove potential markdown/code block fences and surrounding whitespace
    cleaned_string = re.sub(r"^```json\s*", "", json_string.strip(), flags=re.IGNORECASE)
    cleaned_string = re.sub(r"\s*```$", "", cleaned_string)
    cleaned_string = cleaned_string.strip()

    try:
        return json.loads(cleaned_string), None # Attempt direct parsing
    except json.JSONDecodeError as e_initial:
        error_message = f"Initial JSONDecodeError: {e_initial}. String: '{cleaned_string[:200]}...'. "
        # Attempt common fixes
        try:
            # Fix unescaped newlines within strings
            fixed_string = re.sub(r'(?<!\\)\n', r'\\n', cleaned_string)
            # Fix trailing commas
            fixed_string = re.sub(r",\s*([}\]])", r"\1", fixed_string)
            # Replace single quotes around keys/strings with double quotes (Python-style dict handling)
            # This is a simplified approach and may not work for all cases.
            fixed_string = re.sub(r"(?<!\\)'", r'"', fixed_string)

            parsed_json = json.loads(fixed_string)
            # logger.warning(f"JSON parsing required fixes. Original: '{cleaned_string[:100]}...'")
            return parsed_json, None
        except json.JSONDecodeError as e_fixed:
            error_message += f"JSONDecodeError after fixes: {e_fixed}."
            # logger.error(error_message)
            return None, error_message
        except Exception as e_other: # Catch other exceptions during fixing
            error_message += f"Exception during JSON fixing: {e_other}."
            # logger.error(error_message)
            return None, error_message



def sanitize_string(s):
    if not isinstance(s, str):
        return s
    return s.encode('utf-8', 'replace').decode('utf-8')

def sanitize_data(data):
    if isinstance(data, dict):
        return {k: sanitize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    elif isinstance(data, str):
        return sanitize_string(data)
    else:
        return data


def generate_openai_model(data_id, client, messages, temperature, max_tokens, model_name='gpt-4o', json_output=True):
    """
    Example function to call OpenAI's ChatCompletion endpoint.
    This function returns the model's content as a string.
    """
    cnt = 0
    response = None

    while True:
        try:
            cnt += 1
            if cnt == 5:
                break
            if model_name in ['o1-mini']:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": messages}],
                    # temperature=temperature,  # only use for default value 1
                    max_completion_tokens=max_tokens,
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": messages}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            break
        except Exception as e:
            print("Exception:", e)
            print("Id:", data_id)
            time.sleep(10)

    if response is None:
        return None
    
    if json_output:
        return extract_json_from_text(response.choices[0].message.content)
    else:
        return response.choices[0].message.content


def generate_claude_model(data_id, client, messages, temperature, max_tokens, model_name='claude-3-5-sonnet-20240620', json_output=True):   # "claude-3-sonnet-20240229"
    cnt = 0
    response = None
    
    while True:
        try:
            cnt += 1
            if cnt == 5:
                break
            # Claude uses a slightly different API structure from OpenAI
            response = client.messages.create(
                model=model_name,
                messages=[{"role": "user", "content": messages}],
                temperature=temperature,
                max_tokens=max_tokens,
                system=""  # Optional system prompt
            )
            break
        except Exception as e:
            print("Claude API Exception:", e)
            print("Id:", data_id)
            time.sleep(10)

    if response is None:
        return None
    
    # Claude returns response in a slightly different format
    if json_output:
        return extract_json_from_text(response.content[0].text)
    else:
        return response.content[0].text


def generate_vllm_model(data_id, client, messages, temperature, top_p, max_tokens, json_output=True):
    try:
        sampling_params = SamplingParams(
            temperature=temperature, 
            top_p=top_p, 
            max_tokens=max_tokens
        )
        output = client.generate(messages, sampling_params)
        res = output[0].outputs[0].text
        
        if json_output:
            return extract_json_from_text(res)
        else:
            return res
    except Exception as e:
        print("vLLM Exception:", e)
        print("Id:", data_id)
        return None


def calculate_shannon_entropy(data):
    """
    H(X) = - sum(p(x) * log2(p(x))) for all x in X.
    """
    counts = {}
    total_items = 0

    if isinstance(data, list):
        if not data: return 0.0
        counts = Counter(data)
        total_items = len(data)
    elif isinstance(data, dict):
        if not data: return 0.0
        counts = data
        total_items = sum(counts.values())
    else:
        return 0.0

    if total_items == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        probability = count / total_items
        if probability > 0:
            entropy -= probability * math.log2(probability)
            
    return entropy


def main():
    pass

if __name__ == '__main__':
    main()
