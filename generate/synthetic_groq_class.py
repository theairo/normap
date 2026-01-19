import os
import time
import json
import re
import logging
from typing import List, Callable, Any
from tqdm import tqdm
from groq import Groq, RateLimitError, APIError # Requires: pip install groq

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GroqDataFactory:
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        """
        A universal wrapper for generating synthetic data with Groq.
        """
        if not api_key:
            raise ValueError("API Key is required. Set GROQ_API_KEY in your .env")
        
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def _parse_rate_limit(self, error_msg: str) -> float:
        """Extracts wait time from Groq's error message."""
        # Groq often says "Please try again in 4s" or "try again in 1m30s"
        
        # Simple seconds check
        match = re.search(r'try again in (\d+\.?\d*)s', error_msg)
        if match:
            return float(match.group(1)) + 1.0 
        
        # Minutes check
        match_min = re.search(r'try again in (\d+)m(\d+)?s?', error_msg)
        if match_min:
             minutes = float(match_min.group(1))
             seconds = float(match_min.group(2) or 0)
             return (minutes * 60) + seconds + 1.0

        return 60.0  # Default safe wait if parsing fails

    def _fetch_batch_safe(self, batch: List[Any], prompt_builder: Callable, retries: int = 10, plain_inputs_only: bool = False) -> List[dict]:
        """
        Fetches a batch of data with robust Error Handling & JSON cleaning.
        When plain_inputs_only is True, expects newline-separated inputs instead of JSON.
        """
        prompt = prompt_builder(batch)
        
        # Exponential backoff base
        current_sleep = 2.0

        for attempt in range(retries):
            try:
                system_content = "You are a helpful data generation assistant."
                if not plain_inputs_only:
                    system_content += " You MUST output strict JSON only."

                messages=[
                    {
                        "role": "system",
                        "content": system_content,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]

                chat_kwargs = {
                    "messages": messages,
                    "model": self.model_name,
                    "temperature": 0.3,
                }

                if not plain_inputs_only:
                    chat_kwargs["response_format"] = {"type": "json_object"}  # Force JSON mode (Critical for Llama 3)

                chat_completion = self.client.chat.completions.create(**chat_kwargs)

                # Parse Response
                raw_content = chat_completion.choices[0].message.content
                text = raw_content.strip()

                # Clean Markdown fences if present
                if text.startswith("```"):
                    text = re.sub(r"^```\w*\n", "", text)
                    text = re.sub(r"\n```$", "", text)

                if plain_inputs_only:
                    # Split by lines, strip empties
                    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                    return lines

                # Robust JSON Loading
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning(f"JSON Decode Error (Attempt {attempt+1}). Retrying...")
                    continue

                # Llama 3 often wraps array in a root key like {"items": [...]} when in JSON mode
                if isinstance(data, dict):
                    # If it's a dict, look for the first list value
                    found_list = False
                    for key, value in data.items():
                        if isinstance(value, list):
                            data = value
                            found_list = True
                            break
                    
                    # If no list found, treat the dict as a single item if structure matches
                    if not found_list:
                        # Fallback: maybe the prompt forced a single object?
                        # For your prompt requiring an array, this is an edge case.
                        # We wrap it in a list to be safe.
                        data = [data]

                return data

            except Exception as e:
                error_str = str(e)
                
                # Handle Rate Limits (429)
                if "429" in error_str or isinstance(e, RateLimitError):
                    wait_time = self._parse_rate_limit(error_str)
                    
                    # Cap extreme waits (sometimes API says wait 10 mins, better to retry sooner usually)
                    if wait_time > 120: wait_time = 60
                    
                    logger.warning(f"Rate limit hit. Pausing for {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue 

                # Handle Server Errors
                if "500" in error_str or "503" in error_str:
                    logger.warning(f"Groq Server error. Retrying in {current_sleep}s...")
                    time.sleep(current_sleep)
                    current_sleep *= 2
                    continue

                logger.error(f"Critical API Error: {e}")
                return []

        logger.error(f"Failed to fetch batch after {retries} attempts.")
        return []

    def run(self, 
            inputs: List[Any], 
            prompt_builder: Callable[[List[Any]], str], 
            output_file: str, 
            batch_size: int = 3, 
            target_count: int = None,
            sleep_between_batches: float = 1.0,
            inputs_only: bool = False,
            plain_inputs_only: bool = False):
        """
        Generate synthetic data in JSON or plain text formats.

        inputs_only=True saves just the "input" field from each generated item
        line-by-line (plain UTF-8 text) instead of a JSON array of objects.
        plain_inputs_only=True asks the model to emit raw newline-separated inputs
        (no JSON); useful to save tokens. Typically pair with inputs_only=True.
        """
        
        # Load existing
        dataset = []
        if os.path.exists(output_file):
            try:
                if inputs_only:
                    with open(output_file, "r", encoding="utf-8") as f:
                        dataset = [line.strip() for line in f if line.strip()]
                else:
                    with open(output_file, "r", encoding="utf-8") as f:
                        dataset = json.load(f)
                logger.info(f"Resuming from {output_file} ({len(dataset)} samples loaded).")
            except (json.JSONDecodeError, OSError, ValueError):
                logger.warning("Output file corrupted or unreadable. Starting fresh.")

        if target_count and len(dataset) >= target_count:
            logger.info("Target count already reached. Exiting.")
            return

        # Progress bar
        pbar = tqdm(total=target_count or len(inputs)*4, initial=len(dataset), desc="Generating (Groq)")

        for i in range(0, len(inputs), batch_size):
            if target_count and len(dataset) >= target_count:
                break

            batch = inputs[i : i + batch_size]
            if not batch: break

            new_data = self._fetch_batch_safe(batch, prompt_builder, plain_inputs_only=plain_inputs_only)

            if new_data:
                if inputs_only:
                    extracted = []
                    for item in new_data:
                        if isinstance(item, dict) and "input" in item:
                            extracted.append(str(item["input"]))
                        elif isinstance(item, str):
                            extracted.append(item)
                        else:
                            # Skip items without an input field
                            continue

                    if not extracted:
                        logger.warning("Batch returned no extractable 'input' fields. Skipping save for this batch.")
                        continue

                    dataset.extend(extracted)
                    pbar.update(len(extracted))

                    temp_file = output_file + ".tmp"
                    with open(temp_file, "w", encoding="utf-8") as f:
                        f.write("\n".join(dataset) + "\n")
                    os.replace(temp_file, output_file)
                else:
                    dataset.extend(new_data)
                    pbar.update(len(new_data))

                    # Atomic Save
                    temp_file = output_file + ".tmp"
                    with open(temp_file, "w", encoding="utf-8") as f:
                        json.dump(dataset, f, ensure_ascii=False, indent=2)
                    os.replace(temp_file, output_file)
            
            # CRITICAL: Manual throttle for Groq Free Tier
            time.sleep(sleep_between_batches)

        pbar.close()
        logger.info(f"Generation Complete. Saved {len(dataset)} samples to {output_file}")