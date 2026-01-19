import os
import time
import json
import re
import logging
from typing import List, Callable, Any
from tqdm import tqdm
from huggingface_hub import InferenceClient, errors  # pip install huggingface_hub

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HuggingFaceDataFactory:
    def __init__(self, api_key: str, model_name: str = "meta-llama/Llama-3.3-70B-Instruct"):
        """
        A robust wrapper for the Hugging Face Serverless Inference API.
        Default Model: Llama-3.3-70B-Instruct (Free on HF).
        """
        if not api_key:
            raise ValueError("API Key is required. Set HF_TOKEN in your .env")
        
        self.client = InferenceClient(api_key=api_key)
        self.model_name = model_name

    def _fetch_batch_safe(self, batch: List[Any], prompt_builder: Callable, retries: int = 5, plain_inputs_only: bool = False) -> List[dict]:
        """
        Fetches a batch with explicit handling for HF's "Model Loading" (503) errors.
        """
        prompt = prompt_builder(batch)
        
        # HF Serverless needs exponential backoff because 70B models take time to load RAM
        wait_time = 5.0 

        for attempt in range(retries):
            try:
                system_content = "You are a helpful data generation assistant."
                if not plain_inputs_only:
                    system_content += " Output strictly valid JSON."

                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ]

                # Call HF Inference API
                # Note: 'response_format' support varies by model on HF. 
                # We rely on prompt engineering + robust parsing instead of forcing API parameters.
                response = self.client.chat_completion(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.6,
                    stream=False
                )

                raw_content = response.choices[0].message.content
                text = raw_content.strip()

                # Clean Markdown
                if text.startswith("```"):
                    text = re.sub(r"^```\w*\n", "", text)
                    text = re.sub(r"\n```$", "", text)

                if plain_inputs_only:
                    return [ln.strip() for ln in text.splitlines() if ln.strip()]

                # Robust JSON Loading
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    # HF models often chat before JSON. Try to find the first '{' and last '}'
                    start = text.find('{')
                    end = text.rfind('}')
                    if start != -1 and end != -1:
                        try:
                            data = json.loads(text[start:end+1])
                        except:
                            logger.warning(f"JSON Parse Failed (Attempt {attempt+1})")
                            continue
                    else:
                        continue

                # Unpack common wrapper keys
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, list):
                            return value
                    # If dict but no list, wrap it
                    return [data]
                
                return data

            except errors.HfHubHTTPError as e:
                # 503 = Model is loading (Cold Boot). This is NORMAL for 70B models.
                if e.response.status_code == 503:
                    logger.warning(f"Model is loading (503). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 1.5, 60) # Cap wait at 60s
                    continue
                
                # 429 = Rate Limit
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("retry-after", 60))
                    logger.warning(f"Rate limit hit (429). Sleeping {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                logger.error(f"HF API Error: {e}")
                time.sleep(5)

            except Exception as e:
                logger.error(f"Unexpected Error: {e}")
                time.sleep(5)

        logger.error(f"Failed to fetch batch after {retries} attempts.")
        return []

    def run(self, 
            inputs: List[Any], 
            prompt_builder: Callable[[List[Any]], str], 
            output_file: str, 
            batch_size: int = 1, # Keep low for HF Free Tier
            target_count: int = None,
            sleep_between_batches: float = 2.0,
            inputs_only: bool = False,
            plain_inputs_only: bool = False):
        
        # Load existing progress
        dataset = []
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    if inputs_only:
                        dataset = [line.strip() for line in f if line.strip()]
                    else:
                        dataset = json.load(f)
                logger.info(f"Resumed {len(dataset)} samples.")
            except:
                logger.warning("Could not read existing file.")

        if target_count and len(dataset) >= target_count:
            return

        pbar = tqdm(total=target_count or len(inputs), initial=len(dataset), desc="Generating (HF)")

        for i in range(0, len(inputs), batch_size):
            if target_count and len(dataset) >= target_count: break

            batch = inputs[i : i + batch_size]
            if not batch: break

            new_data = self._fetch_batch_safe(batch, prompt_builder, plain_inputs_only=plain_inputs_only)

            if new_data:
                # Atomic Append Logic
                if inputs_only:
                    extracted = []
                    for item in new_data:
                        if isinstance(item, dict) and "input" in item:
                            extracted.append(str(item["input"]))
                        elif isinstance(item, str):
                            extracted.append(item)
                    
                    if extracted:
                        dataset.extend(extracted)
                        pbar.update(len(extracted))
                        
                        # Append to file immediately
                        with open(output_file, "a", encoding="utf-8") as f:
                            for line in extracted:
                                f.write(line + "\n")
                else:
                    # JSON Logic
                    dataset.extend(new_data)
                    pbar.update(len(new_data))
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(dataset, f, ensure_ascii=False, indent=2)

            time.sleep(sleep_between_batches)

        pbar.close()