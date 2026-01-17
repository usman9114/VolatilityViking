
import time
import logging
import random
import asyncio
from openai import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError

logger = logging.getLogger(__name__)

# Global rate limiter: Max 3 concurrent OpenAI calls to prevent 429s
_openai_semaphore = asyncio.Semaphore(3)

async def call_openai_with_retry(client, model, messages, temperature=None, max_retries=5, initial_delay=2.0):
    """
    Executes an OpenAI ChatCompletion with robust exponential backoff.
    Handles 429 (Rate Limit) specifically.
    Uses global semaphore to limit concurrent calls.
    Note: GPT-5 models don't support custom temperature, pass None to use default.
    """
    if not client:
        return None

    async with _openai_semaphore:  # Limit concurrent calls
        delay = initial_delay
        
        for attempt in range(max_retries + 1):
            try:
                # Run synchronous OpenAI call in thread pool
                loop = asyncio.get_event_loop()
                
                # Build request params
                params = {
                    "model": model,
                    "messages": messages
                }
                # Only add temperature if specified (GPT-5 doesn't support custom temp)
                if temperature is not None:
                    params["temperature"] = temperature
                
                response = await loop.run_in_executor(
                    None,
                    lambda: client.chat.completions.create(**params)
                )
                return response.choices[0].message.content
                
            except RateLimitError as e:
                if attempt == max_retries:
                    logger.error(f"OpenAI 429: Rate limit exceeded after {max_retries} retries.")
                    raise e
                
                # Exponential Backoff with Jitter
                sleep_time = delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"OpenAI 429: Retrying in {sleep_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(sleep_time)
                
            except (APITimeoutError, APIConnectionError, InternalServerError) as e:
                if attempt == max_retries:
                    logger.error(f"OpenAI Network Error: {e}. Giving up.")
                    raise e
                    
                sleep_time = 2.0 * (2 ** attempt)
                logger.warning(f"OpenAI Network Error: {e}. Retrying in {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                # Non-retryable error (e.g. 400 Bad Request)
                logger.error(f"OpenAI Fatal Error: {e}")
                raise e
                
        return None
