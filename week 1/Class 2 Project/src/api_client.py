"""
api_client.py
-------------
Motive: Simulate an external API with production-grade error handling.
WHY: In AI Engineering, your code will call OpenAI, Hugging Face, or internal ML APIs.
     Networks fail. Rate limits hit. Timeouts occur. Your code must survive all three.
WHAT IT DOES: Simulates API calls with random failures, implements retry logic,
              and handles every common HTTP exception gracefully.
ANALOGY: This is a *skilled courier*. If the road is blocked (network error),
         they try an alternate route (retry). If the package is wrong (400 error),
         they report back immediately instead of crashing.
"""

import time
import random
from typing import Dict, Any, Optional


class SimulatedAPIClient:
    """
    Simulates an external data enrichment API.

    WHY simulated? In a classroom, we cannot depend on a live paid API.
    This client behaves exactly like a real one (delays, failures, JSON responses)
    so students learn error handling without credit card charges.
    """

    def __init__(self, base_url: str = "https://api.example.com", timeout: int = 10):
        self.base_url = base_url
        self.timeout = timeout
        self.call_count = 0

    def enrich_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates sending a record to an API and receiving enriched data.

        WHAT IT DOES:
          1. Increments call counter.
          2. Simulates network delay (0.1s - 0.5s).
          3. Randomly raises exceptions (ConnectionError, Timeout, HTTPError).
          4. Returns the record with an added 'confidence' score.

        WHY random failures? Real APIs fail ~1-5% of the time. Students must learn
        to code defensively, not optimistically.
        """
        self.call_count += 1

        # Simulate network latency.
        # WHY? Real APIs are not instant. Code that assumes instant responses
        # will hang or timeout in production.
        time.sleep(random.uniform(0.1, 0.5))

        # Simulate failures based on call count to ensure students see errors.
        # Every 5th call fails to teach retry logic.
        if self.call_count % 5 == 0:
            raise ConnectionError("Simulated network partition: cannot reach API endpoint.")
        if self.call_count % 7 == 0:
            raise TimeoutError("Simulated request timeout: server took too long.")
        if self.call_count % 9 == 0:
            # Simulate a 500 Internal Server Error from the API.
            raise RuntimeError("HTTP 500: Simulated internal server error.")

        # Enrich the record with a mock confidence score.
        # WHY confidence? In AI, every prediction should come with a certainty metric.
        record["confidence"] = round(random.uniform(0.75, 0.99), 4)
        return record

    def enrich_with_retry(
        self,
        record: Dict[str, Any],
        max_retries: int = 3,
        delay: float = 2.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Wraps enrich_record with exponential backoff retry logic.

        WHY retry? Transient failures (network blips, rate limits) often resolve
        if you wait a moment and try again. Permanent failures (400 Bad Request)
        should not be retried.

        WHAT IT DOES:
          1. Attempts the API call.
          2. If a transient error occurs, waits, then retries.
          3. If all retries exhaust, returns None instead of crashing.

        ANALOGY: Like calling a busy restaurant. If the line is busy (transient),
        you wait 2 minutes and call again. If you dial the wrong number (permanent),
        no amount of redialing helps.
        """
        last_exception = None

        for attempt in range(1, max_retries + 1):
            try:
                return self.enrich_record(record)
            except (ConnectionError, TimeoutError) as e:
                # WHY catch these two? They are TRANSIENT. A network blip or slow
                # server might recover. A 400 Bad Request would NOT be caught here
                # because retrying a malformed request is pointless.
                last_exception = e
                print(f"  [RETRY {attempt}/{max_retries}] {e} — waiting {delay}s...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff: 2s, 4s, 8s.
            except RuntimeError as e:
                # WHY separate catch? 500 errors are server-side. 
                # Some teams retry 500s; others do not. Here we treat them as fatal.
                print(f"  [FATAL] Server error, not retrying: {e}")
                return None

        print(f"  [FAILED] All retries exhausted. Last error: {last_exception}")
        return None
