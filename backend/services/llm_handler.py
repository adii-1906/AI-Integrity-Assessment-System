import urllib.request
import urllib.error
import json
from typing import Dict, Any
from abc import ABC, abstractmethod
from config import config


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


class GeminiProvider(BaseLLMProvider):
    def __init__(self):
        self.api_key = config.GEMINI_API_KEY
        self.model = config.GEMINI_MODEL
        self.api_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("Gemini API key not set.")

        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 500),
            }
        }).encode("utf-8")

        req = urllib.request.Request(
            self.api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                return {"response": text, "model": self.model, "provider": "gemini"}
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(f"Gemini API error {e.code}: {error_body}")
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {str(e)}")


class LocalProvider(BaseLLMProvider):
    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        return {
            "response": "Analysis complete. Review module scores for detailed findings.",
            "model": "template",
            "provider": "local"
        }


class LLMHandler:
    def __init__(self):
        self.providers = []
        gemini = GeminiProvider()
        if gemini.is_available():
            self.providers.append(gemini)
            print(f"  [LLM] Gemini available — using model: {config.GEMINI_MODEL}")
        else:
            print("  [LLM] Gemini API key not set — using template fallback.")
        self.providers.append(LocalProvider())
        self.active_provider = self.providers[0]

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        for provider in self.providers:
            try:
                return provider.generate(prompt, **kwargs)
            except Exception as e:
                print(f"  [LLM] Provider {provider.__class__.__name__} failed: {e}")
                continue
        return {"response": "", "error": "All providers failed"}