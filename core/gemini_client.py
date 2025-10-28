"""
core/gemini_client.py
Handles Gemini model connection and text-based reasoning.
"""

import aiohttp
from config.settings import GEMINI_API_KEY, GEMINI_MODEL


class GeminiClient:
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.model = GEMINI_MODEL
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

    async def ask_gemini(self, query: str, context: str) -> str:
        """
        Sends query + context to Gemini API and returns response text.
        """
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer based on the context only."}
                    ]
                }
            ]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=payload) as resp:
                data = await resp.json()
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
                except KeyError:
                    return f"[Gemini Error] {data}"
