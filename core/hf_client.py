import aiohttp
from typing import Optional
from config.settings import HUGGINGFACEHUB_API_TOKEN,HF_API_KEY ,FALLBACK_HF_CLIENT


class HuggingFaceClient:
    def __init__(self):
        """
        Initialize Hugging Face client for multimodal models.
        
        Popular free multimodal models:
        - "meta-llama/Llama-3.2-11B-Vision-Instruct" (text + vision)
        - "HuggingFaceM4/idefics2-8b" (text + vision)
        - "microsoft/Phi-3.5-vision-instruct" (text + vision)
        - "Qwen/Qwen2-VL-7B-Instruct" (text + vision)
        
        For text-only RAG, you can use:
        - "mistralai/Mistral-7B-Instruct-v0.3"
        - "meta-llama/Meta-Llama-3-8B-Instruct"
        """
        self.api_key = HUGGINGFACEHUB_API_TOKEN
        self.model = HF_API_KEY
        self.url = f"https://router.huggingface.co/hf-inference/{self.model}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    async def ask_model(self, query: str, context: str, image_url: Optional[str] = None) -> str:
        """
        Sends query + context to Hugging Face model and returns response.
        Supports both text-only and multimodal (text + image) queries.
        
        Args:
            query: User's question
            context: Retrieved context from vector DB
            image_url: Optional image URL for multimodal queries
        """
        # Create prompt based on whether we have an image
        if image_url:
            prompt = f"""Context: {context}

Image URL: {image_url}

Question: {query}

Answer based on the context and image provided."""
        else:
            prompt = f"""Context: {context}

Question: {query}

Answer based on the context only. Be concise and accurate."""

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.url, 
                json=payload, 
                headers=self.headers
            ) as resp:
                if resp.status == 503:
                    # Model is loading
                    return "[Model Loading] The model is currently loading. Please try again in a few moments."
                
                data = await resp.json()
                
                try:
                    print("HF Response:", data)  # Debug print
                    if isinstance(data, list) and len(data) > 0:
                        
                        return data[0].get("generated_text", "").strip()
                    elif isinstance(data, dict):
                        
                        if "error" in data:
                            return f"[HF Error] {data['error']}"
                        return data.get("generated_text", "").strip()
                    return "[Error] Unexpected response format"
                except (KeyError, IndexError) as e:
                    return f"[HF Error] {str(e)}: {data}"


class HuggingFaceClientWithFallback:
    """
    Client with automatic fallback to smaller models if primary fails.
    """
    def __init__(self):
        self.primary = HuggingFaceClient(HF_API_KEY)
        self.fallback = HuggingFaceClient(FALLBACK_HF_CLIENT)
        
    async def ask_model(self, query: str, context: str, image_url: Optional[str] = None) -> str:
        """Try primary model, fallback to smaller model if it fails."""
        result = await self.primary.ask_model(query, context, image_url)
        
        # If primary fails, try fallback (text-only)
        if result.startswith("[") or "error" in result.lower():
            print("Primary model failed, trying fallback...")
            result = await self.fallback.ask_model(query, context, None)
        
        return result