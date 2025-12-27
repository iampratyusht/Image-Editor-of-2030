"""
Agent Decision Script for Image Editing Tasks
This script uses Mistral AI to analyze user prompts and extract
structured information for image editing tasks (remove, replace, background_fill).

Extracts:
- task_type: remove, replace, background_fill
- clip_prompt: Object description for CLIP-based segmentation
- diffusion_prompt: Positive prompt for Stable Diffusion
- negative_diffusion_prompt: Negative prompt for Stable Diffusion
"""

import os
import json
import argparse
from typing import Optional, Literal

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Try to import API key from config file
try:
    from .config import MISTRAL_API_KEY as CONFIG_MISTRAL_KEY
except ImportError:
    CONFIG_MISTRAL_KEY = None


# --- 1. Define Schema for Full Task Extraction ---
class AgentDecision(BaseModel):
    """Schema for structured output from the LLM - Full task extraction."""
    task_type: Literal["remove", "replace", "background_fill"] = Field(
        ..., description="Task category: 'remove' to delete object, 'replace' to swap object with new content, 'background_fill' to change/fill background while keeping object."
    )
    clip_prompt: Optional[str] = Field(
        None, description="Visual description of the object for CLIP-based segmentation. Include spatial location if mentioned. For remove/replace: object to find. For background_fill: object to KEEP."
    )
    diffusion_prompt: Optional[str] = Field(
        None, description="Positive prompt for Stable Diffusion. For 'replace': detailed description of new object. For 'background_fill': detailed description of new background scene. For 'remove': null."
    )
    negative_diffusion_prompt: Optional[str] = Field(
        None, description="Negative prompt for Stable Diffusion - what to avoid. For 'remove': null. For replace/background_fill: things to avoid like 'blurry, low quality, distorted'."
    )


# --- 2. Define Schema for Diffusion-Only Extraction (Manual Mode) ---
class DiffusionPromptExtraction(BaseModel):
    """Schema for extracting only diffusion prompts (used in manual mode after mask selection)."""
    diffusion_prompt: str = Field(
        ..., description="Detailed positive prompt for Stable Diffusion describing what to generate."
    )
    negative_diffusion_prompt: str = Field(
        default="blurry, low quality, distorted, deformed, ugly, bad anatomy",
        description="Negative prompt for Stable Diffusion - what to avoid in generation."
    )


def check_api_key():
    """Verify that the Mistral API key is available (config file or environment)."""
    # First check config file
    if CONFIG_MISTRAL_KEY and CONFIG_MISTRAL_KEY != "your_mistral_api_key_here":
        os.environ["MISTRAL_API_KEY"] = CONFIG_MISTRAL_KEY
        return CONFIG_MISTRAL_KEY
    
    # Then check environment variable
    api_key = os.environ.get("MISTRAL_API_KEY")
    if api_key:
        return api_key
    
    raise ValueError(
        "MISTRAL_API_KEY not found.\n"
        "Please set it in one of these ways:\n"
        "  1. Edit config.py and set MISTRAL_API_KEY = 'your_key'\n"
        "  2. Set environment variable:\n"
        "     Windows PowerShell: $env:MISTRAL_API_KEY = 'your_api_key'\n"
        "     Windows CMD: set MISTRAL_API_KEY=your_api_key\n"
        "     Linux/Mac: export MISTRAL_API_KEY=your_api_key\n\n"
        "Get your API key from: https://console.mistral.ai/"
    )


def list_available_models():
    """List available Mistral models."""
    print("--- Available Mistral Models ---")
    print("  - mistral-tiny (fastest, cheapest)")
    print("  - mistral-small (good balance)")
    print("  - mistral-medium (better quality)")
    print("  - mistral-large-latest (best quality)")
    print("  - open-mistral-7b (open source)")
    print("  - open-mixtral-8x7b (open source)")
    print("  - open-mixtral-8x22b (open source, largest)")


def create_agent_chain(model_name: str = "mistral-small-latest"):
    """Create and return the LangChain agent chain for full task extraction."""
    # Initialize Mistral Model
    llm = ChatMistralAI(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(AgentDecision)
    
    # Define the system prompt
    system_prompt = """You are an AI assistant that analyzes image editing requests and extracts structured information.

Your job is to determine the task type and extract ALL relevant details from user prompts.

## Task Types:
1. **remove** - User wants to DELETE/REMOVE an object from the image
   - Keywords: remove, delete, erase, get rid of, take out, clear
   - Example: "remove the car", "delete the person on the left"
   - NO diffusion prompts needed (uses inpainting model)

2. **replace** - User wants to SWAP/REPLACE an object with something NEW
   - Keywords: replace, change, turn into, make it, swap, convert to
   - Example: "replace the dog with a cat", "change the car to a red sports car"
   - Needs diffusion prompts for generation

3. **background_fill** - User wants to CHANGE THE BACKGROUND while KEEPING a specific object
   - Keywords: change background, new background, put in, place in, different scene
   - Example: "change the background to a beach", "put the person in a forest"
   - Needs diffusion prompts for background generation

## Extraction Rules:

### clip_prompt (for CLIP-based segmentation):
- Visual description of the object to find/segment
- Include spatial location if mentioned (e.g., "cat on the left", "person in center")
- For remove/replace: describe the object to remove/replace
- For background_fill: describe the object to KEEP (foreground subject)

### diffusion_prompt (positive prompt for Stable Diffusion):
- For 'remove': Set to null (no generation needed)
- For 'replace': Detailed description of the NEW object to generate
  - Add visual details: colors, textures, style
  - Add quality terms: "highly detailed", "realistic", "professional"
- For 'background_fill': Detailed description of the NEW background scene
  - Describe the environment, lighting, atmosphere

### negative_diffusion_prompt (what to avoid):
- For 'remove': Set to null
- For 'replace' and 'background_fill': Include quality issues to avoid
  - Default: "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark"
  - Add task-specific negatives if implied by user

## Examples:

1. "remove the tree on the left"
   → task: remove, clip: "tree on the left", diffusion: null, negative: null

2. "replace the dog with a fluffy orange cat"
   → task: replace, clip: "dog", diffusion: "fluffy orange cat, cute, highly detailed, soft fur, realistic", negative: "blurry, low quality, distorted, deformed"

3. "change the background to a sunset beach"
   → task: background_fill, clip: "main subject" (or specific if mentioned), diffusion: "beautiful sunset beach, golden hour lighting, ocean waves, palm trees, warm colors, highly detailed, professional photography", negative: "blurry, low quality, distorted, overexposed"

4. "put the girl in a snowy mountain scene"
   → task: background_fill, clip: "girl", diffusion: "snowy mountain landscape, snow-capped peaks, winter atmosphere, cold blue tones, dramatic scenery, highly detailed", negative: "blurry, low quality, distorted, summer elements"
"""
    
    # Create prompt template and chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    agent_chain = prompt | structured_llm
    
    return agent_chain


def create_diffusion_prompt_chain(model_name: str = "mistral-small-latest"):
    """Create a chain for extracting diffusion prompts only (used in manual mode)."""
    llm = ChatMistralAI(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(DiffusionPromptExtraction)
    
    system_prompt = """You are an expert at crafting detailed prompts for Stable Diffusion image generation.

Given a user's description of what they want to generate, extract:
1. **diffusion_prompt**: A detailed, high-quality positive prompt
   - Add visual details: textures, colors, lighting, materials
   - Add quality modifiers: "highly detailed", "8k", "professional"
   - Keep the core subject/concept from user's input
   
2. **negative_diffusion_prompt**: What to avoid in generation
   - Include quality issues: "blurry, low quality, distorted, deformed"
   - Add any specific things to avoid based on context

## Examples:

User: "a cute puppy"
→ diffusion: "adorable puppy with fluffy fur, cute expression, soft natural lighting, highly detailed, professional pet photography, 8k"
→ negative: "blurry, low quality, distorted, deformed, ugly, bad anatomy"

User: "a futuristic city"
→ diffusion: "futuristic cityscape, neon lights, flying cars, tall skyscrapers, cyberpunk atmosphere, highly detailed, cinematic lighting, 8k"
→ negative: "blurry, low quality, distorted, old buildings, medieval, fantasy"

User: "beautiful forest"
→ diffusion: "lush green forest, sunlight filtering through trees, peaceful atmosphere, detailed foliage, morning mist, professional nature photography, 8k"
→ negative: "blurry, low quality, distorted, dead trees, urban elements"
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Extract diffusion prompts for: {input}")
    ])
    
    return prompt | structured_llm


def process_user_prompt(agent_chain, user_input: str) -> dict:
    """
    Process a user prompt and return structured output with all extracted fields.
    
    Args:
        agent_chain: The LangChain agent chain
        user_input: The user's natural language request
        
    Returns:
        Dictionary containing:
        - task: "remove", "replace", or "background_fill"
        - clip_prompt: Object description for CLIP segmentation
        - diffusion_prompt: Positive prompt for SD (None for remove)
        - negative_diffusion_prompt: Negative prompt for SD (None for remove)
    """
    try:
        decision = agent_chain.invoke({"input": user_input})
    except Exception as e:
        return {"error": str(e)}

    return {
        "task": decision.task_type,
        "clip_prompt": decision.clip_prompt,
        "diffusion_prompt": decision.diffusion_prompt,
        "negative_diffusion_prompt": decision.negative_diffusion_prompt
    }


def extract_diffusion_prompts(user_input: str, model_name: str = "mistral-small-latest") -> dict:
    """
    Extract only diffusion prompts from user input (for manual mode).
    
    Args:
        user_input: User's description of what to generate
        model_name: Mistral model to use
        
    Returns:
        Dictionary containing:
        - diffusion_prompt: Positive prompt for SD
        - negative_diffusion_prompt: Negative prompt for SD
    """
    try:
        check_api_key()
        chain = create_diffusion_prompt_chain(model_name)
        result = chain.invoke({"input": user_input})
        
        return {
            "diffusion_prompt": result.diffusion_prompt,
            "negative_diffusion_prompt": result.negative_diffusion_prompt
        }
    except Exception as e:
        # Fallback: return user input as-is with default negative
        print(f"  Agent extraction failed: {e}")
        return {
            "diffusion_prompt": user_input,
            "negative_diffusion_prompt": "blurry, low quality, distorted, deformed, ugly"
        }


def create_diffusion_prompt_reframer(model_name: str = "mistral-small-latest"):
    """
    Create a chain specifically for reframing simple prompts into detailed diffusion prompts.
    
    Returns:
        A callable that takes a simple prompt and returns an enhanced prompt.
    """
    from langchain_mistralai import ChatMistralAI
    from langchain_core.prompts import ChatPromptTemplate
    
    llm = ChatMistralAI(model=model_name, temperature=0.7)
    
    system_prompt = """You are an expert at crafting detailed, high-quality prompts for Stable Diffusion image generation.

Given a simple user description, enhance it into a detailed, visually descriptive prompt that will produce the best results.

Guidelines:
1. Add visual details: textures, colors, lighting, materials
2. Add quality modifiers: "highly detailed", "8k", "professional photography"
3. Add style hints if appropriate: "photorealistic", "cinematic lighting"
4. Keep the core subject/concept from the user's input
5. Keep the prompt concise but descriptive (under 75 words)
6. Do NOT add negative prompts or technical parameters
7. Return ONLY the enhanced prompt, nothing else

Example:
User: "a cute puppy"
Enhanced: "an adorable golden retriever puppy with fluffy fur, sitting on a grassy meadow, soft natural lighting, warm colors, highly detailed, professional pet photography, 8k resolution"
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Enhance this prompt for Stable Diffusion: {input}")
    ])
    
    chain = prompt | llm
    return chain


def reframe_prompt_for_diffusion(user_prompt: str, task_type: str = "replace", model_name: str = "mistral-small-latest") -> str:
    """
    Reframe a simple user prompt into a detailed Stable Diffusion prompt.
    
    Args:
        user_prompt: Simple user description (e.g., "a cute puppy")
        task_type: "replace" for object replacement, "background" for background fill
        model_name: Mistral model to use
        
    Returns:
        Enhanced prompt for Stable Diffusion, or original if enhancement fails
    """
    try:
        check_api_key()
        chain = create_diffusion_prompt_reframer(model_name)
        
        # Add context based on task type
        if task_type == "background":
            context_prompt = f"Create a background scene description for: {user_prompt}"
        else:
            context_prompt = user_prompt
        
        result = chain.invoke({"input": context_prompt})
        
        # Extract the text content
        enhanced = result.content.strip()
        
        # Validate - make sure it's not empty and makes sense
        if enhanced and len(enhanced) > 5:
            return enhanced
        else:
            return user_prompt
            
    except Exception as e:
        print(f"Prompt reframing failed: {e}")
        return user_prompt


# def main():
#     """Main function to run the agent."""
#     parser = argparse.ArgumentParser(
#         description="Agent Decision Script for Image Editing Tasks"
#     )
#     parser.add_argument(
#         "--list-models",
#         action="store_true",
#         help="List available models and exit"
#     )
#     parser.add_argument(
#         "--model",
#         type=str,
#         default="gemini-2.5-flash",
#         help="Model to use (default: gemini-2.5-flash)"
#     )
#     parser.add_argument(
#         "--prompt",
#         type=str,
#         help="User prompt to process"
#     )
#     parser.add_argument(
#         "--interactive",
#         action="store_true",
#         help="Run in interactive mode"
#     )
    
#     args = parser.parse_args()
    
#     # Check API key first
#     check_api_key()
    
#     # List models if requested
#     if args.list_models:
#         list_available_models()
#         return
    
#     # Create the agent chain
#     print(f"Initializing agent with model: {args.model}")
#     agent_chain = create_agent_chain(args.model)
    
#     # Process single prompt
#     if args.prompt:
#         print(f"\n--- Processing Prompt ---")
#         print(f"Input: {args.prompt}")
#         output = process_user_prompt(agent_chain, args.prompt)
#         print(f"\nOutput:")
#         print(json.dumps(output, indent=2))
#         return
    
#     # Interactive mode
#     if args.interactive:
#         print("\n--- Interactive Mode ---")
#         print("Enter your prompts (type 'quit' or 'exit' to stop):\n")
        
#         while True:
#             try:
#                 user_input = input("You: ").strip()
#                 if user_input.lower() in ['quit', 'exit', 'q']:
#                     print("Goodbye!")
#                     break
#                 if not user_input:
#                     continue
                    
#                 output = process_user_prompt(agent_chain, user_input)
#                 print(f"\nResult:")
#                 print(json.dumps(output, indent=2))
#                 print()
#             except KeyboardInterrupt:
#                 print("\nGoodbye!")
#                 break
#         return
    
#     # Default: run test example
#     print("\n--- TEST: Running Example ---")
#     test_prompt = "change the man in green shirt to a dog walking on 2 legs"
#     print(f"Input: {test_prompt}")
#     output = process_user_prompt(agent_chain, test_prompt)
#     print(f"\nOutput:")
#     print(json.dumps(output, indent=2))
    
#     print("\n--- Additional Test Examples ---")
#     test_prompts = [
#         "remove the car on the left side",
#         "make the lighting warmer like sunset",
#         "replace the tree in the center with a fountain"
#     ]
    
#     for prompt in test_prompts:
#         print(f"\nInput: {prompt}")
#         output = process_user_prompt(agent_chain, prompt)
#         print(f"Output: {json.dumps(output)}")


# if __name__ == "__main__":
#     main()
