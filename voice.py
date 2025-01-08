import asyncio
import websockets
import json
import base64
import wave
import logging
import os
from dotenv import load_dotenv
import os
import json
from dotenv import load_dotenv
from helper import (
    upload_to_the_vector_database,
    generate_embedding,
    retrieve_context,
    generate_storyline,
    generate_alternative_result,
    generate_textual_explanation_scenes_voiceovers,
    generate_pixar_image_base64,
    generate_video_url,
    process_all_scenes_parallel,
    process_scene,
    process_and_merge_videos
)
from openai import OpenAI
import base64
import concurrent.futures
from math import ceil
# Load the .env file
load_dotenv()
# Reassign the loaded configurations back to the same variable name
with open("configurations.json", "r") as json_file:
    configurations = json.load(json_file)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WS_URL = "wss://api.openai.com/v1/realtime"
# Replace the current variables with the loaded JSON data
tokenizer_model_name = configurations["models"]["tokenizer_model_name"]
storyline_model = configurations["models"]["storyline_model"]
storyline_temperature = configurations["models"]["storyline_temperature"]
explanation_model = configurations["models"]["explanation_model"]
explanation_temperature = configurations["models"]["explanation_temperature"]
MODEL = configurations["models"]["MODEL"]
image_generation_model = configurations["models"]["image_generation_model"]
voice_mode = configurations["models"]["voice_mode"]
kling_model = configurations["models"]["kling_model"]

# Audio settings
CHUNK = configurations["audio_settings"]["CHUNK"]
FORMAT = configurations["audio_settings"]["FORMAT"]
CHANNELS = configurations["audio_settings"]["CHANNELS"]
RATE = configurations["audio_settings"]["RATE"]

# Example Usage
user_query = configurations["example_usage"]["user_query"]
explanation_prompt = configurations["example_usage"]["explanation_prompt"]

# Storyline prompts
storyline_prompt_part_1 = configurations["storyline_prompt"]["part_1"]
storyline_prompt_part_2 = configurations["storyline_prompt"]["part_2"]

# Image description prompt
image_description_prompt = configurations["image_description_prompt"]

# Batch size
batch_size = configurations["batch_size"]

# Check variables
print(tokenizer_model_name, storyline_model, storyline_temperature, explanation_model, explanation_temperature, MODEL, image_generation_model, voice_mode, kling_model, CHUNK, FORMAT, CHANNELS, RATE, user_query, explanation_prompt, storyline_prompt_part_1, storyline_prompt_part_2, image_description_prompt, batch_size)


# Logging setup
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RealtimeNarrationClient:
    def __init__(self):
        self.ws = None
        self.audio_buffer = b""

    async def connect(self):
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
        try:
            self.ws = await websockets.connect(f"{WS_URL}?model={MODEL}", extra_headers=headers)
            logger.info("Connected to OpenAI Realtime API")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    async def send_event(self, event):
        try:
            await self.ws.send(json.dumps(event))
            logger.debug(f"Sent event: {event}")
        except Exception as e:
            logger.error(f"Error sending event: {e}")
            raise

    async def receive_events(self):
        try:
            async for message in self.ws:
                event = json.loads(message)
                logger.debug(f"Received event: {event}")
                await self.handle_event(event)
        except Exception as e:
            logger.error(f"Error receiving events: {e}")

    async def handle_event(self, event):
        if event.get("type") == "response.audio.delta":
            audio_data = base64.b64decode(event["delta"])
            self.audio_buffer += audio_data
        elif event.get("type") == "response.audio.done":
            self.save_audio()
            self.audio_buffer = b""
        else:
            logger.debug(f"Unhandled event type: {event.get('type')}")

    def save_audio(self):
        if not self.audio_buffer:
            logger.warning("No audio data to save")
            return
        try:
            with wave.open(AUDIO_FILENAME, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(FORMAT)
                wf.setframerate(RATE)
                wf.writeframes(self.audio_buffer)
            logger.info(f"Audio saved to {AUDIO_FILENAME}")
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")

    async def set_tone(self, tone_description):
        event = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": voice_mode,
                "instructions": tone_description
            }
        }
        await self.send_event(event)

    async def send_text(self, text):
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}]
            }
        }
        await self.send_event(event)
        await self.send_event({"type": "response.create"})

    async def run(self, tone_description, exact_text, voice_mode):
        await self.connect()
        receive_task = asyncio.create_task(self.receive_events())
        try:
            logger.info("Setting tone for narration")
            await self.set_tone(tone_description)
            
            logger.info("Sending narration text")
            await self.send_text(exact_text)
            
            await asyncio.sleep(10)  # Allow time for processing the response
        except Exception as e:
            logger.error(f"Error during run: {e}")
        finally:
            logger.info("Closing connection and session")
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
            await self.ws.close()

async def process_scenes(json_scenes, voice_mode):
    for scene in json_scenes:
        try:
            tone_description = scene.get("voice__attribute", "Speak in a neutral tone.") + " You will repeat everything I say to you in this tone."
            exact_text = scene.get("voiceover", "")
            audio_filename = f"scene_{scene['scene_number']}.wav"

            if not exact_text:
                logger.warning(f"Scene {scene['scene_number']} has no voiceover. Skipping...")
                continue

            logger.info(f"Processing Scene {scene['scene_number']}")
            client = RealtimeNarrationClient()

            # Set a unique filename for each scene's audio
            global AUDIO_FILENAME
            AUDIO_FILENAME = audio_filename

            await client.run(tone_description, exact_text, voice_mode)
            logger.info(f"Audio for Scene {scene['scene_number']} saved as {audio_filename}")
        except Exception as e:
            logger.error(f"Error processing Scene {scene['scene_number']}: {e}")

async def main():
    # Replace 'your-api-key' with your actual OpenAI API key
    client = OpenAI()

    storyline, textual_explanation = generate_textual_explanation_scenes_voiceovers(client, user_query, explanation_prompt, storyline_prompt_part_1, storyline_prompt_part_2, storyline_model, explanation_model, explanation_temperature, storyline_temperature, tokenizer_model_name)

    raw_output = storyline
    # Clean up potential code block markers like ```json
    if raw_output.startswith("```json"):
        raw_output = raw_output.strip("```json").strip("```")

    # Parse JSON content
    try:
        json_output = json.loads(raw_output)
    except json.JSONDecodeError:
        # Fallback if minor formatting issues exist
        cleaned_output = raw_output.replace("\n", "").strip()
        json_output = json.loads(cleaned_output)

    # Process all scenes in parallel with batching
    all_results = process_all_scenes_parallel(json_output, client, image_description_prompt, batch_size, kling_model, image_generation_model)
    output_file = "all_results.json"
    with open(output_file, "w") as file:
        json.dump(all_results, file, indent=4)
    print(f"Results saved to {output_file}")


    # Print results
    for result in all_results:
        if "error" in result:
            print(f"Scene {result['scene_number']} failed: {result['error']}")
        else:
            print(f"Scene {result['scene_number']} Video URL: {result['video_url']}")
    await process_scenes(json_output, voice_mode)
    await process_and_merge_videos("all_results.json", "final_output.mp4")


if __name__ == "__main__":
    asyncio.run(main())