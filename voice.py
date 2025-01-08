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
    process_scene
)
from openai import OpenAI
import base64
import concurrent.futures
from math import ceil
# Load the .env file
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WS_URL = "wss://api.openai.com/v1/realtime"
MODEL = "gpt-4o-realtime-preview-2024-10-01"
AUDIO_FILENAME = "output.wav"

# Audio settings
CHUNK = 1024
FORMAT = 2  # 2 bytes per sample (16-bit PCM)
CHANNELS = 1
RATE = 24000
# Example Usage
user_query = "explain model distillation mentioned in the paper"
explanation_prompt = "You are an educational assistant. Using the following context, answer the question in a concise, informative, and clear manner for a student. Provide answer that is easy for a student to understand. Keep things short and simple, ensuring clarity."
storyline_prompt_part_1 = "Generate a Pixar/Disney-style animated explanation for the concept"
storyline_prompt_part_2 = """**Storyline Requirements**:
    1. **Story Environment**:
       - Create visually engaging scenes with relevant backdrops that evolve logically with the storyline.
           - The motive is to have a story that is used to  explain the concepts//qqueries asked by the user. e.g. DO NOT rept this example take inspration from it "to explain addition" "story is a person bougtht two bananas then  someoone gave him one more banana now he has 3 bananas make it a stry and have voiceovers"
       - The environment should complement and enhance the narrative, helping illustrate key ideas.

        2. **Character Design**:
       - Design relatable and lively characters (e.g., curious kids, a wise mentor, or anthropomorphic objects) that guide the viewer through the concept.
       - Characters must interact dynamically with their surroundings and evolve naturally with the narrative.

    3. **Actions and Visual Metaphors**:
       - Characters actively demonstrate or interact with objects that represent parts of the concept (e.g., glowing nodes for connections, gears for processes, or animated charts for data).
       - Incorporate playful and clear visual metaphors to simplify complex ideas.

    4. **Tone and Mood**:
       - Use vibrant colors, dynamic lighting, and playful animations to maintain an engaging and entertaining tone.
       - Ensure that the tone is consistent, transitioning smoothly from scene to scene as the concept deepens.

    5. **Voiceover Script**:
       - Each scene includes a matching voiceover script:
         - Explains the visuals in simple, engaging language.
         - Uses analogies, humor, and storytelling to clarify and retain viewer interest.
         - Concludes with an encouraging summary that ties all the concepts together.

    **Output Format**:
    Provide the output for each scene in the following structure AD ONLY PUTPUT THE JSON NO TEXT FOR PYTHON FORMATTING and only give JSON NO TEXT OTHER THAN SINGLE JSON FILE FOR ALL SCENES VERY IMPORTANT:
    ```json
    [{{
      "scene_number": 1,
      "image": "Describe the visual elements of the scene: environment, characters, and key props/objects in detail.",
      "action": "Describe what is happening: how characters interact, how objects or visuals move, and how the concept is illustrated.",
      "voiceover": "Provide a narration script that aligns with the visuals and explains the scene clearly and engagingly.",
      "voice__attribute": "A single sentence describing how to speak, such as 'Speak in a calm and friendly tone, like a welcoming radio host. Add additional details as necessary to align with the scene.'"

    }}]
    ```

    **Tips**:
    - Use visual metaphors like flowing rivers for data, glowing gears for systems, or a growing tree for organic processes.
    - Add playful details (e.g., animated chalkboard doodles or talking objects) to make explanations lively.
    - Whatever object is present in the action shoud have all its characteristics defined in the image, good to make the image as descriptive as possible
    Focus on creating a cohesive, fun, and informative narrative that would feel right at home in a Pixar or Disney short film.
    """

image_description_prompt = "make a Pixar like animated photo for the following image description"

batch_size = 3

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
                "voice": "sage",
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

    async def run(self, tone_description, exact_text):
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

async def process_scenes(json_scenes):
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

            await client.run(tone_description, exact_text)
            logger.info(f"Audio for Scene {scene['scene_number']} saved as {audio_filename}")
        except Exception as e:
            logger.error(f"Error processing Scene {scene['scene_number']}: {e}")

async def main():
    # Replace 'your-api-key' with your actual OpenAI API key
    client = OpenAI()

    storyline, textual_explanation = generate_textual_explanation_scenes_voiceovers(client, user_query, explanation_prompt, storyline_prompt_part_1, storyline_prompt_part_2)

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
    all_results = process_all_scenes_parallel(json_output, client, image_description_prompt, batch_size=batch_size)
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
    await process_scenes(json_output)

if __name__ == "__main__":
    asyncio.run(main())
