import concurrent.futures
from math import ceil
import pyarrow.fs
import sycamore
import json
import os
from pinecone import Pinecone
from sycamore.functions.tokenizer import OpenAITokenizer
from sycamore.llms import OpenAIModels, OpenAI
from sycamore.transforms import COALESCE_WHITESPACE
from sycamore.transforms.merge_elements import GreedySectionMerger
from sycamore.transforms.partition import ArynPartitioner
from sycamore.transforms.embed import OpenAIEmbedder
from sycamore.materialize_config import MaterializeSourceMode
from sycamore.utils.pdf_utils import show_pages
from sycamore.transforms.summarize_images import SummarizeImages
from sycamore.context import ExecMode
from pinecone import ServerlessSpec
from openai import OpenAI
import time
import jwt
import base64
import requests
import ffmpeg

def upload_to_the_vector_database(paths, model_name, max_tokens, dimensions):
    """
    Processes and uploads documents to a vector database (Pinecone) for efficient retrieval and similarity search.

    This function:
    - Extracts text, tables, and images from PDF documents.
    - Processes and partitions the content into manageable chunks.
    - Embeds the chunks using an OpenAI embedding model.
    - Uploads the embedded data to a Pinecone vector database.

    Parameters:
    -----------
    paths : list of str
        A list of file paths to the PDF documents to be processed and uploaded.
    model_name : str
        The name of the OpenAI embedding model to be used for generating embeddings (e.g., "text-embedding-ada-002").
    max_tokens : int
        The maximum number of tokens per chunk for partitioning and embedding.
    dimensions : int
        The dimensionality of the embedding vectors (must match the Pinecone index configuration).

    Process Workflow:
    -----------------
    1. **Document Processing**:
       - Reads and parses the PDF documents.
       - Extracts tables, text, and images, optionally using OCR for scanned documents.
       - Splits the content into smaller chunks for efficient embedding.
    2. **Embedding**:
       - Embeds each chunk of content using the specified OpenAI embedding model.
    3. **Uploading to Pinecone**:
       - Writes the embedded chunks to a Pinecone index.
       - Ensures the index is configured with the specified dimensionality and cosine distance metric.
    4. **Verification**:
       - Queries the Pinecone database to verify the successful upload of document chunks.

    Requirements:
    -------------
    - Ensure the Pinecone index is configured and accessible.
    - Provide a valid OpenAI API key and Pinecone API key via environment variables.

    Notes:
    ------
    - The function uses the Sycamore library for efficient ETL (Extract, Transform, Load) operations on documents.
    - The function assumes the Pinecone index name is "test". Modify `index_name` in the function if a different index name is required.

    Example Usage:
    --------------
    ```python
    upload_to_the_vector_database(
        paths=["document1.pdf", "document2.pdf"],
        model_name="text-embedding-ada-002",
        max_tokens=8191,
        dimensions=1536
    )
    ```

    Output:
    -------
    - The function prints a verification output confirming the successful upload of data to the Pinecone vector database.

    """
    # Initialize the Sycamore context
    ctx = sycamore.init(ExecMode.LOCAL)

    # Initialize the tokenizer
    tokenizer = OpenAITokenizer(model_name)

    ds = (
        ctx.read.binary(paths, binary_format="pdf")
        # Partition and extract tables and images
        .partition(partitioner=ArynPartitioner(
            threshold="auto",
            use_ocr=True,
            extract_table_structure=True,
            extract_images=True,
            source="docprep"
        ))
        # Use materialize to cache output. If changing upstream code or input files, change setting from USE_STORED to RECOMPUTE to create a new cache.
        .materialize(path="./materialize/partitioned", source_mode=MaterializeSourceMode.RECOMPUTE)
        # Merge elements into larger chunks
        .merge(merger=GreedySectionMerger(
          tokenizer=tokenizer,  max_tokens=max_tokens, merge_across_pages=False
        ))
        # Split elements that are too big to embed
        .split_elements(tokenizer=tokenizer, max_tokens=max_tokens)
    )

    ds.execute()

    # Display the first 3 pages after chunking
    # show_pages(ds, limit=3)

    ### seperator ###

    embedded_ds = (
        # Copy document properties to each Document's sub-elements
        ds.spread_properties(["path", "entity"])
        # Convert all Elements to Documents
        .explode() 
        # Embed each Document. You can change the embedding model. Make your target vector index matches this number of dimensions.
        .embed(embedder=OpenAIEmbedder(model_name=model_name))
    )
    # To know more about docset transforms, please visit https://sycamore.readthedocs.io/en/latest/sycamore/transforms.html

    ### seperator ###

    #### might need to make a seperate parametrized function for this for now it is here
    #### NEED TO USE SINGLE SINGLESTORE INSTEAD OF PINECONE -> IMPORTANT TO TARGET SPONSOR PRIZE
    
    # Create an instance of ServerlessSpec with the specified cloud provider and region
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    index_name = "test"
    # Write data to a Pinecone index 
    embedded_ds.write.pinecone(index_name=index_name, 
        dimensions=dimensions, 
        distance_metric="cosine",
        index_spec=spec
    )

    ### seperator ###
    #### not working for some reason RATE LIMITS????
#     # Verify data has been loaded using DocSet Query to retrieve chunks
#     print("#### VERIFICATION START ####")
#     query_docs = ctx.read.pinecone(index_name=index_name, api_key=os.getenv('PINECONE_API_KEY'))
#     query_docs.show(show_embedding=False)
#     print("#### VERIFICATION END ####")

### seperator ###

# Function to generate embeddings
def generate_embedding(client, query, tokenizer_model_name):
    """
    Generates an embedding vector for a given query using the specified OpenAI embedding model.

    Parameters:
    -----------
    query : str
        The input query or text for which an embedding vector is to be generated.

    Returns:
    --------
    list of float
        The embedding vector representing the semantic meaning of the input query.

    Notes:
    ------
    - The function uses OpenAI's `client.embeddings.create` to generate embeddings.
    - Ensure the OpenAI API key is set and the `client` instance is correctly initialized.
    - The embedding model (`text-embedding-3-small`) can be replaced with other models as needed.

    Example Usage:
    --------------
    ```python
    query = "Explain how neural networks work."
    embedding_vector = generate_embedding(query)
    print(embedding_vector)
    ```
    """
    response = client.embeddings.create(
        input=query,
        model= tokenizer_model_name  # Replace with your desired model
    )
    return response.data[0].embedding

# Function to retrieve context from Pinecone
def retrieve_context(index, client, user_query, tokenizer_model_name):
    """
    Retrieves relevant context from the Pinecone vector database for a given user query.

    This function:
    - Generates an embedding for the user query using `generate_embedding`.
    - Queries the Pinecone vector database using the embedding to find the top-k matching chunks.
    - Extracts relevant text metadata from the retrieved matches to form the context.

    Parameters:
    -----------
    user_query : str
        The user’s input query for which relevant context is to be retrieved.

    Returns:
    --------
    str
        A concatenated string of relevant text metadata from the top-k matches in the Pinecone vector database.

    Notes:
    ------
    - Ensure the Pinecone index is initialized and accessible via the `index` object.
    - The metadata key `text_representation` should exist in the Pinecone data schema.
    - Modify `top_k` as needed to adjust the number of results retrieved from the Pinecone database.

    Example Usage:
    --------------
    ```python
    user_query = "What are RNNs used for?"
    context = retrieve_context(user_query)
    print(context)
    ```

    Workflow:
    ---------
    1. Generate an embedding for the user query using the `generate_embeddinggenerate_embedding` function.
    2. Query the Pinecone index for the top-k matching chunks using the query embedding.
    3. Extract and concatenate relevant text metadata (`text_representation`) from the matches.

    """
    # Step 1: Generate embedding for the user query
    query_vector = generate_embedding(client, user_query, tokenizer_model_name)
    
    # Step 2: Query Pinecone for relevant matches
    response = index.query(
        vector=query_vector,
        top_k=6,
        include_metadata=True
    )

    # Debugging: Print the full response to verify structure
#     print(response)

    # Step 3: Extract relevant context from metadata
    # Use 'text_representation' instead of 'text'
    context = "\n".join([item["metadata"]["text_representation"] for item in response["matches"]])
    return context


def generate_storyline(client, context, user_query, prompt_part_1, prompt_part_2, storyline_mode, storyline_temperature):
    """
    Generates a Pixar/Disney-style storyline with scenes based on the provided context.
    """
    prompt = f"""
    **{prompt_part_1}: "{user_query}". Based on the following context only** 
    
    **Context**:
    {context}
    
    {prompt_part_2}"""
    # Make the API call
    try:
        response = client.chat.completions.create(
            model = storyline_model,  # Updated to the best ChatGPT model
            messages = [{"role": "user", "content": prompt}],
            max_tokens = 3000,
            temperature = storyline_temperature,
        )
        # Parse and validate JSON response
        output = response.choices[0].message.content
        return output
    except json.JSONDecodeError:
        raise ValueError("The API response is not a valid JSON.")
    except Exception as e:
        raise RuntimeError(f"Error generating storyline: {str(e)}")

# Function to generate an alternative result for the user query
def generate_alternative_result(client, context, user_query, prompt_part_1, explanation_model, explanation_temperature):
    """
    Retrieves context and generates an alternative result for the user query
    using a different temperature setting for diversity.
    """
    # Create a prompt for generating an alternative result
    prompt = f"""
    {prompt_part_1}. Use the following context for answering"

    Context:
    {context}

    Question:
    {user_query}
    """

    # Generate a response using OpenAI
    response = client.chat.completions.create(
            model = explanation_model,  # Updated to the best ChatGPT model
            messages = [{"role": "user", "content": prompt}],
            max_tokens = 3000,
            temperature = explanation_temperature,
        )

    return response.choices[0].text

### seperator ###

def generate_textual_explanation_scenes_voiceovers(client, user_query, explanation_prompt, prompt_part_1, prompt_part_2, storyline_model, explanation_model explanation_temperature, storyline_temperature, tokenizer_model_name):
    
    # Set up Pinecone client
    pinecone_client = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )

    index = pinecone_client.Index(host = "https://test-gfkht3t.svc.aped-4627-b74a.pinecone.io")
    
    # Step 1: Retrieve context from Pinecone
    context = retrieve_context(index, client, user_query, tokenizer_model_name)

    # Step 2: Generate storyline with scenes
    storyline = generate_storyline(client, context, user_query, prompt_part_1, prompt_part_2, storyline_model, storyline_temperature)
    alternative_result = generate_alternative_result(client, context, user_query, explanation_prompt, explanation_model, explanation_temperature)

    # Print the result
    print("STORYLINE AS FOLLOWS:\n", storyline)
    print("\nTEXTUAL EXPLANATION AS FOLLOWS:\n", alternative_result)
    
    return storyline, alternative_result

def generate_video_url(image_base64, prompt, kling_model):
    """
    Generate a video URL from an image file and a prompt.

    :param image_path: Path to the image file
    :param prompt: Text prompt for video generation
    :return: Video URL if available, or an error message
    """
    API_URL = "https://api.klingai.com/v1/videos/image2video"
    # Generate JWT token
    payload = {
        "iss": os.getenv("KLING_ACCESS_KEY"),
        "exp": int(time.time()) + 1800,  # Token expires in 30 minutes
        "nbf": int(time.time()) - 5      # Token valid 5 seconds in the past
    }
    api_token = jwt.encode(payload, os.getenv("KLING_SECRET_KEY"), algorithm="HS256")

    # API request headers
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    # API request payload
    payload = {
        "model_name": kling_model,
        "image": image_base64,
        "prompt": prompt,
        "mode": "std",
        "duration": "5",
        "cfg_scale": 0.5
    }

    # Make the initial API call to generate the video
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return f"Error: {response.text}"

    response_data = response.json()
    task_id = response_data.get("data", {}).get("task_id")
    if not task_id:
        return "Error: Task ID not found in response"

    # Poll the task status until the video is ready
    task_status_url = f"{API_URL}/{task_id}"
    while True:
        status_response = requests.get(task_status_url, headers=headers)
        if status_response.status_code != 200:
            return f"Error: {status_response.text}"

        status_data = status_response.json()
        task_status = status_data.get("data", {}).get("task_status")

        if task_status == "succeed":
            videos = status_data.get("data", {}).get("task_result", {}).get("videos", [])
            if videos:
                return videos[0].get("url", "Video URL not found")
            else:
                return "Error: No videos found in task result"

        elif task_status == "failed":
            return "Error: Task failed to process"

        # Wait before polling again
        time.sleep(10)

def generate_pixar_image_base64(client, prompt, image_generation_model):
    """
    Generates a Pixar-like animated image based on the given image description prompt using OpenAI's DALL·E model,
    and returns the image as a base64 string.

    :param prompt: A string containing the image description.
    :return: A base64-encoded string of the generated image.
    """
    try:
        # Add specific Pixar-like animation style details to the prompt
        enhanced_prompt = prompt
        
        # Call the DALL·E API to generate the image
        response = client.images.generate(
            prompt=enhanced_prompt,
            n=1,
            model = image_generation_model,
            size="1024x1024",
            response_format="b64_json"  # Request base64 format
        )
        
        # Access the base64 image data
        image_base64 = response.data[0].b64_json
        print("Image generated successfully.")
        return image_base64

    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def process_scene(scene, client, image_description_prompt, kling_model, image_generation_model):
    """
    Process a single scene: generate image description, create image, and get video URL.
    """
    try:
        # Create the image description
        image_description = f"{image_description_prompt}: {scene['image']} {scene['action']}"
        
        # Generate the image in base64 format
        image_base64 = generate_pixar_image_base64(client, image_description, image_generation_model)

        # Generate the video URL
        video_url = generate_video_url(image_base64, scene["action"], kling_model) if image_base64 else None

        return {
            "scene_number": scene["scene_number"],
            "video_url": video_url
        }
    except Exception as e:
        return {
            "scene_number": scene["scene_number"],
            "error": str(e)
        }

def process_all_scenes_parallel(json_output, client, image_description_prompt, batch_size=3, kling_model, image_generation_model):
    """
    Process all scenes in parallel in batches of a specified size.
    """
    results = []
    num_batches = ceil(len(json_output) / batch_size)

    for batch_index in range(num_batches):
        # Get the current batch of scenes
        start_index = batch_index * batch_size
        end_index = min(start_index + batch_size, len(json_output))
        batch = json_output[start_index:end_index]

        # Use ThreadPoolExecutor for parallel processing within the batch
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks in the current batch to the executor
            future_to_scene = {
                executor.submit(process_scene, scene, client, image_description_prompt, kling_model, image_generation_model): scene["scene_number"]
                for scene in batch
            }

            # Collect the results as they complete
            for future in concurrent.futures.as_completed(future_to_scene):
                scene_number = future_to_scene[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "scene_number": scene_number,
                        "error": str(e)
                    })
        time.sleep(20)

    # Sort results by scene_number to maintain order
    results.sort(key=lambda x: x["scene_number"])
    return results

def process_and_merge_videos(json_file_path, final_output):
    """
    Process videos from JSON file, repeat them to match audio durations,
    and merge them into one final video.
    """
    def download_file(url, output_path):
        """Download a file from a URL and save it locally."""
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Downloaded: {output_path}")
        else:
            raise Exception(f"Failed to download {url}. Status code: {response.status_code}")

    def repeat_video_to_audio(video_url, audio_path, output_path):
        """Repeat video from URL to match audio duration."""
        # Temporary file for downloaded video
        video_temp_path = "temp_video.mp4"

        # Download video from URL
        download_file(video_url, video_temp_path)

        # Get video and audio durations
        video_info = ffmpeg.probe(video_temp_path)
        audio_info = ffmpeg.probe(audio_path)

        video_duration = float(video_info['streams'][0]['duration'])
        audio_duration = float(audio_info['streams'][0]['duration'])

        # Calculate repetitions
        repetitions = int(audio_duration // video_duration) + 1

        # Generate repeated video
        repeated_video = f"{output_path}_repeated.mp4"
        ffmpeg.input(video_temp_path, stream_loop=repetitions - 1).output(
            repeated_video, t=audio_duration
        ).run()

        # Merge repeated video with audio
        video_input = ffmpeg.input(repeated_video)  # Separate video input
        audio_input = ffmpeg.input(audio_path)      # Separate audio input

        ffmpeg.output(video_input, audio_input, output_path, vcodec="libx264", acodec="aac", strict="experimental").run()

        # Clean up temporary files
        os.remove(video_temp_path)
        os.remove(repeated_video)
        print(f"Final output saved: {output_path}")

    def merge_videos(video_list, final_output):
        """Merge videos into one final video."""
        # Create a temporary text file listing the videos
        with open("videos_to_merge.txt", "w") as f:
            for video in video_list:
                f.write(f"file '{video}'\n")

        # Use FFmpeg to concatenate videos
        ffmpeg.input("videos_to_merge.txt", format="concat", safe=0).output(
            final_output, c="copy"
        ).run()

        # Clean up the temporary text file
        os.remove("videos_to_merge.txt")
        print(f"Merged video saved: {final_output}")

    # Load JSON from a file
    with open(json_file_path, 'r') as file:
        all_results = json.load(file)

    # Process each scene
    output_videos = []
    for scene in all_results:
        scene_number = scene["scene_number"]
        video_url = scene["video_url"]
        audio_path = f"scene_{scene_number}.wav"  # Automatically generate audio path
        output_path = f"scene_{scene_number}_output.mp4"
        
        print(f"Processing Scene {scene_number}...")
        repeat_video_to_audio(video_url, audio_path, output_path)
        output_videos.append(output_path)

    # Merge all scene videos into one final video
        merge_videos(output_videos, final_output)