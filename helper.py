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
import PyPDF2
import pandas as pd
import matplotlib.pyplot as plt
import json

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
        top_k=10,
        include_metadata=True
    )

    # Debugging: Print the full response to verify structure
#     print(response)

    # Step 3: Extract relevant context from metadata
    # Use 'text_representation' instead of 'text'
    context = "\n".join([item["metadata"]["text_representation"] for item in response["matches"]])
    return context


def generate_storyline(client, context, user_query, prompt_part_1, prompt_part_2, storyline_model, storyline_temperature):
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

    return response.choices[0].message.content

### seperator ###

def generate_textual_explanation_scenes_voiceovers(client, user_query, explanation_prompt, prompt_part_1, prompt_part_2, storyline_model, explanation_model, explanation_temperature, storyline_temperature, tokenizer_model_name):
    
    # Set up Pinecone client
    pinecone_client = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )
#     pinecone_client.delete_index("test")

    index = pinecone_client.Index(host = "https://test-gfkht3t.svc.aped-4627-b74a.pinecone.io")
#     index.delete(delete_all=True)

    
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

def process_all_scenes_parallel(json_output, client, image_description_prompt, batch_size, kling_model, image_generation_model):
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

async def process_and_merge_videos(json_file_path, final_output):
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
        ).run(overwrite_output = True)

        # Merge repeated video with audio
        video_input = ffmpeg.input(repeated_video)  # Separate video input
        audio_input = ffmpeg.input(audio_path)      # Separate audio input

        ffmpeg.output(video_input, audio_input, output_path, vcodec="libx264", acodec="aac", strict="experimental").run(overwrite_output = True)

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
        ).run(overwrite_output = True)

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
        
def semantic_parts(client, prompt_textual, semantic_seg_model = "gpt-4o", semantic_seg_temperature = 1):
    """
    Retrieves context and generates an alternative result for the user query
    using a different temperature setting for diversity.

    Parameters:
        client (object): The API client to generate responses (e.g., OpenAI client).
        context (str): Additional context that might be used for processing.
        user_query (str): The user's query or input.
        prompt_textual (str): The text input for generating the prompt.
        explanation_model (str): The model used for generating explanations.
        explanation_temperature (float): The temperature setting for response generation.

    Returns:
        str: The generated alternative response in JSON format.
    """

    # Create a prompt for generating an alternative result
    prompt = f"""
    Break down the given text thesis into 4-6 key concepts that can be queried into an academic database: like the following:
    The output should be in JSON format strictly and only output the points, not any explanation or other texts.
    A sample thesis is what the user sends:
    "The increasing prevalence of low-frequency urban noise pollution, particularly from HVAC systems and construction equipment, creates persistent cognitive strain that manifests not only in immediate stress responses but also in subtle long-term changes to residents' decision-making patterns and risk assessment capabilities, suggesting that urban planning must expand beyond traditional decibel-level regulations to account for the specific cognitive impacts of different sound frequencies."
    
    And the output should look like this:
    {{
      "main_concepts": [
        "Low-frequency noise",
        "Cognitive strain",
        "Regulation limitations",
        "Urban planning"
      ]
    }}

    User prompt:
    {prompt_textual}
    """

    # Generate a response using the client
    response = client.chat.completions.create(
        model=semantic_seg_model,  # Specify the model for response generation
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000,
        temperature=semantic_seg_temperature,
    )

    return response.choices[0].message.content

def retrieve_context_ss(index, client, user_query, tokenizer_model_name):
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
    conn = s2.connect('admin:Ujl3TwqkUOtKnBIBf0RWF6DrLAMtbS9i@svc-9228af81-d01c-4393-abe1-74fcb3b87cf8-dml.aws-oregon-3.svc.singlestore.com:3306/kp')

    # Step 3: Query SingleStore for relevant matches
    query = """
    SELECT text_representation, 
           (1 - (DOT_PRODUCT(embedding, ?) / (L2_NORM(embedding) * L2_NORM(?)))) AS cosine_distance
    FROM vector_index
    ORDER BY cosine_distance ASC
    LIMIT ?;
    """

    query_vector_list = query_vector.tolist()
    with conn.cursor() as cursor:
        cursor.execute(query, (query_vector_list, query_vector_list, top_k))
        results = cursor.fetchall()

    # Debugging: Print the full response to verify structure
#     print(response)

    # Step 3: Extract relevant context from metadata
    # Use 'text_representation' instead of 'text'
    context = "\n".join([item["metadata"]["text_representation"] for item in response["matches"]])
    return context

def upload_to_the_vector_database_ss(paths, model_name, max_tokens, dimensions):
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
#    spec = ServerlessSpec(cloud="aws", region="us-east-1")
#   index_name = "test"
    # Write data to a Pinecone index 
#    embedded_ds.write.pinecone(index_name=index_name, 
#        dimensions=dimensions, 
#        distance_metric="cosine",
#        index_spec=spec
#   )

    # Connect to SingleStore database
    import singlestoredb as s2
    am = embedding_ds.take_all()
    # Define dimensions for embeddings (for reference, not used in schema directly)
    dimensions = 1536  # Example: Number of dimensions for embeddings

    # Create a connection to the database
    conn = s2.connect('admin:Ujl3TwqkUOtKnBIBf0RWF6DrLAMtbS9i@svc-9228af81-d01c-4393-abe1-74fcb3b87cf8-dml.aws-oregon-3.svc.singlestore.com:3306/kp')

    with conn:
        with conn.cursor() as cursor:
            cursor.execute("USE kp;")

            # Check if the connection is open
            if cursor.is_connected():
                print("Connection is open")

            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS vector_index1 (
                id VARCHAR(255) PRIMARY KEY,          -- Unique ID
                embedding VECTOR({dimensions}, F32) NOT NULL,  -- Embedding vector
                path VARCHAR(1024),                   -- Path from metadata
                filetype VARCHAR(255),                -- Filetype from metadata
                text LONGTEXT                         -- Text from metadata
            );
            """)


            print("Table `vector_index` created or verified.")

    import json
    import uuid
    import singlestoredb as s2

    # # Prepare data for insertion
    records = []

    # New Insert
    for doc in am:
        # Extracting the unique document ID, or generating a UUID if it's not present
        unique_id = doc.get('doc_id', str(uuid.uuid4()))

        # Extracting metadata from the 'properties' field
        metadata = json.dumps({
            "path": "".join(doc['properties'].get('path', "").split("/")[-1].split(".")),
            "filetype": doc['properties'].get('filetype', ""),
            "text": doc.get('text_representation', "").split(":")[-1]
        }) 

        # Skip documents without embeddings
        if doc.get('embedding') is None:
            print(f"Skipping document with path: {unique_id}, embedding is None")
            continue

    #     Convert embedding string into a list (it appears to be a string representation of a list)
        embedding = list(doc['embedding'])  # Ensure embedding is a list, use eval safely in controlled environments

        # Append the record to the list of records to be inserted
        records.append((unique_id, embedding, metadata))


    conn = s2.connect('admin:Ujl3TwqkUOtKnBIBf0RWF6DrLAMtbS9i@svc-9228af81-d01c-4393-abe1-74fcb3b87cf8-dml.aws-oregon-3.svc.singlestore.com:3306/kp')

    with conn:
        with conn.cursor() as cursor:
            cursor.execute("USE kp;")
            i = 1
            for record in records:
                i = i + 1
                embedding = record[1]
                metadata_json = record[2]  # Metadata is a JSON string
                metadata = json.loads(metadata_json)

                # Extract metadata fields
                path = metadata.get('path', '')
                filetype = metadata.get('filetype', '')
                text = metadata.get('text', '')

                # Insert into the table
                cursor.execute(f"""
                    INSERT INTO `vector_index1` (id, embedding, path, filetype, text)
                    VALUES ({i}, {embedding}, {path}, {filetype}, {text})
                """)
            conn.commit()

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using PyPDF2.
    """
    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def extract_data_with_gpt(text, prompt_template, model="gpt-4o"):
    """
    Use GPT-4 to extract data or summaries from the given text.
    """
    prompt = prompt_template.format(text=text)
    client = OpenAI()
    try:
        # Call the OpenAI Chat API
        response =  client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Error during GPT API call: {e}")


def visualize_data(data, output_path="graph.png"):
    """
    Visualize the extracted data using a bar chart and save the chart as an image.
    """
    try:
        df = pd.DataFrame(data)
        plt.figure(figsize=(8, 6))
        plt.bar(df['Category'], df['Value'], color='skyblue')
        plt.title('Data Visualization', fontsize=14)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=90, ha='right', fontsize=8)  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to prevent clipping
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Bar chart saved as {output_path}")
    except KeyError as e:
        raise ValueError(f"Missing expected key in data: {e}")

def encode_image_to_base64(image_path):
    """
    Encode an image file to a Base64 string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_pdf(pdf_path):
    """
    Process the PDF to extract text, generate a summary, and save a bar chart visualization.
    """
    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)

    # Define prompts
    summary_prompt_template = """
    Summarize the following paper text into a concise, coherent summary. Focus on the main points, findings, or arguments.

    Paper Text:
    {text}
    """

    data_prompt_template = """
    Analyze the following paper text and identify any numerical data or trends that can be represented in tabular form. 
    Generate the data in JSON format with the following structure:
    [
      {{"Category": "Category Name", "Value": Numerical Value}},
      {{"Category": "Category Name", "Value": Numerical Value}}
    ]

    Ensure the output is valid JSON. Do not include any text, only JSON, nothing else at all.
    Paper Text:
    {text}
    """

    # Generate summary
    summary = extract_data_with_gpt(pdf_text, summary_prompt_template)
    print("Summary of the PDF:\n", summary)

    # Extract numerical data
    extracted_data = extract_data_with_gpt(pdf_text, data_prompt_template)

    # Clean up and parse the JSON output
    raw_output = extracted_data.strip()
    if raw_output.startswith("```json"):
        raw_output = raw_output.strip("```json").strip("```")

    try:
        data = json.loads(raw_output)
        print("Extracted Data:", data)
    except json.JSONDecodeError:
        cleaned_output = raw_output.replace("\n", "").strip()
        data = json.loads(cleaned_output)

    # Save visualization
    visualize_data(data, output_path="graph.png")
    image_base64 = encode_image_to_base64("graph.png")

    # Return the summary
    return summary, image_base64