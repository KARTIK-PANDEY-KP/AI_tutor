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
def generate_embedding(client, query):
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
        model="text-embedding-3-small"  # Replace with your desired model
    )
    return response.data[0].embedding

# Function to retrieve context from Pinecone
def retrieve_context(index, client, user_query):
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
    1. Generate an embedding for the user query using the `generate_embedding` function.
    2. Query the Pinecone index for the top-k matching chunks using the query embedding.
    3. Extract and concatenate relevant text metadata (`text_representation`) from the matches.

    """
    # Step 1: Generate embedding for the user query
    query_vector = generate_embedding(client, user_query)
    
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


def generate_storyline(client, context, user_query):
    """
    Generates a Pixar/Disney-style storyline with scenes based on the provided context.
    """
    
    prompt = f"""
    **Generate a Pixar/Disney-style animated explanation for the concept: "{user_query}". Based on the following context only** 
    
    **Context**:
    {context}
    
    **Storyline Requirements**:
    1. **Story Environment**:
       - Create visually engaging scenes with relevant backdrops that evolve logically with the storyline (e.g., classroom for introductions, whimsical spaces for imagination, futuristic labs for technical depth).
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
      "voiceover": "Provide a narration script that aligns with the visuals and explains the scene clearly and engagingly."
    }}]
    ```

    **Tips**:
    - Use visual metaphors like flowing rivers for data, glowing gears for systems, or a growing tree for organic processes.
    - Add playful details (e.g., animated chalkboard doodles or talking objects) to make explanations lively.

    Focus on creating a cohesive, fun, and informative narrative that would feel right at home in a Pixar or Disney short film.
    """
    # Generate response from OpenAI
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",  # Use the updated model name
        prompt=prompt,
        max_tokens=3000,
        temperature=0.7
    )
    return response.choices[0].text 

# Function to generate an alternative result for the user query
def generate_alternative_result(client, context, user_query):
    """
    Retrieves context and generates an alternative result for the user query
    using a different temperature setting for diversity.
    """

    # Create a prompt for generating an alternative result
    prompt = f"""
    You are an educational assistant. Using the following context, answer the question in a concise, informative, and clear manner for a student.

    Context:
    {context}

    Question:
    {user_query}

    Provide answer that is easy for a student to understand. Keep things short and simple, ensuring clarity.
    """

    # Generate a response using OpenAI
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",  # Use the appropriate model name
        prompt=prompt,
        max_tokens=3000,
        temperature=1.0  # Higher temperature for diversity
    )

    return response.choices[0].text

### seperator ###

def generate_textual_explanation_scenes_voiceovers(user_query):
    
    # Set up Pinecone client
    pinecone_client = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )

    index = pinecone_client.Index(host = "https://test-gfkht3t.svc.aped-4627-b74a.pinecone.io")
    client = OpenAI()
    
    # Step 1: Retrieve context from Pinecone
    context = retrieve_context(index, client, user_query)

    # Step 2: Generate storyline with scenes
    storyline = generate_storyline(client, context, user_query)
    alternative_result = generate_alternative_result(client, context, user_query)

    # Print the result
    print("STORYLINE AS FOLLOWS:\n", storyline)
    print("\nTEXTUAL EXPLANATION AS FOLLOWS:\n", alternative_result)
    
    return storyline, alternative_result