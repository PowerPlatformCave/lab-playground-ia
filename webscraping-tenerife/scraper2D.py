import json
import math
import os
import re
import time
import uuid
import itertools
import threading
from urllib.parse import urljoin
from scipy import spatial            

import requests
from azure.core.exceptions import ResourceExistsError
from azure.data.tables import TableServiceClient
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

# Load environment variables from the .env file
load_dotenv()

# Azure AI Foundry Configuration
AZURE_FOUNDRY_ENDPOINT = os.getenv("AZURE_FOUNDRY_ENDPOINT")
AZURE_FOUNDRY_API_KEY = os.getenv("AZURE_FOUNDRY_API_KEY")
AZURE_FOUNDRY_DEPLOYMENT_NAME = os.getenv("AZURE_FOUNDRY_DEPLOYMENT_NAME")
AZURE_FOUNDRY_API_VERSION = os.getenv("AZURE_FOUNDRY_API_VERSION")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_FOUNDRY_ENDPOINT_EMBBEDINGS = os.getenv("AZURE_FOUNDRY_ENDPOINT_EMBBEDINGS")
AZURE_FOUNDRY_API_KEY_EMBEDDINGS = os.getenv("AZURE_FOUNDRY_API_KEY_EMBEDDINGS")
AZURE_FOUNDRY_DEPLOYMENT_NAME_EMBBEDINGS=os.getenv("AZURE_FOUNDRY_DEPLOYMENT_NAME_EMBBEDINGS")
AZURE_FOUNDRY_API_VERSION_EMBEDDINGS=os.getenv("AZURE_FOUNDRY_API_VERSION_EMBEDDINGS")
INDEX_NAME=os.getenv("INDEX_NAME")
INDEXER_NAME=os.getenv("INDEXER_NAME")
SEARCH_ENDPOINTS=os.getenv("SEARCH_ENDPOINTS")
SEARCH_API_KEY=os.getenv("SEARCH_API_KEY")

# Function to enrich tourist descriptions with Azure AI Foundry
def azure_foundry_enrich_batch(descriptions):
    """
    Enriches a batch of tourist descriptions using Azure AI Foundry.
    Returns a list of enriched descriptions in the same order.
    """
    url = f"{AZURE_FOUNDRY_ENDPOINT}/openai/deployments/{AZURE_FOUNDRY_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_FOUNDRY_API_VERSION}"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_FOUNDRY_API_KEY
    }
    
    prompt_content = (
        "Act as a specialized tourist assistant that enriches descriptions of places in Tenerife. "
        "Below you have a JSON array with several tourist descriptions. "
        "Return ONLY a JSON array, without any markdown format, without additional explanations, "
        "in the same order and with the same length, where each element is "
        "a more detailed and attractive version of the original description. "
        f"Descriptions: {json.dumps(descriptions, ensure_ascii=False)}"
    )
    
    data = {
        "messages": [
            {"role": "system", "content": "You are a specialized tourist assistant in Tenerife. Respond ONLY with valid JSON, no markdown."},
            {"role": "user", "content": prompt_content}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # Search for any JSON array in the response
            json_pattern = r'\[.*\]'
            json_match = re.search(json_pattern, content, re.DOTALL)
            
            if json_match:
                json_content = json_match.group()
                try:
                    enriched_descriptions = json.loads(json_content)
                    # Validate structure and length
                    if isinstance(enriched_descriptions, list) and len(enriched_descriptions) == len(descriptions):
                        return enriched_descriptions
                    else:
                        error_msg = f"The model did not return a list of the expected size. Original: {len(descriptions)}, Received: {len(enriched_descriptions) if isinstance(enriched_descriptions, list) else 'not a list'}"
                        print(error_msg)
                        return [f"Error in response format: {error_msg}" for _ in descriptions]
                except json.JSONDecodeError as e:
                    error_msg = f"Error parsing extracted JSON: {e}"
                    print(error_msg)
                    return [error_msg for _ in descriptions]
            else:
                # If no JSON array is found, try to parse the entire content
                try:
                    enriched_descriptions = json.loads(content)
                    if isinstance(enriched_descriptions, list) and len(enriched_descriptions) == len(descriptions):
                        return enriched_descriptions
                    else:
                        error_msg = "No valid JSON array found in the response"
                        print(error_msg)
                        print(f"Received content: {content[:200]}...")
                        return [f"Error in response format: {error_msg}" for _ in descriptions]
                except json.JSONDecodeError as e:
                    error_msg = f"Error parsing the complete response: {e}"
                    print(error_msg)
                    print(f"Received content: {content[:200]}...")
                    return [f"Error parsing JSON: {error_msg}" for _ in descriptions]
        else:
            error_msg = f"API error: {response.status_code} - {response.text}"
            print(error_msg)
            return [error_msg for _ in descriptions]
    except Exception as e:
        error_msg = f"Exception calling the API: {e}"
        print(error_msg)
        return [error_msg for _ in descriptions]
    
# Main function to perform category scraping, enrichment, and storage
def scrape_categories():
    print("Starting category scraping...")
    
    # Function to display a loading spinner
    def spinner():
        for char in itertools.cycle('|/-\\'):
            if not loading:
                break
            print(f'\rLoading {char}', end='', flush=True)
            time.sleep(0.1)
        print('\r', end='', flush=True)

    # Start the spinner in a separate thread
    loading = True
    spinner_thread = threading.Thread(target=spinner)
    spinner_thread.start()

    start_time = time.time()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
        })

        URL = "https://www.webtenerife.com/que-hacer/"
        page.goto(URL)
        page.wait_for_load_state("networkidle")

        elements = page.query_selector_all("div.card__col article.card")
        total_categories = len(elements)
        
        # Stop the spinner
        loading = False
        spinner_thread.join()
        
        print(f"Categories found: {total_categories}")

        # 1. Extract basic information from the categories
        categories = []
        for item in elements:
            try:
                category_name = item.query_selector("h6.card__title.heading").inner_text().strip()
            except:
                category_name = "No name"

            try:
                description = item.query_selector("div.card__description").inner_text().strip()
            except:
                description = "No description"

            try:
                link = item.query_selector("a").get_attribute("href")
                if not link.startswith("http"):
                    link = urljoin(URL, link)
            except:
                link = "No link"

            categories.append({
                "name": category_name,
                "description": description,
                "link": link
            })

    print(f"Scraping phase completed. Obtained {len(categories)} categories.")
    print("Starting description enrichment with Azure AI Foundry...")

    # 2. Batch processing with Azure AI Foundry
    batch_size = 5  # Reduced to avoid issues with large responses
    total = len(categories)
    enriched_results = []

    # Calculate total number of batches
    total_batches = math.ceil(total / batch_size)
    
    for batch_index in range(total_batches):
        batch_start_time = time.time()
        
        # Get the current batch
        start_idx = batch_index * batch_size
        end_idx = min(start_idx + batch_size, total)
        current_batch = categories[start_idx:end_idx]
        
        # Extract only the descriptions for enrichment
        batch_descriptions = [item["description"] for item in current_batch]
        
        print(f"Processing batch {batch_index + 1}/{total_batches} ({len(batch_descriptions)} descriptions)...")
        
        # Call Azure AI Foundry to enrich the descriptions
        enriched_descriptions = azure_foundry_enrich_batch(batch_descriptions)
        
        # Check for errors in the batch
        has_errors = any("Error" in str(desc) for desc in enriched_descriptions)
        if has_errors:
            print(f"Errors found in batch {batch_index + 1}. Continuing anyway...")
        
        # Add the results
        enriched_results.extend(enriched_descriptions)
        
        # Calculate time statistics
        batch_time = time.time() - batch_start_time
        elapsed_time = time.time() - start_time
        remaining_batches = total_batches - (batch_index + 1)
        estimated_remaining_time = remaining_batches * batch_time
        progress_percent = ((batch_index + 1) / total_batches) * 100
        
        # Show progress
        print(f"Batch {batch_index + 1}/{total_batches} completed in {batch_time:.2f}s")
        print(f"Progress: {progress_percent:.2f}%")
        print(f"Elapsed time: {elapsed_time:.2f}s")
        print(f"Estimated remaining time: {estimated_remaining_time:.2f}s")
        print("-" * 60)

    # 3. Integrate the enriched descriptions with the original data
    for idx, category in enumerate(categories):
        if idx < len(enriched_results):
            category["detailed_description"] = enriched_results[idx]
        else:
            category["detailed_description"] = "Could not generate a detailed description"

    # 4. Save the results to a JSON file in the current directory
    output_file = os.path.join(os.getcwd(), "categories_tenerife.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=4)

    total_time = time.time() - start_time
    print(f"Process completed in {total_time:.2f}s")
    print(f"Results saved in '{output_file}'")

# Function to upload a file to Azure Blob Storage
def upload_to_blob_storage(file_path, container_name, blob_name):
    try:
        # Create the blob service client
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        
        # Check if the container exists, if not, create it
        try:
            container_client = blob_service_client.get_container_client(container_name)
            # Check if it exists by trying to get its properties
            container_client.get_container_properties()
        except Exception:
            print(f"The container '{container_name}' does not exist. Creating it...")
            blob_service_client.create_container(container_name)
            print(f"Container '{container_name}' created successfully.")
        
        # Get the blob client and upload the file
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        print(f"Uploading file '{file_path}' to Azure Blob Storage as JSON...")
        
        # Read the JSON as a string
        with open(file_path, "r", encoding="utf-8") as json_file:
            json_content = json_file.read()
            
        # Upload the content as a string and specify the content_type as application/json
        blob_client.upload_blob(
            json_content, 
            overwrite=True, 
            content_type="application/json"
        )
        
        print(f"JSON file '{file_path}' successfully uploaded to Blob Storage")
        print(f"Blob URL: {blob_client.url}")
    except Exception as e:
        print(f"Error uploading the file to Blob Storage: {e}")

# Function to save data to Azure Table Storage
def save_to_table_storage(json_file_path, table_name):
    try:
        print(f"Saving data from file '{json_file_path}' to Azure Table Storage...")

        # Create the table service client
        table_service_client = TableServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        
        # Check if the table exists
        table_exists = False
        try:
            table_client = table_service_client.get_table_client(table_name)
            # Check if it exists by trying to access its properties
            properties = table_client.get_table_properties()
            table_exists = True
            print(f"Table '{table_name}' found.")
            
            # If the table exists, delete all existing records
            print(f"Deleting all existing records in table '{table_name}'...")
            
            # Query all entities in the table
            entities = table_client.list_entities()
            
            # Counter for tracking deletion
            deleted = 0
            for entity in entities:
                # Delete the entity
                table_client.delete_entity(partition_key=entity["PartitionKey"], row_key=entity["RowKey"])
                deleted += 1
                if deleted % 20 == 0:
                    print(f"Records deleted: {deleted}")
                    
            print(f"{deleted} records deleted from the table.")
            
        except Exception as e:
            table_exists = False
            print(f"Table '{table_name}' does not exist: {e}")
        
        # If the table does not exist or all its content has been deleted, create/use it
        if not table_exists:
            try:
                table_client = table_service_client.create_table(table_name)
                print(f"Table '{table_name}' created successfully.")
            except ResourceExistsError:
                table_client = table_service_client.get_table_client(table_name)
                print(f"Table '{table_name}' already exists and is empty. Using the existing table.")
        
        # Load data from the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            categories = json.load(f)
        
        total_entities = len(categories)
        print(f"Processing {total_entities} records for Table Storage...")
        
        # Add each category as an entity in the table
        added = 0
        errors = 0
        
        for idx, category in enumerate(categories):
            # Get category fields
            name = str(category['name'])[:1024] if 'name' in category else ""
            description = str(category['description'])[:8000] if 'description' in category else ""
            link = str(category.get('link', ''))[:2048]
            detailed_description = str(category.get('detailed_description', ''))[:16000]
            
            # Verify that all required fields have values
            if not (name and description and link and detailed_description):
                print(f"Record skipped due to empty fields: {category}")
                errors += 1
                continue
            
            # Generate a unique random ID using uuid4
            unique_id = str(uuid.uuid4())
            
            # Create a unique RowKey based on the generated ID
            row_key = unique_id
            
            # Get embedding if present, otherwise leave it empty
            embedding_json = ""
            if 'embedding' in category and category['embedding']:
                # The embedding is already in the JSON, just convert it to a string
                embedding_json = json.dumps(category['embedding'])
            
            # Prepare the entity
            entity = {
                "PartitionKey": "categories",  # All categories in the same partition
                "RowKey": row_key,             # Random ID as row key
                "id": unique_id,               # Store the same ID as a column for easier queries
                "name": name,
                "description": description,
                "link": link,
                "detailed_description": detailed_description,
                "contentVector": embedding_json  # Save the embedding as a JSON string
            }
            
            try:
                # Save the entity in the table
                table_client.create_entity(entity)
                added += 1
                
                # Show progress every 10 records
                if added % 10 == 0 or added == total_entities:
                    print(f"Progress: {added}/{total_entities} ({(added/total_entities*100):.1f}%)")
                
            except Exception as e:
                print(f"Error saving entity with ID {unique_id}: {e}")
                print(f"Problematic data: {name}")
                errors += 1
        
        print(f"Process completed. Records added: {added}, errors: {errors}")
        
    except Exception as e:
        print(f"General error saving to Table Storage: {e}")

# Function to generate embeddings for each category
def generate_embeddings(json_file_path):
    """
    Generates embeddings for each category using the text-embedding-3-small model from Azure OpenAI.
    Creates embeddings by combining the name and detailed description of each category.
    """
    print("Starting embedding generation for categories...")
    
    # Load data from the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        categories = json.load(f)
    
    # URL of the Azure OpenAI embeddings endpoint
    embedding_url = f"{AZURE_FOUNDRY_ENDPOINT_EMBBEDINGS}/openai/deployments/{AZURE_FOUNDRY_DEPLOYMENT_NAME_EMBBEDINGS}/embeddings?api-version={AZURE_FOUNDRY_API_VERSION_EMBEDDINGS}"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_FOUNDRY_API_KEY_EMBEDDINGS
    }
    
    # List to store the results
    embeddings_results = []
    
    # Process each category
    total = len(categories)
    print(f"Generating embeddings for {total} categories...")
    
    for idx, category in enumerate(categories):
        # Combine name and detailed description for the embedding
        name = category.get('name', 'No name')
        description = category.get('description', '')
        detailed_description = category.get('detailed_description', '')        
        link = category.get('link', '')

        # Full text for generating the embedding (combination of name and detailed description)
        text = f"{name}. {detailed_description}"
        
        # Data for the embedding request
        data = {
            "input": text,
            "encoding_format": "float"  # Encoding format for the embeddings
        }
        
        try:
            # Call the API to generate the embedding
            response = requests.post(embedding_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                embedding = result["data"][0]["embedding"]
                
                # Add the embedding to the result
                embedding_result = {
                    "id": name,  # Use the name as the identifier
                    "name": name,
                    "description": description,
                    "link": link,
                    "detailed_description": detailed_description,
                    "embedding": embedding
                }
                
                embeddings_results.append(embedding_result)
                print(f"[{idx+1}/{total}] Embedding generated for: {name}")
            else:
                print(f"Error generating embedding for '{name}': {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Exception generating embedding for '{name}': {e}")
        
        # Short pause to avoid rate limits
        if idx % 10 == 9:
            print(f"Short pause after processing 10 items ({idx+1}/{total})...")
            time.sleep(1)
    
    # Save the generated embeddings to a JSON file in the current directory
    embeddings_file = os.path.join(os.getcwd(), "categories_embeddings.json")
    with open(embeddings_file, "w", encoding="utf-8") as f:
        json.dump(embeddings_results, f, ensure_ascii=False, indent=4)
    
    print(f"{len(embeddings_results)} embeddings generated. Saved in '{embeddings_file}'")
    
    # Optionally, also save the embeddings to Azure Blob Storage
    container_name = "tenerifeactivities"
    blob_name = "categories_embeddings.json"
    upload_to_blob_storage(embeddings_file, container_name, blob_name)
    
    return embeddings_results

# Function to find similar categories to a given query
def find_similar_categories(query_text, embeddings_data, top_k=3):
    """
    Finds categories similar to a given query using cosine similarity.
    
    Args:
        query_text: Query text
        embeddings_data: List of dictionaries with embeddings
        top_k: Number of similar results to return
    
    Returns:
        List of the top_k most similar items
    """
    # Generate embedding for the query
    embedding_url = f"{AZURE_FOUNDRY_ENDPOINT}openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_FOUNDRY_API_KEY
    }
    
    data = {
        "input": query_text,
        "encoding_format": "float"
    }
    
    try:
        response = requests.post(embedding_url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            query_embedding = result["data"][0]["embedding"]
            
            # Calculate cosine similarity with all embeddings
            similarities = []
            for item in embeddings_data:
                embedding = item["embedding"]
                similarity = 1 - spatial.distance.cosine(query_embedding, embedding)
                similarities.append((item, similarity))
            
            # Sort by descending similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return the top_k results
            return [(item, score) for item, score in similarities[:top_k]]
        else:
            print(f"Error generating embedding for the query: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Exception generating embedding for the query: {e}")
        return []

# Function to delete all documents from an Azure AI Search index
def delete_all_documents_from_search_index(index_name, search_endpoint, api_key):
    """
    Deletes all documents from the specified Azure AI Search index.
    
    Args:
        index_name: The name of the Azure AI Search index
        search_endpoint: The Azure AI Search service endpoint URL
        api_key: The API key for the Azure AI Search service
    
    Returns:
        True if successful, False otherwise
    """
    print(f"Deleting all documents from Azure AI Search index '{index_name}'...")
    
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    try:
        # First, get the count of documents in the index
        count_url = f"{search_endpoint}/indexes/{index_name}/docs/$count?api-version=2023-07-01-Preview"
        count_response = requests.get(count_url, headers=headers)
        
        if count_response.status_code != 200:
            print(f"Error getting document count: {count_response.status_code} - {count_response.text}")
            return False
            
        document_count = int(count_response.text)
        print(f"Found {document_count} documents in the index.")
        
        if document_count == 0:
            print("No documents to delete.")
            return True
            
        # Use a batch delete operation with a wildcard pattern
        delete_url = f"{search_endpoint}/indexes/{index_name}/docs/index?api-version=2023-07-01-Preview"
        
        # The batch size for delete operations
        batch_size = 1000
        batches = math.ceil(document_count / batch_size)
        
        for batch in range(batches):
            print(f"Processing batch {batch + 1}/{batches}...")
            
            # Get a batch of document keys
            search_url = f"{search_endpoint}/indexes/{index_name}/docs/search?api-version=2023-07-01-Preview"
            search_payload = {
                "search": "*",
                "select": "id",
                "top": batch_size,
                "skip": batch * batch_size
            }
            
            search_response = requests.post(search_url, headers=headers, json=search_payload)
            
            if search_response.status_code != 200:
                print(f"Error searching for documents: {search_response.status_code} - {search_response.text}")
                return False
                
            result = search_response.json()
            document_keys = [doc["id"] for doc in result.get("value", [])]
            
            if not document_keys:
                break
                
            # Create a batch delete action for each document
            actions = [{"@search.action": "delete", "id": key} for key in document_keys]
            
            # Submit delete request
            delete_payload = {"value": actions}
            delete_response = requests.post(delete_url, headers=headers, json=delete_payload)
            
            if delete_response.status_code not in [200, 207]:
                print(f"Error deleting documents: {delete_response.status_code} - {delete_response.text}")
                return False
                
            print(f"Deleted {len(document_keys)} documents in batch {batch + 1}.")
            
        print(f"Successfully deleted all documents from index '{index_name}'.")
        return True
        
    except Exception as e:
        print(f"Exception deleting documents from search index: {e}")
        return False

# Function to activate an Azure AI Search indexer
def run_azure_search_indexer(indexer_name, search_endpoint, api_key):
    """
    Triggers an Azure AI Search indexer to run on demand.
    
    Args:
        indexer_name: The name of the Azure AI Search indexer to run
        search_endpoint: The Azure AI Search service endpoint URL
        api_key: The API key for the Azure AI Search service
    
    Returns:
        True if the indexer was successfully triggered, False otherwise
    """
    print(f"Running Azure AI Search indexer '{indexer_name}'...")
    
    # Construct the URL for running the indexer
    run_url = f"{search_endpoint}/indexers/{indexer_name}/run?api-version=2023-07-01-Preview"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    try:
        # Send the request to run the indexer
        response = requests.post(run_url, headers=headers)
        
        if response.status_code == 202:  # 202 Accepted is the expected response
            print(f"Indexer '{indexer_name}' started successfully.")
            
            # To monitor the indexer status
            status_url = f"{search_endpoint}/indexers/{indexer_name}/status?api-version=2023-07-01-Preview"
            status_response = requests.get(status_url, headers=headers)
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data.get("lastResult", {}).get("status", "Unknown")
                print(f"Indexer status: {status}")
                
                # You could add more detailed status information here if needed
                
            return True
        else:
            print(f"Error running indexer: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception running indexer: {e}")
        return False

# Function to run an Azure AI Search indexer
def wait_for_indexer_completion(indexer_name, search_endpoint, api_key, timeout_seconds=300, poll_interval_seconds=10):
    """
    Runs an Azure AI Search indexer and waits for it to complete.
    
    Args:
        indexer_name: The name of the Azure AI Search indexer
        search_endpoint: The Azure AI Search service endpoint URL
        api_key: The API key for the Azure AI Search service
        timeout_seconds: Maximum time to wait (in seconds)
        poll_interval_seconds: Time between status checks (in seconds)
    
    Returns:
        True if the indexer completed successfully, False otherwise
    """
    print(f"Running indexer '{indexer_name}' and waiting for completion (timeout: {timeout_seconds}s)...")
    
    # Start the indexer
    if not run_azure_search_indexer(indexer_name, search_endpoint, api_key):
        return False
    
    # Wait for completion
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        # Get indexer status
        status_data = get_azure_search_indexer_status(indexer_name, search_endpoint, api_key)
        
        if not status_data:
            print("Failed to get indexer status.")
            return False
        
        status = status_data.get("lastResult", {}).get("status", "")
        
        # Check if indexer has completed
        if status in ["success", "reset"]:
            print(f"Indexer '{indexer_name}' completed successfully.")
            return True
        elif status in ["error", "transientError"]:
            error_message = status_data.get("lastResult", {}).get("errorMessage", "Unknown error")
            print(f"Indexer failed with status '{status}': {error_message}")
            return False
        
        # If still running, wait and check again
        print(f"Indexer status: {status}. Checking again in {poll_interval_seconds} seconds...")
        time.sleep(poll_interval_seconds)
    
    # If we get here, we've timed out
    print(f"Timeout of {timeout_seconds}s exceeded while waiting for indexer to complete.")
    return False

# Function to get the status of an Azure AI Search indexer
def get_azure_search_indexer_status(indexer_name, search_endpoint, api_key):
    """
    Gets the status of an Azure AI Search indexer.
    
    Args:
        indexer_name: The name of the Azure AI Search indexer
        search_endpoint: The Azure AI Search service endpoint URL
        api_key: The API key for the Azure AI Search service
    
    Returns:
        The indexer status information as a dictionary, or None if an error occurs
    """
    print(f"Getting status of Azure AI Search indexer '{indexer_name}'...")
    
    status_url = f"{search_endpoint}/indexers/{indexer_name}/status?api-version=2023-07-01-Preview"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    try:
        response = requests.get(status_url, headers=headers)
        
        if response.status_code == 200:
            status_data = response.json()
            status = status_data.get("lastResult", {}).get("status", "Unknown")
            print(f"Indexer '{indexer_name}' status: {status}")
            
            # Print more detailed information
            if "lastResult" in status_data:
                last_result = status_data["lastResult"]
                
                if "errorMessage" in last_result:
                    print(f"Error message: {last_result['errorMessage']}")
                
                if "itemsProcessed" in last_result:
                    print(f"Items processed: {last_result['itemsProcessed']}")
                    
                if "itemsFailed" in last_result:
                    print(f"Items failed: {last_result['itemsFailed']}")
            
            return status_data
        else:
            print(f"Error getting indexer status: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Exception getting indexer status: {e}")
        return None

if __name__ == "__main__":
    
    output_file = "categories_tenerife.json"    
    output_file_embbedings = "categories_embeddings.json"
    container_name = "actividadestenerife"
    
    # Execute the complete process
    print("\033[94mStarting category scraping...\033[0m")
    scrape_categories()
    
    print("\033[92mGenerating embeddings...\033[0m")
    generate_embeddings(output_file)
    
    print("\033[93mSaving to Azure Table Storage...\033[0m")
    save_to_table_storage(output_file_embbedings, "ActividadesTenerifeCopilot")

    print("\033[95mRunning Azure AI Search indexer...\033[0m")
    wait_for_indexer_completion(INDEXER_NAME, SEARCH_ENDPOINTS, SEARCH_API_KEY)

    #print("\033[96mDeleting all documents from Azure AI Search index...\033[0m")
    #delete_all_documents_from_search_index( INDEX_NAME, SEARCH_ENDPOINTS, SEARCH_API_KEY)
    

