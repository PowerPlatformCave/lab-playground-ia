import os
import json
import time
import math
import re
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin
import requests
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.data.tables import TableServiceClient
from azure.core.exceptions import ResourceExistsError

# Configuración Azure AI Foundry
AZURE_FOUNDRY_ENDPOINT = os.getenv("AZURE_FOUNDRY_ENDPOINT", "https://ai-semantic-enrichment630206263709.openai.azure.com/")
AZURE_FOUNDRY_API_KEY = os.getenv("AZURE_FOUNDRY_API_KEY", "DXyRUF4LP3xS0wgMoCFUJirTsGhdTTFMFUtvb323rM48zTRWNRIMJQQJ99BCACfhMk5XJ3w3AAAAACOGxvuT")
AZURE_FOUNDRY_DEPLOYMENT_NAME = os.getenv("AZURE_FOUNDRY_DEPLOYMENT_NAME", "gpt-4o-mini")

# Función para enriquecer descripciones turísticas con Azure AI Foundry
def azure_foundry_enrich_batch(descriptions):
    """
    Enriquece un lote de descripciones turísticas mediante Azure AI Foundry.
    Devuelve una lista de descripciones enriquecidas en el mismo orden.
    """
    #url = f"{AZURE_FOUNDRY_ENDPOINT}openai/deployments/{AZURE_FOUNDRY_DEPLOYMENT_NAME}/chat/completions?api-version=2023-05-15"
    url= "https://ai-semantic-enrichment630206263709.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-10-21"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_FOUNDRY_API_KEY
    }
    
    prompt_content = (
        "Actúa como un asistente turístico especializado que enriquece descripciones de lugares en Tenerife. "
        "A continuación tienes un array JSON con varias descripciones turísticas. "
        "Devuélveme SOLO un array JSON, sin ningún formato markdown, sin explicaciones adicionales, "
        "en el mismo orden y con la misma longitud, donde cada elemento sea "
        "una versión más detallada y atractiva de la descripción original. "
        f"Descripciones: {json.dumps(descriptions, ensure_ascii=False)}"
    )
    
    data = {
        "messages": [
            {"role": "system", "content": "Eres un asistente turístico especializado en Tenerife. Responde ÚNICAMENTE con JSON válido, sin markdown."},
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
            
            # Buscar cualquier array JSON en la respuesta
            json_pattern = r'\[.*\]'
            json_match = re.search(json_pattern, content, re.DOTALL)
            
            if json_match:
                json_content = json_match.group()
                try:
                    enriched_descriptions = json.loads(json_content)
                    # Validar estructura y longitud
                    if isinstance(enriched_descriptions, list) and len(enriched_descriptions) == len(descriptions):
                        return enriched_descriptions
                    else:
                        error_msg = f"El modelo no devolvió una lista del tamaño esperado. Original: {len(descriptions)}, Recibido: {len(enriched_descriptions) if isinstance(enriched_descriptions, list) else 'no es lista'}"
                        print(error_msg)
                        return [f"Error en el formato de respuesta: {error_msg}" for _ in descriptions]
                except json.JSONDecodeError as e:
                    error_msg = f"Error al parsear el JSON extraído: {e}"
                    print(error_msg)
                    return [error_msg for _ in descriptions]
            else:
                # Si no se encuentra un array JSON, intentamos parsear todo el contenido
                try:
                    enriched_descriptions = json.loads(content)
                    if isinstance(enriched_descriptions, list) and len(enriched_descriptions) == len(descriptions):
                        return enriched_descriptions
                    else:
                        error_msg = "No se encontró un array JSON válido en la respuesta"
                        print(error_msg)
                        print(f"Contenido recibido: {content[:200]}...")
                        return [f"Error en el formato de respuesta: {error_msg}" for _ in descriptions]
                except json.JSONDecodeError as e:
                    error_msg = f"Error al parsear la respuesta completa: {e}"
                    print(error_msg)
                    print(f"Contenido recibido: {content[:200]}...")
                    return [f"Error al parsear JSON: {error_msg}" for _ in descriptions]
        else:
            error_msg = f"Error en la API: {response.status_code} - {response.text}"
            print(error_msg)
            return [error_msg for _ in descriptions]
    except Exception as e:
        error_msg = f"Excepción al llamar a la API: {e}"
        print(error_msg)
        return [error_msg for _ in descriptions]

# Función principal para realizar el scraping de categorías
def scrape_categories():
    print("Iniciando scraping de categorías...")
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
        print(f"Categorías encontradas: {total_categories}")

        # 1. Extraer información básica de las categorías
        categories = []
        for item in elements:
            try:
                category_name = item.query_selector("h6.card__title.heading").inner_text().strip()
            except:
                category_name = "Sin nombre"

            try:
                description = item.query_selector("div.card__description").inner_text().strip()
            except:
                description = "Sin descripción"

            try:
                link = item.query_selector("a").get_attribute("href")
                if not link.startswith("http"):
                    link = urljoin(URL, link)
            except:
                link = "Sin enlace"

            categories.append({
                "nombre": category_name,
                "descripcion": description,
                "link": link
            })

    print(f"Fase de scraping completada. Obtenidas {len(categories)} categorías.")
    print("Iniciando enriquecimiento de descripciones con Azure AI Foundry...")

    # 2. Procesamiento por lotes con Azure AI Foundry
    batch_size = 5  # Reducido para evitar problemas con respuestas grandes
    total = len(categories)
    enriched_results = []

    # Calcular número total de lotes
    total_batches = math.ceil(total / batch_size)
    
    for batch_index in range(total_batches):
        batch_start_time = time.time()
        
        # Obtener el lote actual
        start_idx = batch_index * batch_size
        end_idx = min(start_idx + batch_size, total)
        current_batch = categories[start_idx:end_idx]
        
        # Extraer solo las descripciones para el enriquecimiento
        batch_descriptions = [item["descripcion"] for item in current_batch]
        
        print(f"Procesando lote {batch_index + 1}/{total_batches} ({len(batch_descriptions)} descripciones)...")
        
        # Llamar a Azure AI Foundry para enriquecer las descripciones
        enriched_descriptions = azure_foundry_enrich_batch(batch_descriptions)
        
        # Verificar si hay errores en el batch
        has_errors = any("Error" in str(desc) for desc in enriched_descriptions)
        if has_errors:
            print(f"Se encontraron errores en el lote {batch_index + 1}. Continuando de todos modos...")
        
        # Agregar los resultados
        enriched_results.extend(enriched_descriptions)
        
        # Calcular estadísticas de tiempo
        batch_time = time.time() - batch_start_time
        elapsed_time = time.time() - start_time
        remaining_batches = total_batches - (batch_index + 1)
        estimated_remaining_time = remaining_batches * batch_time
        progress_percent = ((batch_index + 1) / total_batches) * 100
        
        # Mostrar progreso
        print(f"Lote {batch_index + 1}/{total_batches} completado en {batch_time:.2f}s")
        print(f"Progreso: {progress_percent:.2f}%")
        print(f"Tiempo transcurrido: {elapsed_time:.2f}s")
        print(f"Tiempo estimado restante: {estimated_remaining_time:.2f}s")
        print("-" * 60)

    # 3. Integrar las descripciones enriquecidas con los datos originales
    for idx, category in enumerate(categories):
        if idx < len(enriched_results):
            category["descripcion_detallada"] = enriched_results[idx]
        else:
            category["descripcion_detallada"] = "No se pudo generar una descripción detallada"

    # 4. Guardar los resultados en un archivo JSON
    output_file = "categorias_tenerife.json"
    #with open(output_file, "w", encoding="utf-8") as f:
    #    json.dump(categories, f, ensure_ascii=False, indent=4)

    total_time = time.time() - start_time
    print(f"Proceso completado en {total_time:.2f}s")
    print(f"Resultados guardados en '{output_file}'")

    # Subir el archivo JSON a Azure Blob Storage
    container_name = "actividadestenerife"
    blob_name = "categorias_tenerife.json"

# Función para subir un archivo a Azure Blob Storage    
def upload_to_blob_storage(file_path, container_name, blob_name):

    try:
        # Cadena de conexión correcta (simplificada y sin duplicados)
        connect_str = os.getenv(
            "AZURE_STORAGE_CONNECTION_STRING", 
            "DefaultEndpointsProtocol=https;AccountName=stcopilotaie772067053946;AccountKey=w/xJV44jErmpNclpkeapSIzbQwBtTd1htVEhRVlBBvMMCaVKMqCb52BaC1UypEctpFaQmRAoMKyf+AStAHsfkw==;EndpointSuffix=core.windows.net"
        )
        
        # Crear el cliente de servicio blob
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        
        # Verificar si el contenedor existe, si no, crearlo
        try:
            container_client = blob_service_client.get_container_client(container_name)
            # Comprobar si existe intentando obtener sus propiedades
            container_client.get_container_properties()
        except Exception:
            print(f"El contenedor '{container_name}' no existe. Creándolo...")
            blob_service_client.create_container(container_name)
            print(f"Contenedor '{container_name}' creado correctamente.")
        
        # Obtener el cliente blob y subir el archivo
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        print(f"Subiendo archivo '{file_path}' a Azure Blob Storage como JSON...")
        
        # Lee el JSON como string
        with open(file_path, "r", encoding="utf-8") as json_file:
            json_content = json_file.read()
            
        # Sube el contenido como string y especifica el content_type como application/json
        blob_client.upload_blob(
            json_content, 
            overwrite=True, 
            content_type="application/json"
        )
        
        print(f"Archivo JSON '{file_path}' subido exitosamente a Blob Storage")
        print(f"URL del blob: {blob_client.url}")
    except Exception as e:
        print(f"Error al subir el archivo a Blob Storage: {e}")

# Función para guardar los datos en Azure Table Storage
def save_to_table_storage(json_file_path, table_name):
    try:
        print(f"Guardando datos del archivo '{json_file_path}' en Azure Table Storage...")
        
        # Obtener la cadena de conexión de la variable de entorno
        connect_str = os.getenv(
            "AZURE_STORAGE_CONNECTION_STRING", 
            "DefaultEndpointsProtocol=https;AccountName=stcopilotaie772067053946;AccountKey=w/xJV44jErmpNclpkeapSIzbQwBtTd1htVEhRVlBBvMMCaVKMqCb52BaC1UypEctpFaQmRAoMKyf+AStAHsfkw==;EndpointSuffix=core.windows.net"
        )
        
        # Crear el cliente del servicio de tablas
        table_service_client = TableServiceClient.from_connection_string(connect_str)
        
        # Verificar si la tabla existe
        table_exists = False
        try:
            table_client = table_service_client.get_table_client(table_name)
            # Comprobar si existe intentando acceder a ella
            properties = table_client.get_table_properties()
            table_exists = True
            print(f"Tabla '{table_name}' encontrada.")
            
            # Si la tabla existe, borrar todos los registros
            print(f"Eliminando todos los registros existentes en la tabla '{table_name}'...")
            
            # Consultar todas las entidades de la tabla
            entities = table_client.list_entities()
            
            # Contador para seguimiento de eliminación
            deleted = 0
            for entity in entities:
                # Eliminar la entidad
                table_client.delete_entity(partition_key=entity["PartitionKey"], row_key=entity["RowKey"])
                deleted += 1
                if deleted % 20 == 0:
                    print(f"Registros eliminados: {deleted}")
                    
            print(f"Se eliminaron {deleted} registros de la tabla.")
            
        except Exception as e:
            table_exists = False
            print(f"La tabla '{table_name}' no existe: {e}")
        
        # Si la tabla no existe o se ha borrado todo su contenido, crearla/usarla
        if not table_exists:
            try:
                table_client = table_service_client.create_table(table_name)
                print(f"Tabla '{table_name}' creada correctamente.")
            except ResourceExistsError:
                table_client = table_service_client.get_table_client(table_name)
                print(f"Tabla '{table_name}' ya existe y está vacía. Se usará la tabla existente.")
        
        # Cargar datos del archivo JSON
        with open(json_file_path, 'r', encoding='utf-8') as f:
            categories = json.load(f)
        
        total_entities = len(categories)
        print(f"Procesando {total_entities} registros para Table Storage...")
        
        # Añadir cada categoría como una entidad en la tabla
        added = 0
        errors = 0
        
        # Importar uuid para generar IDs aleatorios
        import uuid
        
        for idx, category in enumerate(categories):
            # Generar un ID único aleatorio usando uuid4
            unique_id = str(uuid.uuid4())
            
            # Crear un RowKey único basado en el ID generado
            row_key = unique_id
            
            # Obtener campos de la categoría
            nombre = str(category['nombre'])[:1024] if 'nombre' in category else "Sin nombre"
            descripcion = str(category['descripcion'])[:8000] if 'descripcion' in category else ""
            link = str(category.get('link', ''))[:2048]
            descripcion_detallada = str(category.get('descripcion_detallada', ''))[:16000]
            
            # Obtener embedding si está presente, sino dejarlo vacío
            embedding_json = ""
            if 'embedding' in category and category['embedding']:
                # El embedding ya viene en el JSON, solo lo convertimos a cadena
                embedding_json = json.dumps(category['embedding'])
            
            # Preparar la entidad
            entity = {
                "PartitionKey": "categories",  # Todas las categorías en la misma partición
                "RowKey": row_key,             # ID aleatorio como clave de fila
                "id": unique_id,               # Almacenar el mismo ID como columna para facilitar consultas
                "nombre": nombre,
                "descripcion": descripcion,
                "link": link,
                "descripcion_detallada": descripcion_detallada,
                "contentVector": embedding_json  # Guardar el embedding como JSON string
            }
            
            try:
                # Guardar la entidad en la tabla
                table_client.create_entity(entity)
                added += 1
                
                # Mostrar progreso cada 10 registros
                if added % 10 == 0 or added == total_entities:
                    print(f"Progreso: {added}/{total_entities} ({(added/total_entities*100):.1f}%)")
                
            except Exception as e:
                print(f"Error al guardar la entidad con ID {unique_id}: {e}")
                print(f"Datos problemáticos: {nombre}")
                errors += 1
        
        print(f"Proceso completado. Registros añadidos: {added}, errores: {errors}")
        
    except Exception as e:
        print(f"Error general al guardar en Table Storage: {e}")

# Función para generar embeddings para cada categoría
def generate_embeddings(json_file_path):
    """
    Genera embeddings para cada categoría usando el modelo text-embedding-3-small de Azure OpenAI.
    Crea embeddings combinando el nombre y la descripción detallada de cada categoría.
    """
    print("Iniciando generación de embeddings para las categorías...")
    
    # Cargar los datos del archivo JSON
    with open(json_file_path, 'r', encoding='utf-8') as f:
        categories = json.load(f)
    
    # URL del endpoint de embeddings de Azure OpenAI
    #embedding_url = f"{AZURE_FOUNDRY_ENDPOINT}openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15"
    embedding_url="https://openai-copilot-studio-usa.openai.azure.com/openai/deployments/text-embedding-3-small-2/embeddings?api-version=2023-05-15"
    headers = {
        "Content-Type": "application/json",
        "api-key": "39NsfCaMzHxcl80NdXWAK5qSfCZhmUAstXVAMcL3Od3mbuM77Gg6JQQJ99BCACYeBjFXJ3w3AAABACOGCWFI"
    }
    
    # Lista para almacenar los resultados
    embeddings_results = []
    
    # Procesar cada categoría
    total = len(categories)
    print(f"Generando embeddings para {total} categorías...")
    
    for idx, category in enumerate(categories):
        # Combinar nombre y descripción detallada para el embedding
        nombre = category.get('nombre', 'Sin nombre')
        descripcion_detallada = category.get('descripcion_detallada', '')
        
        # Texto completo para generar el embedding (combinación de nombre y descripción)
        texto = f"{nombre}. {descripcion_detallada}"
        
        # Datos para la solicitud de embeddings
        data = {
            "input": texto,
            "encoding_format": "float"  # Formato de codificación de los embeddings
        }
        
        try:
            # Llamar a la API para generar el embedding
            response = requests.post(embedding_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                embedding = result["data"][0]["embedding"]
                
                # Añadir el embedding al resultado
                embedding_result = {
                    "id": nombre,  # Usar el nombre como identificador
                    "nombre": nombre,
                    "descripcion_detallada": descripcion_detallada,
                    "embedding": embedding
                }
                
                embeddings_results.append(embedding_result)
                print(f"[{idx+1}/{total}] Embedding generado para: {nombre}")
            else:
                print(f"Error al generar embedding para '{nombre}': {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Excepción al generar embedding para '{nombre}': {e}")
        
        # Pequeña pausa para evitar límites de tasa
        if idx % 10 == 9:
            print(f"Pausa breve después de procesar 10 elementos ({idx+1}/{total})...")
            time.sleep(1)
    
    # Guardar los embeddings generados en un archivo JSON
    embeddings_file = "categorias_embeddings.json"
    with open(embeddings_file, "w", encoding="utf-8") as f:
        json.dump(embeddings_results, f, ensure_ascii=False, indent=4)
    
    print(f"Se generaron {len(embeddings_results)} embeddings. Guardados en '{embeddings_file}'")
    
    # Opcionalmente, guardar también los embeddings en Azure Blob Storage
    container_name = "actividadestenerife"
    blob_name = "categorias_embeddings.json"
    upload_to_blob_storage(embeddings_file, container_name, blob_name)
    
    return embeddings_results

# Función para encontrar categorías similares a una consulta dada
def find_similar_categories(query_text, embeddings_data, top_k=3):
    """
    Encuentra categorías similares a una consulta dada usando similitud de coseno.
    
    Args:
        query_text: Texto de consulta
        embeddings_data: Lista de diccionarios con embeddings
        top_k: Número de resultados similares a devolver
    
    Returns:
        Lista de los top_k elementos más similares
    """
    # Generar embedding para la consulta
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
            
            # Calcular similitud de coseno con todos los embeddings
            from scipy import spatial
            
            similarities = []
            for item in embeddings_data:
                embedding = item["embedding"]
                similarity = 1 - spatial.distance.cosine(query_embedding, embedding)
                similarities.append((item, similarity))
            
            # Ordenar por similitud descendente
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Devolver los top_k resultados
            return [(item, score) for item, score in similarities[:top_k]]
        else:
            print(f"Error al generar embedding para la consulta: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Excepción al generar embedding para la consulta: {e}")
        return []

if __name__ == "__main__":
    
    output_file = "categorias_tenerife.json"
    container_name = "actividadestenerife"
    blob_name = "categorias_tenerife.json"
    output_file_embbedings = "categorias_embeddings.json"
    
    # Ejecutar el proceso completo
    scrape_categories()
    generate_embeddings(output_file)
    upload_to_blob_storage(output_file_embbedings, container_name, blob_name);    
    save_to_table_storage(output_file_embbedings, "ActividadesTenerifeCopilot");
    

