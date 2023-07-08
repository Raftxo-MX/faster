from fastapi import FastAPI
import pandas as pd
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from datetime import datetime
import scipy.sparse as sp


app = FastAPI()


# cargo el dataset limpio
df = pd.read_csv('./datos_limpios.csv')


@app.get('/')
def index():   
    return {   'message': '尺闩ﾁ七〤ㄖ ﾁ闩丂セ 闩尸讠',
           'description': 'Implementación de la API con FastAPI en Render',
          'organization': 'soyHENRY.com',
               'carreer': 'soyHenry bootcamp DATA SCIENCE',
               'project': 'PI01_MLops',
                'cohort': 'DTS-12',
               'student': 'Rafal Janusz Wysocki',
           'mail/github': 'raftxo.mx@gmail.com', 
           }

@app.get('/peliculas_idioma/{idioma}')
def peliculas_idioma(idioma:str):
    '''
    Ingresa el idioma en formato abreviado conforme la norma internacional ISO 639-1. 
    Los idiomas más populares son:
        "en" corresponde a inglés (English).
        "fr" corresponde a francés (French).
        "it" corresponde a italiano (Italian).
        "ja" corresponde a japonés (Japanese).
        "de" corresponde a alemán (German).
        "es" corresponde a español (Spanish).
        "ru" corresponde a ruso (Russian).
        "hi" corresponde a hindi (Hindi).
        "ko" corresponde a coreano (Korean).
        "zh" corresponde a chino (Chinese).
        "pl" corresponde a polaco (Polish)
        "ar" corresponde a argentino viste...

    La función devolverá:
        Un diccionario {'idioma':idioma, 'cantidad':respuesta}.

    '''
    df_filtrado=df[df['original_language']==idioma]
    cantidad=len(df_filtrado)
    respuesta = {'idioma':idioma, 'cantidad':cantidad}
  
    return respuesta


@app.get('/peliculas_dia/{dia}')
def peliculas_dia(dia: str):
    '''
    Retorna la cantidad de películas que se estrenaron en un día de la semana específico.

    Args:
        dia (str): El nombre del día de la semana para el cual se desea obtener la información. 
        Puede ser el nombre del día en inglés o en español.

    Returns:
        dict: Un diccionario que contiene el nombre del día de la semana (en formato capitalizado), 
        y la cantidad de películas que se estrenaron en ese día.

    '''
    
    # diccionario de mapeo de días en inglés a español
    dias_ingles = {
        'Monday': 'lunes',
        'Tuesday': 'martes',
        'Wednesday': 'miércoles',
        'Thursday': 'jueves',
        'Friday': 'viernes',
        'Saturday': 'sábado',
        'Sunday': 'domingo'
    }
    
    # diccionario de mapeo de días en español a inglés
    dias_espanol = {unidecode(v): k for k, v in dias_ingles.items()}
    
    # convertir la columna 'release_date' a tipo fecha
    df['release_date'] = pd.to_datetime(df['release_date'])
    
    # obtener el día de la semana para cada fecha en inglés
    df['day_of_week'] = df['release_date'].dt.day_name()
    
    # filtrar el DataFrame por el día de la semana especificado en español
    df_dia = df[df['day_of_week'] == dias_espanol[unidecode(dia.lower())]]
    
    # obtener el nombre del día de la semana en formato capitalizado
    nombre_dia = dia.capitalize()

    # obtener la cantidad de películas para ese día
    cantidad = len(df_dia)

    return {'dia_semana': nombre_dia, 'cantidad': cantidad}


@app.get('/franquicia/{franquicia}')
def franquicia(franquicia):
    '''
    Retorna la cantidad de películas, ganancia total y ganancia promedio para una franquicia específica.

    Args:
        franquicia (str): El nombre de la franquicia de películas.

    Returns:
        dict: Un diccionario que contiene el nombre de la franquicia (en formato capitalizado), 
        la cantidad de películas en la franquicia,
        la ganancia total de la franquicia formateada con separadores de miles, 
        y la ganancia promedio de la franquicia formateada con separadores de miles.
    
    '''

    # filtrar las películas que pertenecen a la franquicia especificada
    peliculas_franquicia = df[df['franchise'].str.contains(franquicia, case=False, na=False)]

    # obtener la cantidad de películas para esa franquicia
    cantidad = len(peliculas_franquicia)

    # calcular la ganancia total y el promedio de ganancia para esa franquicia
    ganancia_total = peliculas_franquicia['revenue'].sum()
    ganancia_promedio = peliculas_franquicia['revenue'].mean()

    return {'franquicia': franquicia.title(), 'cantidad': cantidad, 'ganancia_total': f'{ganancia_total:,}', 'ganancia_promedio': f'{ganancia_promedio:,}'}


@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais):
    '''
    Retorna la cantidad de películas producidas en un país específico.

    Args:
        pais (str): El nombre del país.

    Returns:
        dict: Un diccionario que contiene el nombre del país (en formato capitalizado) y 
        la cantidad de películas producidas en ese país.

    '''

    # filtrar las películas producidas en el país especificado
    peliculas_pais = df[df['production_countries'].str.contains(pais, case=False, na=False)]

    # obtener la cantidad de películas producidas en ese país
    cantidad = len(peliculas_pais)

    return {'pais': pais.title(), 'cantidad': cantidad}


@app.get('/productoras/{productora}')
def productoras(productora:str):
    '''
    Retorna la ganancia total y la cantidad de películas producidas por una productora específica.

    Args:
        productora (str): El nombre de la productora.

    Returns:
        dict: Un diccionario que contiene el nombre de la productora (en formato capitalizado), 
        la ganancia total de las películas producidas por la productora formateada con separadores de miles,
        y la cantidad de películas producidas por la productora.
    
    '''

# filtrar el DataFrame por las filas que contienen la productora especificada en la columna 'production_companies'
    filtered_df = df[df['production_companies'].str.contains(productora, case=False, na=False)]

    
    # calcular el total de 'revenue' y se cuenta el numero de peliculas 
    ganancia_total = filtered_df['revenue'].sum()
    cantidad = filtered_df.shape[0] 

    return {'productora': productora.title(), 'ganancia_total': f'{ganancia_total:,}', 'cantidad': cantidad}

@app.get('/retorno/{pelicula}')
def retorno(pelicula:str):
    """
    Retorna información sobre una película específica.

    Args:
        pelicula (str): El título de la película.

    Returns:
        dict: Un diccionario con los siguientes valores:
            - 'pelicula' (str): El título de la película en formato capitalizado.
            - 'inversion' (str): La inversión de la película formateada con separadores de miles.
            - 'ganancia' (str): La ganancia de la película formateada con separadores de miles.
            - 'retorno' (str): El retorno de la película redondeado a 2 decimales y formateado con separadores de miles.
            - 'anio' (int): El año en el que se lanzó la película.

    """
    

    # filtrar el DataFrame por el título de la película especificada
    pelicula_filtrada = df[df['title'].str.contains(pelicula, case=False, na=False)]

    # obtener los valores de inversión, ganancia, retorno y año de lanzamiento
    inversion = pelicula_filtrada['budget'].values[0]
    ganancia = pelicula_filtrada['revenue'].values[0]
    retorno = round(pelicula_filtrada['return'].values[0],2)
    anio = int(pelicula_filtrada['release_year'].values[0])
    
    return {'pelicula': pelicula.title(), 'inversion': f'{inversion:,}', 'ganancia': f'{ganancia:,}', 'retorno': f'{retorno:,}', 'anio': anio}


#ML

def preprocess_text(text):
    """
    Realiza el preprocesamiento de un texto dado.
    
    Args:
        text (str): El texto a preprocesar.
    
    Returns:
        str: El texto preprocesado.
    
    """

    # convierte el texto a minúsculas
    text = text.lower()

    # elimina los caracteres de puntuación
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    # aplica stemming o lematización si es necesario
    
    return text

# preprocesado de las columnas overview, tagline, genres y belong_to_collection
df['processed_text'] = df['overview'] + ' ' + df['tagline'] + ' ' + df['franchise'] + ' ' + df['genres']
df['processed_text'] = df['processed_text'].map(preprocess_text)

# calcula TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix_alpha = tfidf.fit_transform(df['processed_text'])
tfidf_matrix_beta = tfidf.fit_transform(df['title']) / 1.5
tfidf_matrix_gama = tfidf.fit_transform(df['genres']) * 1.15

tfidf_matrix = sp.hstack([tfidf_matrix_alpha, tfidf_matrix_beta, tfidf_matrix_gama]).tocsr()




@app.get('/recomendacion/{titulo}')
def recomendacion(titulo:str):
    """
    Lleva a cabo la recomendación de películas basada en la similitud de contenido utilizando TF-IDF.

    Args:
        titulo (str): El título de la película de referencia para la cual se realizará la recomendación.

    Returns:
        list: Una lista que contiene los títulos de las películas recomendadas con mayor similitud de contenido.
    """

    movie = df[df['title'].str.contains(titulo, case=False, na=False)]
    if movie.empty:
        return [{'error': 'Movie not found'}]
    
    movie_index = movie.index[0]
    movie_vector = tfidf_matrix.getrow(movie_index)
    
    # Calcula la similitud del coseno entre la película de entrada y todas las demás películas
    cosine_similarities = linear_kernel(movie_vector, tfidf_matrix).flatten()
    
    # Obtiene los índices de las películas ordenadas por puntajes de similitud
    similar_movie_indices = cosine_similarities.argsort()[::-1]
    
    # Filtra la propia película de entrada
    similar_movie_indices = similar_movie_indices[similar_movie_indices != movie_index]
    
    # Filtra las películas con un puntaje de similitud inferior a 0.35
    high_similarity_indices = similar_movie_indices[cosine_similarities[similar_movie_indices] >= 0.2]
    
    # Obtiene los títulos de las películas con mayor similitud
    recommended_movies = df.iloc[high_similarity_indices[:5]]['title'].tolist()
    
    return recommended_movies
