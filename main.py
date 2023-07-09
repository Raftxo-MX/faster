from fastapi import FastAPI
import pandas as pd
import re
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
    """
    尺闩ﾁ七〤ㄖ ﾁ闩丂セ 闩尸讠<br>
    Implementación de la API con FastAPI en Render<br>
    PI01_MLops DTS-12<br>
    Rafa J.W.
    """
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
    Ingresa el idioma en formato abreviado conforme la norma internacional ISO 639-1.<br> 
    Los idiomas más populares son:<br>
        "en" corresponde a inglés (English).<br>
        "fr" corresponde a francés (French).<br>
        "it" corresponde a italiano (Italian).<br>
        "ja" corresponde a japonés (Japanese).<br>
        "de" corresponde a alemán (German).<br>
        "es" corresponde a español (Spanish).<br>
        "ru" corresponde a ruso (Russian).<br>
        "hi" corresponde a hindi (Hindi).<br>
        "ko" corresponde a coreano (Korean).<br>
        "zh" corresponde a chino (Chinese).<br>
        "pl" corresponde a polaco (Polish)<br>
        "ar" corresponde a argentino viste...<br>
<br>
    La función devolverá:<br>
        Un diccionario {'idioma':idioma, 'cantidad':respuesta}.<br>

    '''
    df_filtrado=df[df['original_language']==idioma]
    cantidad=len(df_filtrado)
    respuesta = {'idioma':idioma, 'cantidad':cantidad}
  
    return respuesta

@app.get('/peliculas_duracion/{pelicula}')
def peliculas_duracion(pelicula:str):
    """
    Ingresa un título o parte del título y la API devolverá<br>
    ese título con su duración y año de estreno (o varios)<br>
    """

    # Verificar que se proporcione al menos un carácter como argumento
    if len(pelicula.strip()) == 0:
        mensaje_error = {'error': 'Debe proporcionar al menos un carácter para buscar'}
        return mensaje_error
    
    # Escapar la cadena de búsqueda para evitar errores de expresión regular
    pelicula_escaped = re.escape(pelicula)
    
    # Crear una lista vacía para almacenar los resultados
    resultados = []
    
    # Filtrar las filas que contienen el texto especificado en el título
    df_filtrado = df[df['title'].str.contains(pelicula_escaped, case=False)]
    
    # Verificar si hay resultados
    if len(df_filtrado) == 0:
        mensaje_error = {'error': f'No se encontró ninguna película que contenga la palabra "{pelicula}"'}
        return mensaje_error
    
    # Iterar sobre las filas filtradas y agregar un diccionario de resultados para cada una
    for indice, fila in df_filtrado.iterrows():
        # Agregar un máximo de 10 resultados
        if len(resultados) == 10:
            break
        resultado = {'titulo': fila['title'], 'duracion (minutos)': fila['runtime'], 'anio_lanzamiento': fila['release_year']}
        resultados.append(resultado)
    
    # Devolver la lista de resultados
    return resultados

@app.get('/franquicia/{franquicia}')
def franquicia(franquicia):
    '''
    Ingresar la franquicia (no hace falta poner Collection)<br>

    Devuelve la cantidad de películas en dicha franquicia, ganancia total y promedio.
   
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

    '''

    # filtrar las películas producidas en el país especificado
    peliculas_pais = df[df['produced_in'].str.contains(pais, case=False, na=False)]

    # obtener la cantidad de películas producidas en ese país
    cantidad = len(peliculas_pais)

    return {'pais': pais.title(), 'cantidad': cantidad}


@app.get('/productoras_exitosas/{productora}')
def productoras_exitosas(productora:str):
    '''
    Ingresar una productora<br>
    (Algunas productoras famosas: 'Warner Bros','TriStar Pictures')<br>

    Devuelve revenue total y la cantidad de películas que realizó.
   
    '''

    # filtrar el DataFrame por las filas que contienen la productora especificada en la columna 'produced_by'
    filtered_df = df[df['produced_by'].str.contains(productora, case=False, na=False)]

    
    # calcular el total de 'revenue' y el numero de peliculas
    ganancia_total = filtered_df['revenue'].sum()
    cantidad = filtered_df.shape[0] 

    return {'productora': productora.title(), 'revenue_total': f'{ganancia_total:,}', 'cantidad': cantidad}

@app.get('/get_director/{nombre_director}')
def get_director(nombre_director: str):
    """
    Ingresar el nombre de un director incluído en el dataset.<br>

    Devuelve el éxito del mismo medido a través del retorno.<br>

    Además devuelve el nombre de cada película con su fecha de lanzamiento, retorno individual,
    costo y ganancia de la misma en formato lista.
    
    """
    # Filtramos el DataFrame por el nombre del director ingresado
    pelis_del_director = df[df['directed_by'] == nombre_director]

    if pelis_del_director.empty:
        return f"No se encontraron películas dirigidas por {nombre_director}."

    # Limpiamos los valores 'N/A' y reemplazamos los valores 'inf' y 'nan' en el DataFrame
    pelis_del_director = pelis_del_director.replace({'N/A': df.nan, df.inf: df.nan})

    # Calculamos el promedio de éxito de las películas del director sin considerar los valores NaN
    exito_director = pelis_del_director['return'].mean(skipna=True)

    # Limitamos la cantidad de películas a devolver
    max_movies = 10  # Se puede ajustar este valor según tus necesidades
    pelis_del_director = pelis_del_director.head(max_movies)

    # Crea una lista de películas dirigidas por el director
    lista_pelis = pelis_del_director[['title', 'release_date', 'return', 'budget', 'revenue']].to_dict(orient='records')

    return {'director_success': exito_director, 'lista_peliculas': lista_pelis}


# ML - el modelo se hizo en ML_pi01_dts12.ipynb
#      exportando las recomendaciones a un fichero CSV

# Cargamos el archivo recomendaciones.csv como un dataframe
df_recomendaciones = pd.read_csv('recomendaciones.csv')

# ENDPOINT RECOMENDACION
@app.get('/recomendacion/{titulo}')
def recomendacion(titulo):
    """
    Introduce un título de una película.<br>
    La función devolverá 5 títulos de películas recomendadas.
    """
    # Buscar el título en el dataframe
    filtro = df_recomendaciones['title'] == titulo
    if filtro.any():
        # Obtener las recomendaciones para el título dado
        recomendaciones = df_recomendaciones.loc[filtro, 'recomendaciones'].values[0]
        return {'lista recomendada': recomendaciones}
    else:
        return {'error': 'El título no se encuentra en la base de datos'}
 # Ejemplo de uso de la función
titulo_pelicula = 'Título de la película'
resultado = recomendacion(titulo_pelicula)
print(resultado)
