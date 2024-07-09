import os
import re
import json

def obtenerTuplaswsp():
    # Obtiene el directorio actual de trabajo
    directorio_actual = os.getcwd()

    # Construye la ruta al archivo _chat.txt
    ruta_archivo = os.path.join(directorio_actual, '_chat.txt')

    # Preparar la lista para almacenar las tuplas
    tuplas = []

    # Variables temporales para el inicio y fin de las tuplas
    inicio_tupla = None
    final_tupla = None

    # Abre el archivo en modo de lectura
    with open(ruta_archivo, 'r', encoding='utf-8') as file:
        # Recorre cada línea en el archivo
        # Asumiendo que la apertura del archivo y la inicialización de variables ya se hizo antes
        for linea in file:  # Itera directamente sobre las líneas del archivo
            linea_limpia = re.sub(r"\[.*?\]", "", linea).strip()
            if '.opus' in linea_limpia or '.jpg' in linea_limpia or '.webp' in linea_limpia:
                continue
            if 'Mi Gatita 😻:' in linea_limpia:
                texto_despues = linea_limpia.split('Mi Gatita 😻:', 1)[1].strip()
                inicio_tupla = texto_despues  # Guarda el texto después del patrón
                # Verifica si la línea contiene "Benjamin Macias:"
            elif 'Benjamin Macias:' in linea_limpia and inicio_tupla:
                texto_despues = linea_limpia.split('Benjamin Macias:', 1)[1].strip()
                tuplas.append((inicio_tupla, texto_despues))
                inicio_tupla = None  # Resetear para la próxima tupla

    # Imprimir las tuplas resultantes
    for tupla in tuplas:
        print(tupla)
    print(f"Total de tuplas: {len(tuplas)}")

    # Convertir la lista de tuplas a un formato compatible con JS (p.ej., un array de arrays)
    datos_js = [list(tupla) for tupla in tuplas]

    # Convertir a cadena JSON
    datos_json = json.dumps(datos_js)

    # Escribir en un archivo JS
    ruta_archivo_js = os.path.join(directorio_actual, 'datos.js')
    with open(ruta_archivo_js, 'w', encoding='utf-8') as archivo_js:
        # Asignar la cadena JSON a una variable en el archivo JS
        archivo_js.write(f"const datos = {datos_json};\n")

    # Ahora, 'datos.js' contiene un array de arrays asignado a la variable 'datos'
    return tuplas

def obtenerTuplasPeli():
    directorio_actual = os.getcwd()

    # Construye la ruta al archivo _chat.txt
    ruta_archivo = os.path.join(directorio_actual, 'TheTruman.txt')

    with open(ruta_archivo, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regex para encontrar las líneas de tiempo y las líneas de subtítulos
    subtitle_blocks = re.split(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', content)
    
    # Remover entradas vacías y limpiar el contenido
    subtitle_blocks = [block.strip().replace('\n', ' ') for block in subtitle_blocks if block.strip()]

    # Convertir a formato de lista de tuplas
    subtitle_list = [(block,) for block in subtitle_blocks]

    return subtitle_list

