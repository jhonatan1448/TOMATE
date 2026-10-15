import json
import streamlit as st

st.set_page_config(
    page_title="Fertirriego de tomate",
    page_icon="🍅",
    layout="wide",
)

st.title("🍅 Modelo de fertirriego de tomate")
st.caption("Catálogo de fertilizantes y simulación hidráulica")

st.warning(
    "Esta aplicación es un simulador, no una recomendación agronómica. "
    "No calcula dosis ni desbloquea fertilizantes pendientes de validación."
)

tab_catalogo, tab_riego, tab_alertas = st.tabs(
    ["Catálogo", "Sistema de riego", "Alertas"]
)


def obtener_productos(datos):
    """Reconoce formatos comunes sin modificar el catálogo."""
    if isinstance(datos, list):
        return datos

    if isinstance(datos, dict):
        for clave in (
            "fertilizantes", "productos", "catalogo",
            "catalogo_fertilizantes", "fertilizers", "products",
        ):
            contenido = datos.get(clave)
            if isinstance(contenido, list):
                return contenido
            if isinstance(contenido, dict):
                return [
                    {"nombre_en_catalogo": nombre, "datos": valor}
                    for nombre, valor in contenido.items()
                ]

        # Catálogo organizado como nombre del producto: composición.
        if datos and all(isinstance(v, dict) for v in datos.values()):
            return [
                {"nombre_en_catalogo": nombre, "datos": valor}
                for nombre, valor in datos.items()
            ]

    return []


with tab_catalogo:
    st.subheader("Cargar catálogo de fertilizantes")
    st.write(
        "Selecciona el archivo catalogo_fertilizantes.json "
        "que descargaste desde Colab."
    )

    archivo = st.file_uploader(
        "Archivo del catálogo",
        type=["json"],
    )

    if archivo is not None:
        try:
            datos = json.loads(
                archivo.getvalue().decode("utf-8-sig")
            )
        except (ValueError, UnicodeError) as error:
            st.error(f"No se pudo leer el archivo JSON: {error}")
        else:
            productos = obtener_productos(datos)

            if productos:
                st.success(
                    f"Se reconocieron {len(productos)} entradas. "
                    "Revisa que correspondan a tus fertilizantes."
                )

                busqueda = st.text_input(
                    "Buscar producto"
                ).strip().casefold()

                for indice, producto in enumerate(productos, start=1):
                    if isinstance(producto, dict):
                        nombre = str(
                            producto.get("nombre")
                            or producto.get("nombre_comercial")
                            or producto.get("name")
                            or producto.get("producto")
                            or producto.get("nombre_en_catalogo")
                            or f"Producto {indice}"
                        )
                    else:
                        nombre = f"Producto {indice}"

                    texto = json.dumps(
                        producto, ensure_ascii=False
                    ).casefold()

                    if busqueda and busqueda not in texto:
                        continue

                    with st.expander(nombre):
                        st.json(producto)
            else:
                st.info(
                    "No se reconoció la organización de los productos. "
                    "Puedes consultar el contenido original abajo."
                )

            with st.expander("Ver catálogo original completo"):
                st.json(datos)

            st.caption(
                "Los datos se muestran tal como están en el archivo. "
                "No se completan valores desconocidos, no se convierten "
                "unidades ni se modifican estados de validación."
            )
    else:
        st.info(
            "Puedes usar la pestaña Sistema de riego "
            "aunque todavía no hayas cargado el catálogo."
        )


with tab_riego:
    st.subheader("Configuración hidráulica")
    st.caption(
        "El cálculo supone sectores iguales regados uno después de otro. "
        "Los valores iniciales proceden de la configuración informada."
    )

    col1, col2 = st.columns(2)

    with col1:
        plantas = st.number_input(
            "Número total de plantas",
            min_value=1,
            value=5000,
            step=1,
        )
        sectores = st.number_input(
            "Número de sectores",
            min_value=1,
            max_value=int(plantas),
            value=min(4, int(plantas)),
            step=1,
        )
        goteros = st.number_input(
            "Goteros equivalentes por planta",
            min_value=0.1,
            value=6.0,
            step=0.1,
        )
        caudal_gotero = st.number_input(
            "Caudal de cada gotero (L/h)",
            min_value=0.01,
            value=1.0,
            step=0.1,
        )

    with col2:
        minutos_riego = st.number_input(
            "Riego total por sector (minutos)",
            min_value=0.1,
            value=15.0,
            step=1.0,
        )
        succion = st.number_input(
            "Succión del Venturi (L/min)",
            min_value=0.0,
            value=1.0,
            step=0.1,
        )
        capacidad = st.number_input(
            "Capacidad del tanque de solución madre (L)",
            min_value=0.1,
            value=200.0,
            step=10.0,
        )

    if plantas % sectores != 0:
        st.warning(
            "Las plantas no se reparten exactamente en sectores iguales. "
            "Los resultados por sector representan un promedio."
        )

    st.markdown("**Distribución del tiempo por sector**")
    st.caption(
        "La inyección empieza en cero por seguridad. "
        "Modificar estos tiempos solo simula el consumo de solución."
    )

    pre = st.number_input(
        "Pre-riego con agua sola (minutos)",
        min_value=0.0,
        value=0.0,
        step=0.5,
    )
    inyeccion = st.number_input(
        "Inyección por sector (minutos)",
        min_value=0.0,
        value=0.0,
        step=0.5,
    )

    post = minutos_riego - pre - inyeccion

    plantas_sector = plantas / sectores
    caudal_sector_h = plantas_sector * goteros * caudal_gotero
    caudal_sector_min = caudal_sector_h / 60
    agua_planta = goteros * caudal_gotero * minutos_riego / 60
    agua_sector = caudal_sector_h * minutos_riego / 60
    agua_total = agua_sector * sectores

    st.subheader("Resultados teóricos")

    a, b, c = st.columns(3)
    a.metric("Plantas por sector", f"{plantas_sector:,.1f}")
    b.metric("Caudal por sector", f"{caudal_sector_min:,.2f} L/min")
    c.metric("Agua por planta y ciclo", f"{agua_planta:,.2f} L")

    st.table([
        {
            "Concepto": "Caudal nominal por sector",
            "Resultado": f"{caudal_sector_h:,.1f} L/h",
        },
        {
            "Concepto": "Agua por sector y ciclo",
            "Resultado": f"{agua_sector:,.1f} L",
        },
        {
            "Concepto": "Agua total por ciclo",
            "Resultado": f"{agua_total:,.1f} L",
        },
        {
            "Concepto": "Tiempo total de riego secuencial",
            "Resultado": f"{minutos_riego * sectores:,.1f} minutos",
        },
    ])

    if post < -0.000001:
        st.error(
            "El pre-riego más la inyección supera el tiempo total "
            "por sector. Corrige los tiempos para simular la inyección."
        )
    else:
        post = max(0.0, post)
        solucion_sector = succion * inyeccion
        solucion_total = solucion_sector * sectores
        relacion = 100 * succion / caudal_sector_min

        st.write(f"**Post-riego disponible:** {post:.1f} minutos")

        st.table([
            {
                "Concepto": "Solución madre consumida por sector",
                "Resultado": f"{solucion_sector:,.2f} L",
            },
            {
                "Concepto": "Solución madre consumida en todo el ciclo",
                "Resultado": f"{solucion_total:,.2f} L",
            },
            {
                "Concepto": "Succión / caudal nominal durante inyección",
                "Resultado": f"{relacion:.3f} %",
            },
        ])

        if inyeccion == 0 or succion == 0:
            st.info("No hay consumo de solución madre en esta simulación.")
        else:
            if pre == 0 or post == 0:
                st.warning(
                    "La simulación no reserva pre-riego o post-riego. "
                    "Antes de inyectar deben verificarse la presurización "
                    "y el tiempo real de lavado de las líneas."
                )

            if solucion_total > capacidad:
                st.error(
                    "El consumo simulado supera la capacidad de un tanque "
                    "lleno para atender todos los sectores."
                )
            else:
                st.info(
                    "Por volumen nominal, un tanque lleno cubriría "
                    "el consumo simulado. Esto no valida una mezcla "
                    "ni una dosis de fertilizante."
                )

    st.caption(
        "Los volúmenes de riego se calculan con el caudal nominal "
        "de los goteros; no se suma nuevamente el caudal del Venturi. "
        "La relación de succión no es una concentración de nutrientes. "
        "No se incluyen tiempos de cambio entre sectores."
    )

    st.warning(
        "Los 6 goteros equivalentes por planta y sus caudales deben "
        "verificarse en campo. Dos cintas por surco no bastan para "
        "confirmar ese número sin conocer la separación entre emisores. "
        "El volumen calculado no determina la necesidad de agua: "
        "ajusta el riego según humedad, clima y estado del cultivo."
    )


with tab_alertas:
    st.subheader("Pendientes técnicos y restricciones")

    st.error(
        "Van Iperen Producción: el dato supuesto de 7,30 % de boro "
        "no está verificado. Mantener excluido de la dosificación "
        "hasta confirmar su composición."
    )

    st.warning(
        "Haifa NCal: confirmar si el 26 % corresponde a Ca o CaO. "
        "No son porcentajes equivalentes de calcio elemental."
    )

    st.warning(
        "Van Iperen Inicio: confirmar a qué elemento corresponde "
        "el 0,006 % señalado como zinc/cobre."
    )

    st.warning(
        "Solupotasse: mantener pendientes los valores de "
        "50 % K2O y 18 % S hasta comprobar la etiqueta."
    )

    st.info(
        "Abonisa es el abonado de fondo informado. No inyectarlo "
        "por Venturi sin confirmación expresa de aptitud para "
        "fertirriego y solubilidad por parte del fabricante."
    )

    st.info(
        "No mezclar fuentes de calcio con sulfatos o fosfatos "
        "en una misma solución madre concentrada sin una "
        "validación técnica de compatibilidad."
    )

    st.write(
        "Para un plan nutricional faltan análisis de agua y suelo, "
        "evaluación del cultivo y detalles de aplicación del abonado "
        "de fondo. El contenido total de ese abono no equivale "
        "a nutrientes inmediatamente disponibles."
    )

    st.caption(
        "Estas alertas son recordatorios del contexto informado; "
        "no constituyen una validación automática del JSON. "
        "Esta versión no guarda cambios permanentemente."
    )
