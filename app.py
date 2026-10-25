import copy
import hashlib
import json

import numpy as np
import pandas as pd
import streamlit as st

try:
    from scipy.optimize import lsq_linear
except ImportError:
    lsq_linear = None


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="TOMATE · Formulador de fertirriego",
    page_icon="🍅",
    layout="wide",
)

ELEMENTOS = [
    "N", "P", "K", "Ca", "Mg", "S",
    "B", "Cu", "Fe", "Mn", "Zn", "Mo",
]

FORMAS_N = ["N_nitrico", "N_amoniacal", "N_ureico"]

# Fracción elemental dentro del óxido.
OXIDOS = {
    "P": ("P2O5", 0.4364),
    "K": ("K2O", 0.8301),
    "Ca": ("CaO", 0.7147),
    "Mg": ("MgO", 0.6030),
    "S": ("SO3", 0.4005),
}

PERFILES_CONFIRMADOS = {
    "Van Iperen Producción": {
        "nutrientes": {
            "N_total": 12.0,
            "N_nitrico": 11.0,
            "N_amoniacal": 1.0,
            "P2O5": 5.0,
            "K2O": 40.0,
            "B": 0.02,
            "Cu": 0.011,
            "Fe": 0.1,
            "Mn": 0.05,
            "Zn": 0.02,
            "Mo": 0.01,
        },
    },
    "Van Iperen MagBoro": {
        "nutrientes": {
            "N_total": 11.0,
            "N_nitrico": 11.0,
            "MgO": 15.7,
            "B": 0.13,
        },
        "bloquear": True,
        "pendientes": [
            "Lote informado vencido en abril de 2025. "
            "No habilitar este lote para aplicación."
        ],
    },
    "Haifa Poly-Feed Producción": {
        "nutrientes": {
            "N_total": 10.0,
            "N_nitrico": 10.0,
            "P2O5": 10.0,
            "K2O": 43.0,
            "B": 0.02,
            "Cu": 0.01,
            "Fe": 0.1,
            "Mn": 0.05,
            "Zn": 0.02,
        },
    },
    "Haifa DKP": {
        "nutrientes": {
            "N_total": 0.0,
            "P2O5": 41.0,
            "K2O": 54.0,
        },
        "advertencias": [
            "Revisar alcalinidad y compatibilidad. "
            "La composición no permite calcular el pH final."
        ],
    },
    "Naturzinc-WSP": {
        "nutrientes": {"Zn": 13.0},
        "advertencias": ["Zn declarado 100 % quelatado con EDTA."],
    },
    "Naturmanganeso-WSP": {
        "nutrientes": {"Mn": 12.0},
        "advertencias": ["Mn declarado 100 % quelatado con EDTA."],
    },
    "Naturquel-Fe 4,8": {
        "nutrientes": {"Fe": 6.0},
        "bloquear": True,
        "pendientes": [
            "Aclarar con el proveedor la discrepancia EDDHA/EDTA."
        ],
        "advertencias": [
            "El 4,8 % orto-orto está incluido en el 6 % de Fe total."
        ],
    },
    "Kellus Cu": {
        "nutrientes": {"Cu": 14.5},
        "bloquear": True,
        "pendientes": [
            "Identificar el agente quelatante, si corresponde."
        ],
        "advertencias": [
            "Los metales pesados no se contabilizan como nutrientes. "
            "Esto no constituye una validación de inocuidad."
        ],
    },
    "Molib-K": {
        "nutrientes": {"K": 2.49, "B": 8.0, "Mo": 10.0},
        "bloquear": True,
        "pendientes": [
            "Confirmar con el fabricante solubilidad y aptitud "
            "para fertirriego."
        ],
        "advertencias": [
            "K expresado como elemento, no como K2O.",
            "Por cada unidad de Mo aporta 0,8 de B y 0,249 de K.",
        ],
    },
}


# ============================================================
# FUNCIONES GENERALES
# ============================================================

def numero(valor):
    """Devuelve float finito; null y valores desconocidos no son cero."""
    if valor is None or isinstance(valor, bool):
        return None

    if isinstance(valor, str):
        valor = valor.strip().replace("%", "").replace(",", ".")
        if valor.lower() in {"", "null", "none", "nd", "n/d", "-"}:
            return None

    try:
        resultado = float(valor)
        return resultado if np.isfinite(resultado) else None
    except (TypeError, ValueError):
        return None


def booleano(valor):
    if isinstance(valor, bool):
        return valor
    if isinstance(valor, str):
        texto = valor.strip().lower()
        if texto in {"true", "sí", "si", "yes", "1"}:
            return True
        if texto in {"false", "no", "0"}:
            return False
    if isinstance(valor, (int, float)) and valor in (0, 1):
        return bool(valor)
    return None


def lista_textos(valor):
    if valor is None:
        return []
    if isinstance(valor, str):
        return [valor] if valor.strip() else []
    if isinstance(valor, list):
        return [str(v) for v in valor if v is not None]
    if isinstance(valor, dict):
        return [f"{k}: {v}" for k, v in valor.items()]
    return [str(valor)]


def unicos(valores):
    return list(dict.fromkeys(valores))


def normalizar_catalogo(datos):
    """
    Admite productos como diccionario o lista.
    Conserva los metadatos y los campos originales.
    """
    if not isinstance(datos, dict):
        raise ValueError("La raíz del JSON debe ser un objeto.")

    resultado = copy.deepcopy(datos)
    productos = resultado.get("productos")

    if isinstance(productos, list):
        convertidos = {}
        for indice, producto in enumerate(productos, start=1):
            if not isinstance(producto, dict):
                raise ValueError(f"El producto {indice} no es un objeto.")
            identificador = str(producto.get("id", f"producto_{indice}"))
            if identificador in convertidos:
                raise ValueError(f"ID duplicado: {identificador}")
            convertidos[identificador] = producto
        productos = convertidos

    if not isinstance(productos, dict) or not productos:
        raise ValueError(
            "Se requiere una clave 'productos' con un diccionario "
            "o una lista no vacía."
        )

    normalizados = {}
    for identificador, producto in productos.items():
        if not isinstance(producto, dict):
            raise ValueError(f"Producto inválido: {identificador}")

        producto = copy.deepcopy(producto)
        producto.setdefault(
            "nombre",
            producto.get("nombre_comercial", str(identificador)),
        )
        normalizados[str(identificador)] = producto

    resultado["productos"] = normalizados
    return resultado


def aplicar_perfil_confirmado(producto, nombre_perfil):
    nuevo = copy.deepcopy(producto)
    perfil = PERFILES_CONFIRMADOS[nombre_perfil]
    declarados = perfil["nutrientes"]

    nutrientes = {
        ("N_total" if elemento == "N" else elemento): 0.0
        for elemento in ELEMENTOS
    }
    nutrientes.update({forma: 0.0 for forma in FORMAS_N})

    for elemento, (oxido, _) in OXIDOS.items():
        if oxido in declarados:
            nutrientes.pop(elemento, None)

    nutrientes.update(declarados)

    nuevo.setdefault(
        "composicion_antes_actualizacion",
        copy.deepcopy(nuevo.get("nutrientes_pct", {})),
    )
    nuevo["nutrientes_pct"] = nutrientes
    nuevo["equivalencias_elementales_pct"] = {}
    nuevo["composicion_confirmada_usuario"] = True
    nuevo["perfil_confirmado"] = nombre_perfil
    nuevo["criterio_no_declarados"] = (
        "0 % para balance por acuerdo del usuario; "
        "no demuestra ausencia analítica."
    )

    nuevo["pendientes"] = unicos(
        lista_textos(nuevo.get("pendientes"))
        + perfil.get("pendientes", [])
    )
    nuevo["observaciones"] = unicos(
        lista_textos(nuevo.get("observaciones"))
        + perfil.get("advertencias", [])
        + [nuevo["criterio_no_declarados"]]
    )

    if perfil.get("bloquear"):
        nuevo["bloqueado_dosificacion_automatica"] = True

    # No modificar verificado, soluble ni estado.
    return nuevo


def interpretar_producto(identificador, producto):
    nutrientes = producto.get("nutrientes_pct", {})
    equivalencias = producto.get("equivalencias_elementales_pct", {})
    motivos = []
    valores = {}

    if not isinstance(nutrientes, dict):
        nutrientes = {}
        motivos.append("Formato de nutrientes_pct inválido")

    if not isinstance(equivalencias, dict):
        equivalencias = {}

    # Un porcentaje inválido bloquea, incluso si no entra en el balance.
    for clave, valor in nutrientes.items():
        n = numero(valor)
        if valor is not None and (
            n is None or n < 0 or n > 100
        ):
            motivos.append(f"Porcentaje inválido: {clave}")

    for elemento in ELEMENTOS:
        if elemento == "N":
            valor = numero(
                nutrientes.get("N_total", nutrientes.get("N"))
            )
        else:
            directo = numero(nutrientes.get(elemento))
            convertido = None

            if elemento in OXIDOS:
                oxido, factor = OXIDOS[elemento]
                porcentaje_oxido = numero(nutrientes.get(oxido))
                if porcentaje_oxido is not None:
                    convertido = porcentaje_oxido * factor

            if directo is not None and convertido is not None:
                if not np.isclose(
                    directo, convertido, rtol=0.02, atol=0.01
                ):
                    motivos.append(
                        f"Inconsistencia entre {elemento} y su óxido"
                    )

            # Nunca sumar elemento y óxido.
            valor = directo if directo is not None else convertido

        if valor is None:
            valor = numero(equivalencias.get(elemento))

        if valor is not None and not 0 <= valor <= 100:
            motivos.append(f"Valor elemental inválido: {elemento}")
            valor = None

        valores[elemento] = np.nan if valor is None else valor

    formas = {
        forma: numero(nutrientes.get(forma))
        for forma in FORMAS_N
    }
    conocidas = [v for v in formas.values() if v is not None]
    total_n = valores["N"]

    if np.isfinite(total_n):
        if sum(conocidas) > total_n + 0.05:
            motivos.append("Las formas de N exceden el N total")
        elif len(conocidas) == 3 and not np.isclose(
            sum(conocidas), total_n, atol=0.05, rtol=0
        ):
            motivos.append("Las formas de N no coinciden con N total")

    desconocidos = [
        elemento for elemento, valor in valores.items()
        if not np.isfinite(valor)
    ]
    if desconocidos:
        motivos.append(
            "Composición incompleta: " + ", ".join(desconocidos)
        )

    bloqueo = booleano(
        producto.get("bloqueado_dosificacion_automatica")
    )
    if bloqueo is not False:
        motivos.append(
            "Bloqueo de dosificación activo"
            if bloqueo is True
            else "Autorización de dosificación no declarada"
        )

    if booleano(producto.get("verificado")) is not True:
        motivos.append("Verificación de uso pendiente")

    if booleano(producto.get("soluble")) is not True:
        motivos.append("Solubilidad para uso pendiente")

    motivos += lista_textos(producto.get("pendientes"))

    nombre = str(producto.get("nombre", identificador))
    if "abonisa" in nombre.casefold():
        motivos.append("Abonisa excluido del sistema Venturi")

    estado = str(producto.get("estado", "no declarado"))
    if any(
        palabra in estado.casefold()
        for palabra in ("bloque", "pendiente", "vencid", "no_apto", "no apto")
    ):
        motivos.append(f"Estado restrictivo: {estado}")

    return {
        "ID": identificador,
        "Producto": nombre,
        "Estado": estado,
        "Composición confirmada": (
            producto.get("composicion_confirmada_usuario") is True
        ),
        "Bloqueo original": bloqueo,
        "Apto para cálculo": not motivos,
        "Motivos": " | ".join(unicos(motivos)),
        "Desconocidos": ", ".join(desconocidos),
        **valores,
        **{
            forma: np.nan if valor is None else valor
            for forma, valor in formas.items()
        },
    }


def tabla_catalogo(catalogo):
    return pd.DataFrame([
        interpretar_producto(identificador, producto)
        for identificador, producto in catalogo["productos"].items()
    ])


# ============================================================
# IMPORTACIÓN
# ============================================================

st.title("🍅 TOMATE · Formulador de fertirriego")

st.caption(
    "Balance nutricional elemental. Los resultados no validan "
    "compatibilidad química, pH, CE ni seguridad agronómica."
)

with st.sidebar:
    st.header("Catálogo")

    archivo = st.file_uploader(
        "Cargar catálogo de fertilizantes",
        type=["json"],
    )

    if archivo is not None:
        contenido = archivo.getvalue()
        firma = hashlib.sha256(contenido).hexdigest()

        if firma != st.session_state.get("firma_catalogo"):
            try:
                datos = json.loads(contenido.decode("utf-8-sig"))
                nuevo_catalogo = normalizar_catalogo(datos)
                st.session_state["catalogo_importado"] = copy.deepcopy(
                    nuevo_catalogo
                )
                st.session_state["catalogo_trabajo"] = nuevo_catalogo
                st.session_state["firma_catalogo"] = firma
            except Exception as exc:
                st.error(f"No se pudo importar: {exc}")
                st.stop()

    if "catalogo_trabajo" not in st.session_state:
        st.info("Carga tu JSON para comenzar.")
        st.stop()

    if st.button("Restaurar catálogo importado"):
        st.session_state["catalogo_trabajo"] = copy.deepcopy(
            st.session_state["catalogo_importado"]
        )
        st.rerun()

    st.caption(
        "Los cambios son temporales y no sobrescriben el JSON."
    )


# ============================================================
# PERFILES CONFIRMADOS
# ============================================================

with st.sidebar.expander("Integrar porcentajes confirmados"):
    productos = st.session_state["catalogo_trabajo"]["productos"]

    perfil = st.selectbox(
        "Perfil confirmado",
        list(PERFILES_CONFIRMADOS),
    )

    crear = st.checkbox("Añadir un producto que no existe")

    if crear:
        nuevo_id = st.text_input("ID nuevo y único").strip()
        nuevo_nombre = st.text_input("Nombre comercial").strip()
        destino = None
    else:
        destino = st.selectbox(
            "Producto correspondiente en el catálogo",
            list(productos),
            format_func=lambda k: f"{productos[k]['nombre']} · {k}",
        )

    aceptar = st.checkbox(
        "Confirmo que corresponde exactamente al perfil seleccionado"
    )

    if st.button("Aplicar composición"):
        try:
            if not aceptar:
                raise ValueError("Confirma la correspondencia del producto.")

            candidato = copy.deepcopy(
                st.session_state["catalogo_trabajo"]
            )

            if crear:
                if not nuevo_id or not nuevo_nombre:
                    raise ValueError("Indica ID y nombre.")
                if nuevo_id in candidato["productos"]:
                    raise ValueError("El ID ya existe.")

                nombres = {
                    str(p["nombre"]).strip().casefold()
                    for p in candidato["productos"].values()
                }
                if nuevo_nombre.casefold() in nombres:
                    raise ValueError("El nombre ya existe.")

                destino = nuevo_id
                candidato["productos"][destino] = {
                    "nombre": nuevo_nombre,
                    "estado": "pendiente_validacion_uso",
                    "bloqueado_dosificacion_automatica": True,
                    "verificado": False,
                    "soluble": None,
                }

            candidato["productos"][destino] = aplicar_perfil_confirmado(
                candidato["productos"][destino],
                perfil,
            )

            st.session_state["catalogo_trabajo"] = candidato
            st.session_state["aviso_perfil"] = (
                f"Composición de {perfil} incorporada. "
                "Se mantienen bloqueos y pendientes."
            )
            st.rerun()

        except Exception as exc:
            st.error(str(exc))

    st.caption(
        "Esta operación confirma porcentajes, no autoriza aplicaciones."
    )

if "aviso_perfil" in st.session_state:
    st.success(st.session_state.pop("aviso_perfil"))


# ============================================================
# CONFIGURACIÓN DEL CULTIVO
# ============================================================

with st.sidebar:
    st.header("Cultivo y calendario")
    st.caption(
        "Valores iniciales según lo acordado. "
        "Compruébalos antes de calcular."
    )

    superficie = st.number_input(
        "Superficie del cultivo (m²)",
        min_value=1.0,
        value=2500.0,
        step=100.0,
    )
    plantas = st.number_input(
        "Número de plantas",
        min_value=1,
        value=5000,
        step=100,
    )
    dias_referencia = st.number_input(
        "Ciclo de referencia del catálogo (días)",
        min_value=1,
        value=120,
    )
    semanas_etapa = st.number_input(
        "Semanas por etapa",
        min_value=1,
        value=5,
    )
    aplicaciones_semana = st.number_input(
        "Aplicaciones por semana",
        min_value=1,
        max_value=7,
        value=5,
    )
    referencia_mgo = st.number_input(
        "Referencia del ciclo (kg MgO/ha)",
        min_value=0.0,
        value=89.3,
        step=0.1,
    )

catalogo = st.session_state["catalogo_trabajo"]
tabla = tabla_catalogo(catalogo)

hectareas = superficie / 10000
aplicaciones_etapa = semanas_etapa * aplicaciones_semana
aplicaciones_totales = 3 * aplicaciones_etapa
dias_calendario = 3 * semanas_etapa * 7

etapas = [
    ("Vegetativo", 0.30),
    ("Desarrollo", 0.40),
    ("Llenado", 0.30),
]

plan = pd.DataFrame([
    {
        "Etapa": nombre,
        "Distribución (%)": fraccion * 100,
        "Aplicaciones": aplicaciones_etapa,
        "MgO (kg/ha · etapa)": referencia_mgo * fraccion,
        "MgO (kg · cultivo · etapa)": (
            referencia_mgo * fraccion * hectareas
        ),
        "Mg elemental (g · cultivo · aplicación)": (
            referencia_mgo
            * fraccion
            * hectareas
            * OXIDOS["Mg"][1]
            * 1000
            / aplicaciones_etapa
        ),
    }
    for nombre, fraccion in etapas
])

tab_plan, tab_catalogo, tab_calculo = st.tabs([
    "Plan nutricional",
    "Fertilizantes",
    "Formulador",
])


# ============================================================
# PLAN
# ============================================================

with tab_plan:
    c1, c2, c3 = st.columns(3)
    c1.metric("Superficie", f"{superficie:,.0f} m²")
    c2.metric("Plantas", f"{plantas:,}")
    c3.metric("Aplicaciones", aplicaciones_totales)

    st.dataframe(plan.round(4), hide_index=True, use_container_width=True)

    st.info(
        f"Calendario de fertilización: {3 * semanas_etapa} semanas "
        f"({dias_calendario} días naturales). "
        f"Ciclo de referencia: {dias_referencia} días. "
        "Son conceptos distintos; no se redistribuyen automáticamente."
    )

    st.warning(
        "89,3 kg/ha de MgO es la referencia acordada, no una "
        "recomendación universal. El resto de objetivos debe definirse "
        "con el análisis de agua, suelo o sustrato y el criterio técnico."
    )


# ============================================================
# CATÁLOGO
# ============================================================

with tab_catalogo:
    st.subheader("Catálogo de trabajo de esta sesión")

    c1, c2, c3 = st.columns(3)
    c1.metric("Productos", len(tabla))
    c2.metric(
        "Composiciones confirmadas",
        int(tabla["Composición confirmada"].sum()),
    )
    c3.metric(
        "Aptos para cálculo",
        int(tabla["Apto para cálculo"].sum()),
    )

    st.dataframe(tabla, hide_index=True, use_container_width=True)

    st.caption(
        "N, P, K, Ca, Mg, S y micronutrientes se muestran como "
        "porcentajes elementales. Las formas de N ya están incluidas "
        "en el N total. Las celdas vacías indican datos desconocidos."
    )

    detalle_id = st.selectbox(
        "Consultar registro completo",
        list(catalogo["productos"]),
        format_func=lambda k: f"{catalogo['productos'][k]['nombre']} · {k}",
    )
    st.json(catalogo["productos"][detalle_id])

    st.info(
        "Los nutrientes no declarados se convierten a cero solamente "
        "al aplicar un perfil confirmado. El resto de valores null "
        "permanece como desconocido. Esta pantalla no retira bloqueos."
    )

    with st.expander("Metadatos originales"):
        st.json({
            k: v for k, v in catalogo.items()
            if k != "productos"
        })


# ============================================================
# FORMULADOR
# ============================================================

with tab_calculo:
    st.subheader("Balance por aplicación y para todo el cultivo")

    etapa = st.selectbox("Etapa", plan["Etapa"].tolist())
    fila_plan = plan.loc[plan["Etapa"] == etapa].iloc[0]

    volumen = st.number_input(
        "Agua total entregada al cultivo por aplicación (litros)",
        min_value=0.0,
        value=0.0,
        step=100.0,
        help=(
            "No es el volumen del tanque madre. Introduce el volumen "
            "de agua realmente aplicado al cultivo."
        ),
    )

    mg_referencia = float(
        fila_plan["Mg elemental (g · cultivo · aplicación)"]
    )
    st.info(
        f"Referencia para {etapa}: {mg_referencia:.3f} g de Mg "
        "elemental por aplicación en todo el cultivo."
    )

    st.markdown("**Objetivos y análisis del agua**")
    st.caption(
        "Activa únicamente nutrientes con objetivo definido. "
        "Los objetivos están en gramos elementales por aplicación "
        "para todo el cultivo. El agua se expresa en mg/L del elemento, "
        "no del ion ni del óxido."
    )

    objetivos_base = pd.DataFrame({
        "Elemento": ELEMENTOS,
        "Usar objetivo": [e == "Mg" for e in ELEMENTOS],
        "Objetivo total (g/aplicación)": [
            mg_referencia if e == "Mg" else 0.0
            for e in ELEMENTOS
        ],
        "Agua (mg/L elemental)": [0.0] * len(ELEMENTOS),
    })

    objetivos = st.data_editor(
        objetivos_base,
        hide_index=True,
        disabled=["Elemento"],
        use_container_width=True,
        key=(
            f"objetivos_{etapa}_{superficie}_{referencia_mgo}_"
            f"{aplicaciones_etapa}"
        ),
        column_config={
            "Usar objetivo": st.column_config.CheckboxColumn(),
            "Objetivo total (g/aplicación)": (
                st.column_config.NumberColumn(min_value=0.0)
            ),
            "Agua (mg/L elemental)": (
                st.column_config.NumberColumn(min_value=0.0)
            ),
        },
    )

    disponibles = tabla.loc[tabla["Apto para cálculo"]].copy()

    if disponibles.empty:
        st.warning(
            "No hay productos aptos para cálculo. Confirmar porcentajes "
            "no elimina bloqueos de uso. Consulta los motivos individuales "
            "en la pestaña Fertilizantes."
        )

    else:
        nombres = disponibles.set_index("ID")["Producto"].to_dict()

        ids = st.multiselect(
            "Productos autorizados que participarán",
            disponibles["ID"].tolist(),
            format_func=lambda k: f"{nombres[k]} · {k}",
        )

        if ids:
            seleccionados = (
                disponibles.set_index("ID").loc[ids].reset_index()
            )

            st.markdown("**Límites por producto**")
            st.caption(
                "Introduce un límite técnico en gramos por aplicación. "
                "Un límite de cero excluye el producto. "
                "No se asignan máximos agronómicos automáticamente."
            )

            limites = st.data_editor(
                pd.DataFrame({
                    "ID": ids,
                    "Producto": [nombres[k] for k in ids],
                    "Máximo (g/aplicación)": [0.0] * len(ids),
                }),
                hide_index=True,
                disabled=["ID", "Producto"],
                use_container_width=True,
                key="limites_" + "|".join(ids),
                column_config={
                    "Máximo (g/aplicación)": (
                        st.column_config.NumberColumn(min_value=0.0)
                    ),
                },
            )

            revisar = st.checkbox(
                "He revisado objetivos, análisis del agua, límites "
                "y compatibilidad de los productos"
            )

            if st.button("Calcular balance", type="primary"):
                try:
                    if lsq_linear is None:
                        raise ValueError(
                            "Falta scipy. Agrégalo a requirements.txt."
                        )
                    if volumen <= 0:
                        raise ValueError(
                            "Introduce el volumen de agua por aplicación."
                        )
                    if not revisar:
                        raise ValueError(
                            "Confirma la revisión técnica de los datos."
                        )

                    objetivos_num = objetivos[
                        [
                            "Objetivo total (g/aplicación)",
                            "Agua (mg/L elemental)",
                        ]
                    ].to_numpy(dtype=float)

                    maximos = limites[
                        "Máximo (g/aplicación)"
                    ].to_numpy(dtype=float)

                    if (
                        not np.isfinite(objetivos_num).all()
                        or (objetivos_num < 0).any()
                        or not np.isfinite(maximos).all()
                        or (maximos < 0).any()
                    ):
                        raise ValueError(
                            "Los objetivos, el agua y los límites deben "
                            "ser números finitos no negativos."
                        )

                    activos = objetivos[
                        "Usar objetivo"
                    ].fillna(False).to_numpy(dtype=bool)

                    if not activos.any():
                        raise ValueError("Activa al menos un objetivo.")

                    permitidos = maximos > 0
                    if not permitidos.any():
                        raise ValueError(
                            "Introduce al menos un límite mayor que cero."
                        )

                    # g de elemento por cada g de fertilizante.
                    matriz = (
                        seleccionados[ELEMENTOS]
                        .to_numpy(dtype=float).T / 100
                    )

                    objetivo_total = objetivos_num[:, 0]
                    aporte_agua = objetivos_num[:, 1] * volumen / 1000
                    faltante = np.maximum(
                        objetivo_total - aporte_agua, 0.0
                    )

                    # Ajuste relativo; para objetivos cero se utiliza
                    # una escala numérica de 1 g, no un umbral agronómico.
                    escalas = np.where(
                        objetivo_total[activos] > 0,
                        objetivo_total[activos],
                        1.0,
                    )

                    a = matriz[activos][:, permitidos]
                    a = a / escalas[:, None]
                    b = faltante[activos] / escalas

                    solucion = lsq_linear(
                        a,
                        b,
                        bounds=(
                            np.zeros(int(permitidos.sum())),
                            maximos[permitidos],
                        ),
                        method="trf",
                    )

                    if not solucion.success:
                        raise ValueError(
                            "El ajuste no convergió: " + solucion.message
                        )

                    dosis = np.zeros(len(ids))
                    dosis[permitidos] = solucion.x
                    aporte_fertilizantes = matriz @ dosis
                    aporte_total = aporte_fertilizantes + aporte_agua

                    resultado_dosis = pd.DataFrame({
                        "Producto": seleccionados["Producto"].tolist(),
                        "g/aplicación · cultivo": dosis,
                        "g/planta · aplicación": dosis / plantas,
                        "kg/etapa · cultivo": (
                            dosis * aplicaciones_etapa / 1000
                        ),
                    })

                    st.markdown("**Dosis del ajuste matemático**")
                    st.dataframe(
                        resultado_dosis.round(5),
                        hide_index=True,
                        use_container_width=True,
                    )

                    balance = pd.DataFrame({
                        "Elemento": ELEMENTOS,
                        "Objetivo activo": activos,
                        "Objetivo (g)": np.where(
                            activos, objetivo_total, np.nan
                        ),
                        "Agua (g)": aporte_agua,
                        "Fertilizantes (g)": aporte_fertilizantes,
                        "Total (g)": aporte_total,
                        "Diferencia total − objetivo (g)": np.where(
                            activos,
                            aporte_total - objetivo_total,
                            np.nan,
                        ),
                        "Concentración total (mg/L)": (
                            aporte_total * 1000 / volumen
                        ),
                    })

                    st.markdown("**Balance elemental por aplicación**")
                    st.dataframe(
                        balance.round(5),
                        hide_index=True,
                        use_container_width=True,
                    )

                    detalle_n = []
                    usados = dosis > 1e-8

                    for clave, etiqueta in [
                        ("N_nitrico", "N nítrico"),
                        ("N_amoniacal", "N amoniacal"),
                        ("N_ureico", "N ureico"),
                    ]:
                        porcentajes = seleccionados[
                            clave
                        ].to_numpy(dtype=float)

                        gramos = np.nan
                        if np.isfinite(porcentajes[usados]).all():
                            gramos = float(np.sum(
                                porcentajes[usados]
                                * dosis[usados] / 100
                            ))

                        detalle_n.append({
                            "Forma": etiqueta,
                            "Fertilizantes (g/aplicación)": gramos,
                            "Añadido al agua (mg/L de N)": (
                                gramos * 1000 / volumen
                            ),
                        })

                    st.markdown("**Desglose del N de los fertilizantes**")
                    st.dataframe(
                        pd.DataFrame(detalle_n).round(5),
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.caption(
                        "Estas formas ya están incluidas en el N total. "
                        "No se suman otra vez. No se desglosa el N del agua."
                    )

                    if (aporte_agua[activos] > objetivo_total[activos]).any():
                        st.warning(
                            "El agua por sí sola supera al menos un "
                            "objetivo. El fertilizante no puede retirar "
                            "ese exceso."
                        )

                    if (~activos).any():
                        st.warning(
                            "Hay nutrientes sin objetivo: se muestran "
                            "sus aportes, pero no se limita su exceso. "
                            "Revisa especialmente B, Cu y Mo antes de "
                            "considerar una aplicación."
                        )

                    for identificador in ids:
                        observaciones = lista_textos(
                            catalogo["productos"][identificador].get(
                                "observaciones"
                            )
                        )
                        if observaciones:
                            st.info(
                                nombres[identificador] + ": "
                                + " | ".join(observaciones)
                            )

                    st.warning(
                        "Resultado matemático, no receta autorizada. "
                        "El ajuste puede dejar déficits o excesos y no "
                        "optimiza costos. Las concentraciones son "
                        "promedios del volumen declarado, no del tanque "
                        "madre ni del pulso de inyección. No se calcula "
                        "pH, CE, precipitación ni compatibilidad."
                    )

                except Exception as exc:
                    st.error(str(exc))
        else:
            st.info("Selecciona al menos un producto apto.")
