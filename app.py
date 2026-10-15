import json
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import lsq_linear


st.set_page_config(
    page_title="TOMATE · Formulador",
    page_icon="🍅",
    layout="wide",
)

ELEMENTOS = [
    "N", "P", "K", "Ca", "Mg", "S",
    "Fe", "Mn", "Zn", "B", "Cu", "Mo", "Co",
]
MACROS = ["N", "P", "K", "Ca", "Mg", "S"]
ETAPAS = ["Vegetativo", "Desarrollo", "Llenado"]

FILAS = [
    ("N", 293.3, 30, 40, 30, "N", 1.0),
    ("P2O5", 152.0, 38, 32, 30, "P", 0.4364),
    ("K2O", 626.7, 23, 32, 45, "K", 0.8301),
    ("CaO", 296.0, 30, 40, 30, "Ca", 0.7147),
    ("MgO", 89.3, 30, 40, 30, "Mg", 0.6030),
    ("S", 84.0, 38, 32, 30, "S", 1.0),
    ("Fe", 4.3, 25, 30, 45, "Fe", 1.0),
    ("Mn", 2.0, 25, 30, 45, "Mn", 1.0),
    ("Zn", 1.07, 25, 30, 45, "Zn", 1.0),
    ("B", 0.80, 25, 30, 45, "B", 1.0),
    ("Cu", 0.67, 25, 30, 45, "Cu", 1.0),
    ("Mo", 0.01, 25, 30, 45, "Mo", 1.0),
    ("Co", 0.13, 25, 30, 45, "Co", 1.0),
]

FACTORES_PLAN = {r[0]: (r[5], r[6]) for r in FILAS}
OXIDOS = {
    "P": ("P2O5", 0.4364),
    "K": ("K2O", 0.8301),
    "Ca": ("CaO", 0.7147),
    "Mg": ("MgO", 0.6030),
    "S": ("SO3", 0.4005),
}

COLUMNAS = [
    "ID", "Producto", "Estado",
    "Bloqueo original", "Apto para cálculo",
    "Motivos", "Desconocidos",
] + ELEMENTOS


def porcentaje(valor, etiqueta):
    """None sigue siendo desconocido. Nunca se reemplaza por cero."""
    if valor is None:
        return np.nan
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ValueError(f"{etiqueta}: se esperaba número o null.")
    valor = float(valor)
    if not np.isfinite(valor) or not 0 <= valor <= 100:
        raise ValueError(f"{etiqueta}: porcentaje fuera de 0–100.")
    return valor


def lista_textos(valor):
    if valor is None:
        return []
    if isinstance(valor, list):
        return [str(x) for x in valor]
    return [str(valor)]


def interpretar_producto(identificador, producto):
    if not isinstance(producto, dict):
        raise ValueError(f"{identificador}: producto inválido.")

    nombre = producto.get("nombre")
    if not isinstance(nombre, str) or not nombre.strip():
        raise ValueError(f"{identificador}: falta nombre.")

    nombre = nombre.strip()
    nutrientes = producto.get("nutrientes_pct", {})
    equivalencias = producto.get("equivalencias_elementales_pct", {})

    if not isinstance(nutrientes, dict):
        raise ValueError(f"{nombre}: nutrientes_pct debe ser un objeto.")
    if not isinstance(equivalencias, dict):
        raise ValueError(
            f"{nombre}: equivalencias_elementales_pct debe ser un objeto."
        )

    motivos = []
    valores = {}

    for elemento in ELEMENTOS:
        clave = "N_total" if elemento == "N" else elemento
        candidatos = []

        directo = porcentaje(
            nutrientes.get(clave), f"{nombre} / {clave}"
        )
        if np.isfinite(directo):
            candidatos.append(("declaración elemental", directo))

        if elemento in OXIDOS:
            oxido, factor = OXIDOS[elemento]
            valor_oxido = porcentaje(
                nutrientes.get(oxido), f"{nombre} / {oxido}"
            )
            if np.isfinite(valor_oxido):
                candidatos.append((oxido, valor_oxido * factor))

        equivalente = porcentaje(
            equivalencias.get(elemento),
            f"{nombre} / equivalencia {elemento}",
        )
        if np.isfinite(equivalente):
            candidatos.append(("equivalencia", equivalente))

        if candidatos:
            # Se usa una sola representación: nunca se suman equivalentes.
            valores[elemento] = candidatos[0][1]
            for _, otro in candidatos[1:]:
                if not np.isclose(
                    valores[elemento], otro,
                    rtol=0.002, atol=0.0001,
                ):
                    motivos.append(
                        f"Declaraciones contradictorias para {elemento}"
                    )
        else:
            valores[elemento] = np.nan

    # Las formas de N no se suman al N total.
    formas = [
        porcentaje(nutrientes.get(k), f"{nombre} / {k}")
        for k in ["N_nitrico", "N_amoniacal", "N_ureico"]
    ]
    conocidas = [x for x in formas if np.isfinite(x)]
    if np.isfinite(valores["N"]) and conocidas:
        if sum(conocidas) > valores["N"] + 0.02:
            motivos.append("Las formas de N superan el N total")
        if len(conocidas) == 3 and not np.isclose(
            sum(conocidas), valores["N"], atol=0.02, rtol=0
        ):
            motivos.append("El desglose completo de N no coincide con N total")

    if sum(v for v in valores.values() if np.isfinite(v)) > 100.01:
        raise ValueError(f"{nombre}: suma elemental superior a 100%.")

    bloqueo = producto.get("bloqueado_dosificacion_automatica", True)
    if not isinstance(bloqueo, bool):
        raise ValueError(f"{nombre}: el bloqueo debe ser true o false.")

    estado = str(producto.get("estado", "sin_validar"))
    if bloqueo:
        motivos.append("Bloqueo original del catálogo")

    # No se infiere verificación ni solubilidad por el nombre comercial.
    # Estos campos son requisitos de una futura versión validada del catálogo.
    if producto.get("verificado") is not True:
        motivos.append("Sin verificación documental explícita")
    if producto.get("soluble") is not True:
        motivos.append("Solubilidad para fertirriego sin confirmar")

    pendientes = lista_textos(producto.get("pendientes"))
    if pendientes:
        motivos.append("Pendientes técnicos sin resolver")

    criterios = producto.get("criterios_porcentaje") or {}
    if not isinstance(criterios, dict):
        raise ValueError(f"{nombre}: criterios_porcentaje inválido.")
    if any(v not in (None, "", "=") for v in criterios.values()):
        motivos.append("Contiene umbrales, no porcentajes exactos")

    if "pendiente" in estado.casefold():
        motivos.append("Estado documental pendiente")

    desconocidos = [
        e for e, valor in valores.items() if not np.isfinite(valor)
    ]
    if desconocidos:
        motivos.append("Composición elemental incompleta para el balance")

    if "abonis" in nombre.casefold():
        motivos.append("Abono de fondo excluido del Venturi")

    if not any(np.isfinite(v) and v > 0 for v in valores.values()):
        motivos.append("Sin aportes cuantificados positivos")

    return {
        "ID": identificador,
        "Producto": nombre,
        "Estado": estado,
        "Bloqueo original": bloqueo,
        "Apto para cálculo": not motivos,
        "Motivos": " | ".join(dict.fromkeys(motivos)),
        "Desconocidos": ", ".join(desconocidos),
        **valores,
    }


def leer_catalogo(contenido):
    if not isinstance(contenido, dict):
        raise ValueError("La raíz del JSON debe ser un objeto.")

    if contenido.get("version_esquema") != 1:
        raise ValueError(
            "Se requiere el catálogo original con version_esquema: 1."
        )

    productos = contenido.get("productos")
    if not isinstance(productos, dict) or not productos:
        raise ValueError("productos debe ser un diccionario no vacío.")

    filas = [
        interpretar_producto(str(k), producto)
        for k, producto in productos.items()
    ]
    tabla = pd.DataFrame(filas, columns=COLUMNAS)

    if tabla["Producto"].duplicated().any():
        raise ValueError("Hay nombres de producto repetidos.")

    contexto = contenido.get("contexto_cultivo") or {}
    if not isinstance(contexto, dict):
        raise ValueError("contexto_cultivo debe ser un objeto.")

    return tabla, contexto


def numero_contexto(contexto, clave, defecto):
    valor = contexto.get(clave, defecto)
    if (
        isinstance(valor, bool)
        or not isinstance(valor, (int, float))
        or not np.isfinite(valor)
        or valor <= 0
    ):
        raise ValueError(f"contexto_cultivo / {clave}: valor inválido.")
    return float(valor)


def validar_no_negativos(datos, mensaje):
    matriz = np.asarray(datos, dtype=float)
    if not np.isfinite(matriz).all() or (matriz < 0).any():
        st.error(mensaje)
        st.stop()


st.title("🍅 TOMATE · Plan y formulador de fertirriego")
st.caption(
    "Simulación para revisión técnica. No controla equipos "
    "ni certifica dosis, solubilidad o compatibilidad."
)

# ---------- IMPORTACIÓN ----------
with st.sidebar:
    st.header("Catálogo original")
    archivo = st.file_uploader(
        "Carga catalogo-fertilizantes.json",
        type=["json"],
    )

    if archivo is not None:
        datos_archivo = archivo.getvalue()
        huella = hashlib.sha256(datos_archivo).hexdigest()

        if huella != st.session_state.get("huella_catalogo"):
            try:
                original = json.loads(datos_archivo.decode("utf-8-sig"))
                tabla_nueva, contexto_nuevo = leer_catalogo(original)

                area_nueva = numero_contexto(
                    contexto_nuevo, "superficie_m2", 2500
                )
                plantas_nuevas = numero_contexto(
                    contexto_nuevo, "plantas", 5000
                )
                if not plantas_nuevas.is_integer():
                    raise ValueError("El número de plantas debe ser entero.")

                st.session_state["catalogo_original"] = original
                st.session_state["catalogo_tabla"] = tabla_nueva
                st.session_state["contexto_catalogo"] = contexto_nuevo
                st.session_state["superficie"] = area_nueva
                st.session_state["plantas"] = int(plantas_nuevas)
                st.session_state["area_ok"] = False
                st.session_state["hidraulica_ok"] = False
                st.session_state["huella_catalogo"] = huella
            except Exception as exc:
                st.error(f"No se pudo importar: {exc}")
                st.stop()

catalogo = st.session_state.get(
    "catalogo_tabla", pd.DataFrame(columns=COLUMNAS)
)
original = st.session_state.get("catalogo_original", {})
contexto = st.session_state.get("contexto_catalogo", {})

st.session_state.setdefault("superficie", 2500.0)
st.session_state.setdefault("plantas", 5000)

# ---------- CONFIGURACIÓN ----------
with st.sidebar:
    st.header("Cultivo")
    plantas = st.number_input(
        "Plantas totales", min_value=1, step=100, key="plantas"
    )
    sectores = st.number_input(
        "Sectores iguales", min_value=1, value=4, step=1
    )
    superficie = st.number_input(
        "Superficie cultivada · m²",
        min_value=1.0, step=10.0, key="superficie",
    )
    area_ok = st.checkbox(
        "Confirmo la superficie cultivada", key="area_ok"
    )
    st.info(
        "3 etapas × 5 semanas × 5 aplicaciones/semana "
        "= 75 aplicaciones."
    )

    st.header("Riego e inyección")
    caudal = st.number_input(
        "Agua antes del Venturi · L/min por sector",
        min_value=0.1, value=125.0,
    )
    minutos = st.number_input(
        "Riego por sector · minutos",
        min_value=0.1, value=15.0,
    )
    inyeccion = st.number_input(
        "Inyección por sector · minutos",
        min_value=0.1, value=10.0,
    )
    succion = st.number_input(
        "Succión medida · L/min",
        min_value=0.01, value=1.0,
    )
    tanque = st.number_input(
        "Volumen final de solución madre · L",
        min_value=1.0, value=200.0,
    )
    hidraulica_ok = st.checkbox(
        "Caudal y succión comprobados en campo",
        key="hidraulica_ok",
    )
    st.caption(
        "Una aplicación por jornada de fertilización en todos los "
        "sectores. Sectores con igual caudal, duración y reparto."
    )

if inyeccion > minutos:
    st.error("La inyección no puede durar más que el riego.")
    st.stop()

ha = superficie / 10000
dias_semana = 5
dias_etapa = 25
agua_sector = caudal * minutos
madre_sector = succion * inyeccion
agua_dia = agua_sector * sectores
madre_dia = madre_sector * sectores
final_dia = agua_dia + madre_dia
final_inyeccion = (caudal + succion) * inyeccion * sectores

tabs = st.tabs([
    "1 · Etapas",
    "2 · Agua y ppm",
    "3 · Fertilizantes",
    "4 · Formulador",
    "5 · Tanque y alertas",
])

# ---------- ETAPAS ----------
with tabs[0]:
    st.subheader("Distribución por etapas")
    referencia = contexto.get("ciclo_referencia_dias", "no indicado")
    st.caption(
        f"Ciclo registrado en el catálogo: {referencia} días. "
        "Calendario simulado: 15 semanas, 105 días calendario "
        "y 75 jornadas de fertilización."
    )
    st.warning(
        "Los totales kg/ha son la referencia de tu código anterior. "
        "Importar el catálogo no verifica su origen ni que correspondan "
        "al ciclo registrado. Redistribuirlos requiere revisión agronómica."
    )

    base = pd.DataFrame(
        [r[:5] for r in FILAS],
        columns=["Declaración", "kg/ha ciclo"] + ETAPAS,
    )
    plan = st.data_editor(
        base,
        disabled=["Declaración"],
        hide_index=True,
        key="plan_etapas",
        column_config={
            "kg/ha ciclo": st.column_config.NumberColumn(min_value=0.0),
            **{
                e: st.column_config.NumberColumn(
                    f"{e} · %", min_value=0.0, max_value=100.0
                )
                for e in ETAPAS
            },
        },
    )
    validar_no_negativos(
        plan[["kg/ha ciclo"] + ETAPAS],
        "Completa el plan con números finitos no negativos.",
    )
    if (
        (plan[ETAPAS] > 100).any().any()
        or not np.allclose(
            plan[ETAPAS].sum(axis=1), 100, atol=1e-6, rtol=0
        )
    ):
        st.error("Cada fila debe repartir exactamente 100% entre etapas.")
        st.stop()

    etapa = st.selectbox("Etapa de la simulación", ETAPAS)
    indice = ETAPAS.index(etapa)
    st.write(
        f"Semanas **{indice * 5 + 1} a {(indice + 1) * 5}** · "
        "**25 aplicaciones**."
    )

    objetivo_g = pd.Series(0.0, index=ELEMENTOS)
    detalle = []
    for _, fila in plan.iterrows():
        elemento, factor = FACTORES_PLAN[fila["Declaración"]]
        kg = fila["kg/ha ciclo"] * ha * fila[etapa] / 100
        gramos = kg * 1000 * factor / dias_etapa
        objetivo_g[elemento] += gramos
        detalle.append({
            "Declaración original": fila["Declaración"],
            "kg declarados/etapa": kg,
            "Elemento": elemento,
            "g elementales/día · cultivo": gramos,
            "g elementales/día · sector": gramos / sectores,
            "kg elementales/semana": gramos * dias_semana / 1000,
        })
    st.dataframe(pd.DataFrame(detalle).round(5), hide_index=True)
    st.caption(
        "Día significa jornada de fertilización. Son masas de nutrientes, "
        "no masas de productos comerciales. No se presupone que el "
        "cobalto de la referencia sea necesario para tomate."
    )

# ---------- AGUA ----------
with tabs[1]:
    st.subheader("Agua y concentraciones elementales")
    a, b, c = st.columns(3)
    a.metric("Agua portadora / jornada", f"{agua_dia:,.1f} L")
    b.metric("Solución madre / jornada", f"{madre_dia:,.1f} L")
    c.metric("Volumen final / planta", f"{final_dia / plantas:.3f} L")

    st.write(
        f"Riego secuencial: **{minutos * sectores:.1f} min**. "
        f"Succión / agua portadora: **{100 * succion / caudal:.2f}%**."
    )
    st.caption(
        "El caudal es el medido antes de añadir solución madre. "
        "Se aproximan volúmenes aditivos y succión constante."
    )

    agua_df = st.data_editor(
        pd.DataFrame({
            "Elemento": ELEMENTOS,
            "ppm agua": np.zeros(len(ELEMENTOS)),
        }),
        disabled=["Elemento"],
        hide_index=True,
        key="analisis_agua",
        column_config={
            "ppm agua": st.column_config.NumberColumn(min_value=0.0),
        },
    )
    ppm_agua = agua_df.set_index("Elemento")["ppm agua"].reindex(ELEMENTOS)
    validar_no_negativos(ppm_agua, "Revisa los valores del agua.")

    analisis_ok = st.checkbox(
        "Análisis confirmado para los elementos modelados"
    )
    misma_agua = st.checkbox(
        "La solución madre se prepara con la misma agua analizada"
    )
    st.caption(
        "Introduce mg/L del elemento, no del ion: NO₃ no equivale a N. "
        "Un cero no debe usarse para sustituir un dato desconocido."
    )

    # Modelo válido si la solución madre utiliza la misma agua.
    agua_g = ppm_agua * final_dia / 1000
    ppm_plan = objetivo_g * 1000 / final_dia

    modo = st.radio(
        "Objetivo",
        ["Distribución por etapas", "PPM manuales"],
        horizontal=True,
    )
    if modo == "PPM manuales":
        manual = st.data_editor(
            pd.DataFrame({
                "Elemento": ELEMENTOS,
                "ppm objetivo": ppm_plan.values,
            }),
            disabled=["Elemento"],
            hide_index=True,
            key=f"ppm_manual_{etapa}",
            column_config={
                "ppm objetivo": st.column_config.NumberColumn(min_value=0.0),
            },
        )
        ppm_obj = manual.set_index("Elemento")["ppm objetivo"].reindex(
            ELEMENTOS
        )
    else:
        ppm_obj = ppm_plan.copy()

    validar_no_negativos(ppm_obj, "Revisa las ppm objetivo.")
    objetivo_total_g = ppm_obj * final_dia / 1000
    deficit_g = (objetivo_total_g - agua_g).clip(lower=0)

    st.dataframe(pd.DataFrame({
        "Elemento": ELEMENTOS,
        "Objetivo promedio · ppm": ppm_obj.values,
        "Agua · ppm": ppm_agua.values,
        "Déficit · g/jornada": deficit_g.values,
        "Durante inyección si se cubre el déficit · ppm":
            (ppm_agua + deficit_g * 1000 / final_inyeccion).values,
    }).round(5), hide_index=True)

    if (agua_g > objetivo_total_g + 1e-9).any():
        st.warning(
            "El agua supera algún objetivo. Añadir fertilizantes "
            "no elimina ese exceso."
        )
    if not (analisis_ok and misma_agua):
        st.warning(
            "Balance de agua provisional. El cálculo de productos requiere "
            "confirmar el análisis y el origen del agua de preparación."
        )
    st.info(
        "Abonisa no se inyecta ni se descuenta automáticamente: su masa "
        "aplicada al suelo no determina su disponibilidad diaria."
    )

# ---------- CATÁLOGO ----------
with tabs[2]:
    st.subheader("Tu catálogo original")
    if catalogo.empty:
        st.info("Carga tu JSON en la barra lateral.")
    else:
        total = len(catalogo)
        aptos = int(catalogo["Apto para cálculo"].sum())
        a, b, c = st.columns(3)
        a.metric("Productos importados", total)
        b.metric(
            "Bloqueos originales",
            int(catalogo["Bloqueo original"].sum()),
        )
        c.metric("Aptos para cálculo", aptos)

        st.dataframe(
            catalogo.drop(columns=["ID"]).round(5),
            hide_index=True,
        )
        st.caption(
            "Celdas vacías = desconocido, no cero. Los porcentajes "
            "mostrados son elementales. Los datos originales se conservan."
        )

        ids = catalogo["ID"].tolist()
        nombres = catalogo.set_index("ID")["Producto"].to_dict()
        elegido = st.selectbox(
            "Consultar producto",
            ids,
            format_func=lambda x: nombres[x],
        )
        fila = catalogo.set_index("ID").loc[elegido]
        producto = original["productos"][elegido]

        if fila["Apto para cálculo"]:
            st.success("Cumple los requisitos de entrada del cálculo.")
        else:
            st.warning(fila["Motivos"])

        st.markdown("**Pendientes del catálogo**")
        pendientes = lista_textos(producto.get("pendientes"))
        if pendientes:
            for texto in pendientes:
                st.write(f"• {texto}")
        else:
            st.write(
                "Sin pendientes específicos anotados. Esto no equivale "
                "a validación documental."
            )

        st.markdown("**Observaciones originales**")
        for texto in lista_textos(producto.get("observaciones")):
            st.write(f"• {texto}")

        with st.expander("Ver ficha original completa"):
            st.json(producto)

        st.info(
            "Esta pantalla no permite desbloquear productos. Para una "
            "futura ficha validada se requieren verificación, solubilidad, "
            "ausencia de pendientes y composición suficiente para el balance. "
            "No se deben rellenar desconocidos con cero sin respaldo técnico."
        )

# ---------- FORMULADOR ----------
resultado = None
seleccionados = pd.DataFrame()
dosis_g = None

with tabs[3]:
    st.subheader("Formulación no negativa")
    st.caption(
        "Los productos con micronutrientes no se excluyen por contenerlos. "
        "Se exige composición conocida para todos los elementos modelados "
        "y se conservan los bloqueos originales."
    )

    disponibles = catalogo.loc[
        catalogo["Apto para cálculo"].eq(True)
    ].copy()

    nombres = st.multiselect(
        "Productos validados a utilizar",
        disponibles["Producto"].tolist(),
    )
    confirmado = st.checkbox(
        "Entiendo que el resultado es una simulación pendiente "
        "de revisión agronómica"
    )

    if disponibles.empty:
        st.warning(
            "No hay productos aptos. Con tu catálogo actual esto es "
            "esperable: los 18 conservan bloqueo documental. "
            "Consulta sus pendientes en Fertilizantes."
        )
    elif not nombres:
        st.info("Selecciona productos validados.")
    elif not all([
        area_ok, hidraulica_ok, analisis_ok, misma_agua, confirmado
    ]):
        st.warning(
            "Confirma superficie, hidráulica, análisis del agua, agua "
            "de preparación y revisión técnica para calcular."
        )
    else:
        seleccionados = disponibles.loc[
            disponibles["Producto"].isin(nombres)
        ].copy().reset_index(drop=True)

        A = seleccionados[ELEMENTOS].to_numpy(dtype=float).T / 100
        b = deficit_g.to_numpy(dtype=float)

        # Ajuste relativo de TODOS los elementos, incluidos micros.
        # Los pisos numéricos son ponderaciones, no límites de seguridad.
        piso_ppm = np.array([
            0.1 if e in MACROS else 0.001 for e in ELEMENTOS
        ])
        escala = np.maximum(b, piso_ppm * final_dia / 1000)

        ajuste = lsq_linear(
            A / escala[:, None],
            b / escala,
            bounds=(0, np.inf),
            tol=1e-10,
            max_iter=1000,
        )

        if not ajuste.success or not np.isfinite(ajuste.x).all():
            st.error("El ajuste no convergió. Revisa objetivos y productos.")
        else:
            dosis_g = np.maximum(ajuste.x, 0)
            dosis_g[dosis_g < 1e-8] = 0

            resultado = pd.DataFrame({
                "Producto": seleccionados["Producto"],
                "g/jornada · cultivo": dosis_g,
                "g/jornada · sector": dosis_g / sectores,
                "kg/semana · cultivo": dosis_g * 5 / 1000,
                "kg/etapa · cultivo": dosis_g * 25 / 1000,
            })
            st.dataframe(resultado.round(6), hide_index=True)
            st.caption(
                "Semana = 5 aplicaciones; etapa = 25 aplicaciones "
                "idénticas. No multipliques otra vez por los sectores."
            )

            aportes = A @ dosis_g
            alcanzado = (
                aportes + agua_g.to_numpy(dtype=float)
            ) * 1000 / final_dia
            durante = (
                ppm_agua.to_numpy(dtype=float)
                + aportes * 1000 / final_inyeccion
            )
            diferencia = alcanzado - ppm_obj.to_numpy(dtype=float)

            st.dataframe(pd.DataFrame({
                "Elemento": ELEMENTOS,
                "Objetivo promedio · ppm": ppm_obj.values,
                "Alcanzado promedio · ppm": alcanzado,
                "Diferencia · ppm": diferencia,
                "Durante inyección · ppm": durante,
            }).round(6), hide_index=True)

            tolerancia = np.maximum(
                piso_ppm, 0.05 * ppm_obj.to_numpy(dtype=float)
            )
            if (np.abs(diferencia) > tolerancia).any():
                st.error(
                    "Hay faltantes o excesos fuera del criterio numérico "
                    "de ajuste. No tratar el resultado como receta."
                )
            else:
                st.info(
                    "El ajuste cumple el criterio numérico: 5% del "
                    "objetivo, con piso de 0,1 ppm para macros y "
                    "0,001 ppm para micros. NO son límites agronómicos."
                )

            st.warning(
                "El optimizador no impone límites agronómicos máximos, "
                "no valida seguridad de micronutrientes, no estima CE "
                "y no optimiza formas de nitrógeno. Aproximar objetivos "
                "no equivale a autorizar una aplicación."
            )

# ---------- TANQUE ----------
with tabs[4]:
    st.subheader("Solución madre")
    st.write(
        f"Consumo por sector: **{madre_sector:.2f} L**. "
        f"Consumo por jornada: **{madre_dia:.2f} L**."
    )
    st.write(
        f"Una preparación de **{tanque:.1f} L** equivale a "
        f"**{tanque / madre_dia:.2f} jornadas**, solo hidráulicamente."
    )
    st.caption(
        "No valida estabilidad, almacenamiento ni cambio de etapa."
    )

    if resultado is None:
        st.info(
            "Las cantidades de tanque se mostrarán cuando exista "
            "una formulación con productos validados."
        )
    else:
        mascara = dosis_g > 0
        activos = seleccionados.loc[mascara].copy()
        activos["g_dia"] = dosis_g[mascara]

        if activos.empty:
            st.info("El cálculo no asignó fertilizante.")
        else:
            tiene_ca = activos["Ca"] > 0
            tiene_ps = (activos["P"] > 0) | (activos["S"] > 0)
            conflicto = bool(tiene_ca.any() and tiene_ps.any())

            if conflicto:
                st.error(
                    "Se bloquea la preparación de un concentrado único: "
                    "se detectan aportes de calcio y fósforo/azufre."
                )
                st.caption(
                    "Es una alerta conservadora basada en elementos. "
                    "La especie química real debe verificarse en las fichas."
                )
                grupos = []
                for _, fila in activos.iterrows():
                    ca = fila["Ca"] > 0
                    ps = fila["P"] > 0 or fila["S"] > 0
                    if ca and ps:
                        grupo = "Revisar ficha: ambos grupos"
                    elif ca:
                        grupo = "A · asignación preliminar"
                    elif ps:
                        grupo = "B · asignación preliminar"
                    else:
                        grupo = "Asignación pendiente"
                    grupos.append(grupo)

                st.dataframe(pd.DataFrame({
                    "Producto": activos["Producto"].values,
                    "Grupo preliminar": grupos,
                    "g/jornada · cultivo": activos["g_dia"].values,
                }), hide_index=True)
                st.warning(
                    "No se calculan concentraciones A/B sin definir "
                    "tiempos y succión de cada preparación. Con un único "
                    "Venturi no se puede asignar a ambos grupos todo el "
                    "tiempo disponible. Revisar también pre-riego, lavado "
                    "y tránsito hidráulico."
                )
            else:
                concentracion = activos["g_dia"] / madre_dia
                st.dataframe(pd.DataFrame({
                    "Producto": activos["Producto"].values,
                    "g/L solución madre": concentracion.values,
                    "kg para volumen final preparado":
                        (concentracion * tanque / 1000).values,
                }).round(6), hide_index=True)

                st.warning(
                    "No detectar la alerta calcio–P/S no demuestra "
                    "compatibilidad. Verificar sales, quelatos, pH, "
                    "temperatura y solubilidad de la mezcla."
                )
                st.caption(
                    "Completar hasta el volumen final indicado; no añadir "
                    "los productos a ese volumen de agua ya lleno."
                )
                if tanque < madre_dia:
                    st.warning(
                        "El volumen preparado no alcanza para una jornada. "
                        "Harían falta reposiciones a igual concentración."
                    )

    st.markdown("**Antes de cualquier aplicación**")
    st.markdown(
        "- Revisar necesidad del cultivo, suelo y abonado de fondo.\n"
        "- Confirmar composición y autorización de uso de cada producto.\n"
        "- Comprobar análisis de agua, pH y conductividad eléctrica.\n"
        "- Verificar solubilidad y compatibilidad a concentración real.\n"
        "- No inyectar Abonisa ni productos de composición incierta.\n"
        "- Revisar los aportes totales de micronutrientes.\n"
        "- No considerar la referencia de cobalto como una necesidad validada."
    )

st.caption(
    "El JSON original no se modifica. Los datos cargados permanecen "
    "solo durante la sesión; conserva tu archivo original. "
    "Código pendiente de prueba en tu despliegue."
)
