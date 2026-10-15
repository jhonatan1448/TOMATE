import json

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import lsq_linear


st.set_page_config(
    page_title="TOMATE · Formulador",
    page_icon="🍅",
    layout="wide",
)

st.title("🍅 TOMATE · Plan y formulador de fertirriego")
st.caption(
    "Simulación de objetivos nutricionales. "
    "No controla equipos ni constituye un plan agronómico validado."
)

# Los totales se conservan en las unidades declaradas de la tabla.
ETAPAS = ["Vegetativo", "Desarrollo", "Llenado"]
ELEMENTOS = ["N", "P", "K", "Ca", "Mg", "S", "Fe", "Mn", "Zn", "B", "Cu", "Mo", "Co"]
MACROS = ["N", "P", "K", "Ca", "Mg", "S"]

# Declaración de la referencia y conversión a masa elemental.
FILAS = [
    ("N", 293.3, 30, 40, 30, "N", 1.0),
    ("P2O5", 152.0, 38, 32, 30, "P", 0.4364),
    ("K2O", 626.7, 23, 32, 45, "K", 0.8301),
    ("CaO", 296.0, 30, 40, 30, "Ca", 0.7147),
    ("MgO", 89.3, 30, 40, 30, "Mg", 0.6030),
    ("S", 84.0, 38, 32, 30, "S", 1.0),
    ("Fe", 4.3, 25, 30, 45, "Fe", 1.0),
    ("Mn", 2.00, 25, 30, 45, "Mn", 1.0),
    ("Zn", 1.07, 25, 30, 45, "Zn", 1.0),
    ("B", 0.80, 25, 30, 45, "B", 1.0),
    ("Cu", 0.67, 25, 30, 45, "Cu", 1.0),
    ("Mo", 0.01, 25, 30, 45, "Mo", 1.0),
    ("Co", 0.13, 25, 30, 45, "Co", 1.0),
]
FACTORES = {r[0]: (r[5], r[6]) for r in FILAS}

# Formato del catálogo de ESTA versión:
# nutrientes elementales en porcentaje de masa.
COLUMNAS_PRODUCTO = [
    "Producto", "Verificado", "Soluble", "Bloqueado",
    "Contiene_calcio", "Contiene_fosfatos", "Contiene_sulfatos",
    "N", "P", "K", "Ca", "Mg", "S", "Fe", "Mn", "Zn", "B", "Cu", "Mo", "Co",
]
BOOLEANOS = [
    "Verificado", "Soluble", "Bloqueado",
    "Contiene_calcio", "Contiene_fosfatos", "Contiene_sulfatos",
]


def producto_vacio():
    fila = {c: 0.0 for c in ELEMENTOS}
    fila.update({
        "Producto": "Completar con etiqueta",
        "Verificado": False,
        "Soluble": False,
        "Bloqueado": True,
        "Contiene_calcio": False,
        "Contiene_fosfatos": False,
        "Contiene_sulfatos": False,
    })
    return fila


def validar_catalogo(df):
    faltan = set(COLUMNAS_PRODUCTO) - set(df.columns)
    if faltan:
        raise ValueError("Faltan columnas: " + ", ".join(sorted(faltan)))
    df = df[COLUMNAS_PRODUCTO].copy()
    for c in BOOLEANOS:
        if not df[c].map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError(f"{c} debe contener verdadero/falso, no texto.")
    for c in ELEMENTOS:
        df[c] = pd.to_numeric(df[c], errors="raise")
        if not np.isfinite(df[c]).all() or not df[c].between(0, 100).all():
            raise ValueError(f"{c}: porcentajes finitos entre 0 y 100.")
    if (df[ELEMENTOS].sum(axis=1) > 100.01).any():
        raise ValueError("La suma de porcentajes elementales supera 100%.")
    df["Producto"] = df["Producto"].fillna("").astype(str).str.strip()
    if df["Producto"].eq("").any() or df["Producto"].duplicated().any():
        raise ValueError("Usa nombres de producto únicos y no vacíos.")
    return df


# ---------- CONFIGURACIÓN ----------
with st.sidebar:
    st.header("Cultivo")
    plantas = st.number_input("Plantas totales", 1, value=5000, step=100)
    sectores = st.number_input("Sectores iguales", 1, value=4, step=1)

    superficie = st.number_input(
        "Superficie cultivada real · m²",
        min_value=1.0, value=1800.0, step=10.0,
        help=(
            "1800 m² es solo la estimación 5000 × 0,30 × 1,20. "
            "Confirma la superficie neta representada por la referencia kg/ha."
        ),
    )
    area_ok = st.checkbox("Confirmo la superficie cultivada")

    semanas = 5
    dias_semana = 5
    dias_etapa = semanas * dias_semana

    st.info("3 etapas × 5 semanas · 5 días/semana · 75 aplicaciones")

    st.header("Riego e inyección")
    caudal = st.number_input(
        "Caudal de agua por sector · L/min",
        min_value=0.1, value=125.0, step=1.0,
    )
    minutos = st.number_input(
        "Riego por sector · min",
        min_value=0.1, value=15.0, step=1.0,
    )
    inyeccion = st.number_input(
        "Inyección por sector · min",
        min_value=0.1, value=10.0, step=1.0,
    )
    succion = st.number_input(
        "Succión medida del Venturi · L/min",
        min_value=0.01, value=1.0, step=0.1,
    )
    tanque = st.number_input(
        "Volumen final preparado de solución madre · L",
        min_value=1.0, max_value=200.0, value=200.0, step=1.0,
    )
    hidraulica_ok = st.checkbox("Caudal y succión comprobados en campo")

    st.caption(
        "Se supone una aplicación diaria en todos los sectores, "
        "con igual tiempo y caudal. No se establece un tiempo de riego recomendado."
    )

if inyeccion > minutos:
    st.error("La inyección no puede durar más que el riego.")
    st.stop()

ha = superficie / 10000.0
agua_sector = caudal * minutos
madre_sector = succion * inyeccion
agua_dia = agua_sector * sectores
madre_dia = madre_sector * sectores

# Modelo: caudal ingresado = agua portadora antes del Venturi.
# Se suma el volumen de solución madre al volumen final entregado.
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
    st.warning(
        "La referencia original corresponde a un ciclo de 180 días. "
        "Conservar sus totales en 15 semanas aumenta la intensidad del aporte. "
        "Los valores deben revisarse antes de convertirse en un plan de aplicación."
    )

    base = pd.DataFrame(
        [r[:5] for r in FILAS],
        columns=["Declaración", "kg/ha ciclo"] + ETAPAS,
    )
    plan = st.data_editor(
        base,
        disabled=["Declaración"],
        hide_index=True,
        use_container_width=True,
        key="plan",
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

    columnas_num = ["kg/ha ciclo"] + ETAPAS
    datos_plan = plan[columnas_num].to_numpy(dtype=float)
    if not np.isfinite(datos_plan).all() or (datos_plan < 0).any():
        st.error("Completa los valores del plan con números no negativos.")
        st.stop()
    if not np.allclose(plan[ETAPAS].sum(axis=1), 100.0):
        st.error("Cada nutriente debe sumar exactamente 100% entre las etapas.")
        st.stop()

    etapa = st.selectbox("Etapa para todos los cálculos", ETAPAS)
    indice = ETAPAS.index(etapa)
    st.caption(
        f"Semanas {indice * 5 + 1} a {(indice + 1) * 5} · "
        "25 días de fertilización en esta etapa."
    )

    objetivo_g = pd.Series(0.0, index=ELEMENTOS)
    detalle = []
    for _, r in plan.iterrows():
        declaracion = r["Declaración"]
        elemento, factor = FACTORES[declaracion]
        kg_etapa = r["kg/ha ciclo"] * ha * r[etapa] / 100.0
        g_dia_declarados = kg_etapa * 1000.0 / dias_etapa
        g_dia_elementales = g_dia_declarados * factor
        objetivo_g[elemento] += g_dia_elementales
        detalle.append({
            "Declaración": declaracion,
            "kg/etapa · cultivo": kg_etapa,
            "kg/semana · cultivo": kg_etapa / semanas,
            "g/día · cultivo": g_dia_declarados,
            "g/día · sector": g_dia_declarados / sectores,
            "Elemento": elemento,
            "g elementales/día": g_dia_elementales,
        })

    st.dataframe(
        pd.DataFrame(detalle).round(4),
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "Son cantidades de nutrientes, no cantidades de fertilizante comercial. "
        "El nitrógeno se calcula como N total; esta versión no optimiza "
        "la relación entre N nítrico y N amoniacal."
    )

# ---------- AGUA Y PPM ----------
with tabs[1]:
    st.subheader("Hidráulica y objetivos elementales")
    a, b, c = st.columns(3)
    a.metric("Agua portadora / día", f"{agua_dia:,.0f} L")
    b.metric("Solución madre / día", f"{madre_dia:,.1f} L")
    c.metric(
        "Volumen final / planta",
        f"{final_dia / plantas:.3f} L",
    )

    st.write(
        f"Relación succión/agua: **{100 * succion / caudal:.2f}%**. "
        f"Riego secuencial total: **{minutos * sectores:.1f} minutos**."
    )
    st.caption(
        "El caudal introducido es el agua antes de la inyección. "
        "Si tu medición incluye la solución madre, ajusta este dato "
        "para no contar ese volumen dos veces."
    )

    st.markdown("**Análisis del agua · ppm elementales**")
    st.caption(
        "Introduce resultados expresados como elemento, no como ion u óxido. "
        "Por ejemplo, nitrato NO₃ no equivale a N."
    )
    agua_df = st.data_editor(
        pd.DataFrame({"Elemento": ELEMENTOS, "ppm agua": [0.0] * len(ELEMENTOS)}),
        disabled=["Elemento"], hide_index=True,
        column_config={
            "ppm agua": st.column_config.NumberColumn(min_value=0.0)
        },
        key="agua",
    )
    analisis_ok = st.checkbox("Los valores proceden de un análisis de agua")
    ppm_agua = agua_df.set_index("Elemento")["ppm agua"].reindex(ELEMENTOS)
    if not np.isfinite(ppm_agua).all() or (ppm_agua < 0).any():
        st.error("Completa las ppm del agua con valores no negativos.")
        st.stop()

    # Supone solución madre preparada con la misma agua de origen.
    agua_g = ppm_agua * final_dia / 1000.0
    ppm_del_plan = objetivo_g * 1000.0 / final_dia

    modo = st.radio(
        "Origen del objetivo",
        ["Distribución por etapas", "Ingresar ppm manualmente"],
        horizontal=True,
    )
    if modo == "Ingresar ppm manualmente":
        st.caption(
            "Las ppm manuales reemplazan el objetivo de la etapa para esta "
            "simulación. Son ppm promedio del riego completo."
        )
        manual = st.data_editor(
            pd.DataFrame({
                "Elemento": ELEMENTOS,
                "ppm objetivo": ppm_del_plan.values,
            }),
            disabled=["Elemento"], hide_index=True,
            column_config={
                "ppm objetivo": st.column_config.NumberColumn(min_value=0.0)
            },
            key=f"manual_{etapa}",
        )
        ppm_obj = manual.set_index("Elemento")["ppm objetivo"].reindex(ELEMENTOS)
    else:
        ppm_obj = ppm_del_plan.copy()

    if not np.isfinite(ppm_obj).all() or (ppm_obj < 0).any():
        st.error("Los objetivos deben ser valores finitos y no negativos.")
        st.stop()

    objetivo_total_g = ppm_obj * final_dia / 1000.0
    deficit_g = (objetivo_total_g - agua_g).clip(lower=0.0)

    resumen_ppm = pd.DataFrame({
        "Elemento": ELEMENTOS,
        "Objetivo promedio · ppm": ppm_obj.values,
        "Agua · ppm": ppm_agua.values,
        "Por aportar con fertilizantes · g/día": deficit_g.values,
        "Ppm durante inyección si se alcanza objetivo":
            (ppm_agua + deficit_g * 1000.0 / final_inyeccion).values,
    })
    st.dataframe(
        resumen_ppm.round(4), hide_index=True, use_container_width=True
    )
    if (agua_g > objetivo_total_g + 1e-9).any():
        st.warning(
            "El agua supera algún objetivo. El formulador no puede "
            "eliminar nutrientes ni corregirlo con dosis negativas."
        )
    if not analisis_ok:
        st.warning(
            "Sin análisis confirmado, el descuento del agua es provisional. "
            "Cero significa desconocido si no has medido su composición."
        )
    st.info(
        "Abonisa no se descuenta automáticamente: conocer la masa aplicada "
        "no permite estimar su disponibilidad diaria. Tampoco se incorpora "
        "como producto para el Venturi."
    )

# ---------- CATÁLOGO ----------
with tabs[2]:
    st.subheader("Catálogo editable")
    st.warning(
        "Todos los nutrientes de este catálogo deben estar en % de masa "
        "ELEMENTAL. No copies P₂O₅, K₂O, CaO o MgO directamente en P, K, Ca o Mg."
    )
    st.write(
        "Conversiones: P = P₂O₅ × 0,4364; K = K₂O × 0,8301; "
        "Ca = CaO × 0,7147; Mg = MgO × 0,6030."
    )
    st.caption(
        "No actives Verificado ni Soluble sin comprobar la ficha del producto. "
        "Marca también si contiene calcio, fosfatos o sulfatos. "
        "El programa no verifica etiquetas por sí mismo."
    )

    if "catalogo_base" not in st.session_state:
        st.session_state.catalogo_base = pd.DataFrame([producto_vacio()])
        st.session_state.catalogo_version = 0

    archivo = st.file_uploader(
        "Importar catálogo exportado por esta versión", type=["json"]
    )
    if archivo is not None and st.button("Importar catálogo"):
        try:
            contenido = json.load(archivo)
            if not isinstance(contenido, dict) or contenido.get("version") != "tomate-1":
                raise ValueError(
                    "Formato no reconocido. El JSON anterior necesita adaptación."
                )
            importado = validar_catalogo(pd.DataFrame(contenido["productos"]))
            st.session_state.catalogo_base = importado
            st.session_state.catalogo_version += 1
            st.rerun()
        except Exception as exc:
            st.error(f"No se importó el catálogo: {exc}")

    editado = st.data_editor(
        st.session_state.catalogo_base,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key=f"catalogo_{st.session_state.catalogo_version}",
        column_config={
            **{c: st.column_config.CheckboxColumn(c) for c in BOOLEANOS},
            **{
                c: st.column_config.NumberColumn(
                    f"{c} · % elemental", min_value=0.0, max_value=100.0
                )
                for c in ELEMENTOS
            },
        },
    )

    catalogo_valido = True
    try:
        catalogo = validar_catalogo(editado)
    except Exception as exc:
        st.error(str(exc))
        catalogo_valido = False
        catalogo = pd.DataFrame(columns=COLUMNAS_PRODUCTO)

    if catalogo_valido:
        exportacion = {
            "version": "tomate-1",
            "base": "porcentaje en masa elemental",
            "productos": catalogo.to_dict(orient="records"),
        }
        st.download_button(
            "Descargar catálogo para conservarlo",
            json.dumps(exportacion, ensure_ascii=False, indent=2),
            file_name="catalogo_tomate.json",
            mime="application/json",
        )

    st.caption(
        "El catálogo editado permanece durante la sesión, no en una base "
        "de datos permanente. Descárgalo para no perder los cambios."
    )

# ---------- FORMULADOR ----------
resultado = None
seleccionados = pd.DataFrame()
dosis_g = None

with tabs[3]:
    st.subheader("Formular con productos verificados")
    st.info(
        "Esta versión formula N, P, K, Ca, Mg y S. "
        "Los micronutrientes se muestran en el balance, pero no se dosifican "
        "automáticamente. Los productos con micronutrientes quedan excluidos."
    )

    disponibles = catalogo.copy()
    if not disponibles.empty:
        nombre = disponibles["Producto"].str.casefold()
        bloqueo_nombre = nombre.str.contains(r"abonis|boro", regex=True)
        tiene_micro = disponibles[
            [e for e in ELEMENTOS if e not in MACROS]
        ].sum(axis=1) > 0
        aptos = (
            disponibles["Verificado"]
            & disponibles["Soluble"]
            & ~disponibles["Bloqueado"]
            & ~bloqueo_nombre
            & ~tiene_micro
            & (disponibles[MACROS].sum(axis=1) > 0)
        )
        disponibles = disponibles.loc[aptos]

    nombres = st.multiselect(
        "Productos que deseas utilizar",
        disponibles["Producto"].tolist(),
    )

    confirmado = st.checkbox(
        "Entiendo que es una simulación; revisaré concentraciones, "
        "solubilidad, compatibilidad y objetivos antes de aplicar."
    )

    if not nombres:
        st.warning(
            "Registra y selecciona fertilizantes con composición verificada "
            "para obtener cantidades comerciales."
        )
    elif not (area_ok and hidraulica_ok and confirmado):
        st.warning(
            "Confirma superficie, mediciones hidráulicas y revisión de la "
            "simulación para habilitar el cálculo."
        )
    else:
        seleccionados = disponibles[
            disponibles["Producto"].isin(nombres)
        ].copy().reset_index(drop=True)

        # A: gramos de elemento por gramo de producto.
        A = seleccionados[MACROS].to_numpy(dtype=float).T / 100.0
        b = deficit_g.reindex(MACROS).to_numpy(dtype=float)

        # Ajuste no negativo con ponderación relativa.
        # No es optimización de costes ni garantía de coincidencia exacta.
        escala = np.maximum(b, 0.1)
        ajuste = lsq_linear(
            A / escala[:, None],
            b / escala,
            bounds=(0.0, np.inf),
            tol=1e-10,
            max_iter=1000,
        )
        if not ajuste.success:
            st.error("El cálculo no convergió. Revisa productos y objetivos.")
        else:
            dosis_g = np.maximum(ajuste.x, 0.0)
            dosis_g[dosis_g < 1e-6] = 0.0

            resultado = pd.DataFrame({
                "Producto": seleccionados["Producto"],
                "g/día · cultivo": dosis_g,
                "g/día · sector": dosis_g / sectores,
                "kg/semana · cultivo": dosis_g * dias_semana / 1000.0,
                "kg/etapa · cultivo": dosis_g * dias_etapa / 1000.0,
            })
            st.dataframe(
                resultado.round(4),
                hide_index=True, use_container_width=True,
            )
            st.caption(
                "Semana = 5 aplicaciones; etapa = 25 aplicaciones idénticas. "
                "No multipliques de nuevo la dosis diaria por el número de sectores."
            )

            aportes = (
                seleccionados[ELEMENTOS].to_numpy(dtype=float).T / 100.0
            ) @ dosis_g
            total_g = aportes + agua_g.to_numpy(dtype=float)
            alcanzadas = total_g * 1000.0 / final_dia
            durante = (
                ppm_agua.to_numpy(dtype=float)
                + aportes * 1000.0 / final_inyeccion
            )
            diferencias = alcanzadas - ppm_obj.to_numpy(dtype=float)

            balance = pd.DataFrame({
                "Elemento": ELEMENTOS,
                "Objetivo · ppm promedio": ppm_obj.values,
                "Alcanzado · ppm promedio": alcanzadas,
                "Diferencia · ppm": diferencias,
                "Durante inyección · ppm": durante,
            })
            st.markdown("**Balance de la fórmula calculada**")
            st.dataframe(
                balance.round(4), hide_index=True, use_container_width=True
            )

            errores_macro = np.abs(diferencias[:len(MACROS)])
            tolerancias = np.maximum(
                0.1, 0.05 * ppm_obj.reindex(MACROS).to_numpy(dtype=float)
            )
            if (errores_macro > tolerancias).any():
                st.error(
                    "La combinación no alcanza todos los objetivos de macros "
                    "dentro del criterio numérico de revisión. "
                    "Hay faltantes o excesos: revisa el balance."
                )
            else:
                st.success(
                    "El ajuste de macros está dentro del criterio numérico "
                    "de revisión: 5% del objetivo o 0,1 ppm. "
                    "Este criterio NO es un límite de seguridad agronómica."
                )

            st.caption(
                "El ajuste puede sacrificar un objetivo para aproximar otros. "
                "No decide automáticamente qué producto comprar, no estima "
                "conductividad eléctrica y no valida la seguridad de las ppm."
            )

# ---------- TANQUE ----------
with tabs[4]:
    st.subheader("Solución madre y compatibilidad")
    st.write(
        f"Consumo por sector: **{madre_sector:.1f} L**. "
        f"Consumo diario para todos los sectores: **{madre_dia:.1f} L**."
    )
    st.write(
        f"Una preparación de **{tanque:.0f} L** equivale hidráulicamente a "
        f"**{tanque / madre_dia:.2f} jornadas** con esta configuración."
    )
    st.caption(
        "La equivalencia no valida almacenamiento, estabilidad de la mezcla "
        "ni su uso al cambiar de etapa."
    )

    if resultado is None:
        st.info("Completa el formulador para ver las cantidades de solución madre.")
    else:
        activos = seleccionados.loc[dosis_g > 0].copy()
        activos["g_dia"] = dosis_g[dosis_g > 0]

        if activos.empty:
            st.info("La simulación no asignó fertilizante.")
        else:
            calcio = (
                activos["Contiene_calcio"] | (activos["Ca"] > 0)
            )
            fosfato_sulfato = (
                activos["Contiene_fosfatos"]
                | activos["Contiene_sulfatos"]
                | (activos["P"] > 0)
                | (activos["S"] > 0)
            )

            conflicto = bool(calcio.any() and fosfato_sulfato.any())

            if conflicto:
                st.error(
                    "NO preparar todos estos productos juntos en el concentrado: "
                    "se detecta calcio junto con fuentes de fósforo o azufre."
                )
                st.warning(
                    "Separación preliminar: A = fuentes de calcio; "
                    "B = fuentes de fósforo/azufre. Los restantes productos "
                    "requieren revisión. Esta separación no certifica compatibilidad."
                )
                grupos = []
                for _, fila in activos.iterrows():
                    es_ca = bool(fila["Contiene_calcio"] or fila["Ca"] > 0)
                    es_ps = bool(
                        fila["Contiene_fosfatos"]
                        or fila["Contiene_sulfatos"]
                        or fila["P"] > 0 or fila["S"] > 0
                    )
                    if es_ca and es_ps:
                        grupo = "REVISAR ficha: contiene ambos grupos"
                    elif es_ca:
                        grupo = "A · revisar"
                    elif es_ps:
                        grupo = "B · revisar"
                    else:
                        grupo = "Asignación pendiente"
                    grupos.append(grupo)

                st.dataframe(pd.DataFrame({
                    "Producto": activos["Producto"].values,
                    "Separación preliminar": grupos,
                    "g/día · cultivo": activos["g_dia"].values,
                }), hide_index=True, use_container_width=True)

                st.info(
                    "Se bloquea el cálculo de una mezcla única de 200 L. "
                    "Con un solo tanque deben prepararse e inyectarse por separado, "
                    "con lavado entre preparaciones. Antes de calcular concentraciones "
                    "A/B hay que definir minutos de inyección y succión de cada una. "
                    "No asignes a cada grupo todo el tiempo de inyección disponible."
                )
            else:
                st.warning(
                    "No se detectó la incompatibilidad básica calcio–fosfato/sulfato. "
                    "Esto NO demuestra que la mezcla sea compatible o soluble."
                )
                concentracion = activos["g_dia"] / madre_dia
                tanque_df = pd.DataFrame({
                    "Producto": activos["Producto"].values,
                    "g/L de solución madre": concentracion.values,
                    "kg para volumen preparado":
                        (concentracion * tanque / 1000.0).values,
                })
                st.dataframe(
                    tanque_df.round(4), hide_index=True, use_container_width=True
                )
                st.caption(
                    "Las cantidades son para completar hasta el volumen final "
                    "indicado, no para agregarlas a ese volumen de agua ya lleno. "
                    "El modelo aproxima volúmenes aditivos."
                )
                if tanque < madre_dia:
                    st.warning(
                        "Una preparación no alcanza para todos los sectores del día. "
                        "Se necesitan reposiciones a la misma concentración."
                    )

    st.markdown("**Revisiones obligatorias antes de cualquier aplicación**")
    st.markdown(
        "- Confirmar análisis de agua, necesidad del cultivo y aporte del suelo.\n"
        "- Verificar solubilidad de cada producto y de la mezcla a la temperatura real.\n"
        "- Revisar pH y conductividad eléctrica; esta versión no los predice.\n"
        "- No inyectar Abonisa ni productos de composición incierta.\n"
        "- Reservar pre-riego y lavado según el tiempo de tránsito del sistema.\n"
        "- No considerar los micronutrientes de la referencia como dosis autorizadas.\n"
        "- No asumir que el cobalto de la tabla es necesario para el tomate."
    )

st.caption(
    "Datos de referencia editables · Catálogo sin persistencia en servidor · "
    "Resultados para revisión técnica, no ejecución automática."
)
