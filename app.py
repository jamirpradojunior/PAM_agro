import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from pathlib import Path
import unicodedata
import numpy as np
from branca.element import Element # Importante para injetar o CSS no mapa

# --- CONFIGURAÇÃO ---
# 🌱 Ícone da página
st.set_page_config(page_title="Agro Viewer Pro", page_icon="🌱", layout="wide")

# --- ESTILO CSS (Global) ---
st.markdown("""
    <style>
    html, body, [class*="css"]  { font-size: 110%; }
    h1 { font-size: 2.5rem !important; }
    h2 { font-size: 2rem !important; }
    h3 { font-size: 1.8rem !important; }
    .stSidebar { font-size: 1.1rem !important; }
    [data-testid="stMetricValue"] { font-size: 2.2rem !important; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 1. DADOS ESTRATÉGICOS (PROTEC)
# ==============================================================================

PROTEC_UNIDADES = [
    # MG
    "ARAXÁ", "ARAGUARI", "CAPINÓPOLIS", "COROMANDEL", "FRUTAL", "IBIÁ",
    "PARACATU", "PATOS DE MINAS", "PATROCÍNIO", "PASSOS", "SANTA JULIANA",
    "SÃO GOTARDO", "UBERABA", "UBERLÂNDIA", "UNAÍ",
    # SP
    "AVARÉ", "MOGI MIRIM"
]

CATEGORIAS = {
    "Grãos & Cereais": ["Arroz", "Aveia", "Centeio", "Cevada", "Ervilha", "Fava", "Feijão", "Girassol", "Milho", "Soja", "Sorgo", "Trigo", "Triticale"],
    "Industriais & Fibras": ["Algodão arbóreo", "Algodão herbáceo", "Borracha", "Cana-de-açúcar", "Fumo", "Juta", "Malva", "Mamona", "Rami", "Sisal ou agave"],
    "Frutas": ["Abacate", "Abacaxi", "Açaí", "Azeitona", "Banana", "Caju", "Caqui", "Castanha de caju", "Coco-da-baía", "Dendê", "Figo", "Goiaba", "Guaraná", "Laranja", "Limão", "Maçã", "Mamão", "Manga", "Maracujá", "Marmelo", "Melancia", "Melão", "Noz", "Pera", "Pêssego", "Tangerina", "Urucum", "Uva"],
    "Café & Estimulantes": ["Café", "Chá-da-índia", "Erva-mate", "Cacau"],
    "Hortaliças & Tubérculos": ["Alho", "Batata-doce", "Batata-inglesa", "Cebola", "Mandioca", "Palmito", "Pimenta-do-reino", "Tomate"],
    "Outros": ["Alfafa fenada", "Tungue", "Geral"]
}

# ==============================================================================
# 2. FUNÇÕES AUXILIARES
# ==============================================================================

def limpar_nome_cultura(nome):
    if not isinstance(nome, str): return str(nome)
    n = nome.lower()
    if "cana" in n: return "Cana-de-açúcar"
    if "café" in n or "cafe" in n: return "Café"
    if "algodão" in n: return "Algodão"
    if "borracha" in n: return "Borracha"
    return nome.split('(')[0].strip()

def normalizar_nome(nome):
    """Remove acentos e coloca em maiúsculo para comparar nomes de cidades"""
    if not isinstance(nome, str): return ""
    nfkd = unicodedata.normalize('NFKD', nome)
    return "".join([c for c in nfkd if not unicodedata.mirrored(c) and not unicodedata.combining(c)]).upper().strip()

def compute_regional_metric(gdf, col_valor, raio_km):
    if gdf.empty or raio_km == 0:
        return gdf, col_valor
    
    lats = np.deg2rad(gdf.geometry.centroid.y.values)
    lons = np.deg2rad(gdf.geometry.centroid.x.values)
    vals = gdf[col_valor].fillna(0).values
    
    lat1, lat2 = lats[:, None], lats[None, :]
    lon1, lon2 = lons[:, None], lons[None, :]
    
    dphi = lat2 - lat1
    dlambda = lon2 - lon1
    a = np.sin(dphi/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    dist_km = 6371 * c 
    
    mask = dist_km <= raio_km
    regional_vals = (mask * vals[None, :]).sum(axis=1)
    
    gdf_out = gdf.copy()
    col_nova = f"regional_{col_valor}"
    gdf_out[col_nova] = regional_vals
    return gdf_out, col_nova

# --- CARGA DE DADOS ---
@st.cache_data
def load_data():
    data_dir = Path("data/processed")
    arquivos_pam = list(data_dir.glob("pam_*.parquet"))
    
    if not arquivos_pam: return None, None
    
    pam_path = max(arquivos_pam, key=lambda p: p.stat().st_mtime)
    df = pd.read_parquet(pam_path)
    
    df['cultura_original'] = df['cultura']
    df['cultura'] = df['cultura_original'].apply(limpar_nome_cultura)
    if 'ano' not in df.columns: df['ano'] = '2023'
    else: df['ano'] = df['ano'].astype(str)
    
    geo_path = data_dir / "municipios_leve.parquet"
    if geo_path.exists():
        gdf_full = gpd.read_parquet(geo_path)
        gdf_full['CD_MUN'] = gdf_full['CD_MUN'].astype(str)
        
        # 1. Extrai coordenadas da PROTEC
        df_protec_coords = pd.DataFrame()
        if 'NM_MUN' in gdf_full.columns:
            gdf_full['nm_norm'] = gdf_full['NM_MUN'].apply(normalizar_nome)
            protec_norm = [normalizar_nome(p) for p in PROTEC_UNIDADES]
            
            mask_protec = gdf_full['nm_norm'].isin(protec_norm)
            protec_geo = gdf_full[mask_protec].copy()
            
            if not protec_geo.empty:
                df_protec_coords = protec_geo[['NM_MUN', 'geometry']].copy()
                df_protec_coords['lat'] = df_protec_coords.geometry.centroid.y
                df_protec_coords['lon'] = df_protec_coords.geometry.centroid.x
        
        # 2. Filtra GeoDataFrame para o Mapa
        df['cd_mun'] = df['cd_mun'].astype(str)
        estados_nos_dados = df['cd_mun'].str[:2].unique()
        gdf_mapa = gdf_full[gdf_full['CD_MUN'].str[:2].isin(estados_nos_dados)].copy()
        
        return df, gdf_mapa, df_protec_coords
    
    return df, None, None

# --- INÍCIO ---
with st.spinner("Carregando inteligência agrícola..."):
    df_raw, gdf_geo, df_protec = load_data()

if df_raw is None:
    st.error("❌ Sem dados.")
    st.stop()

# ==============================================================================
# 3. FILTROS
# ==============================================================================
st.sidebar.header("🔎 Filtros")

# A. TEMPO
anos_disponiveis = sorted(df_raw['ano'].unique())
modo_tempo = "Ano Único"
if len(anos_disponiveis) > 1:
    modo_tempo = st.sidebar.radio("Tempo:", ["Ano Único", "Evolução (Diferença)"])
    if modo_tempo == "Ano Único":
        ano_selecionado = st.sidebar.selectbox("Ano:", anos_disponiveis, index=len(anos_disponiveis)-1)
    else:
        c1, c2 = st.sidebar.columns(2)
        ano_base = c1.selectbox("De:", anos_disponiveis, index=0)
        ano_selecionado = c2.selectbox("Até:", anos_disponiveis, index=len(anos_disponiveis)-1)
else:
    ano_selecionado = anos_disponiveis[0]
    st.sidebar.info(f"Dados: {ano_selecionado}")

st.sidebar.markdown("---")

# B. CULTURA
cat_sel = st.sidebar.selectbox("Categoria:", ["Todas"] + list(CATEGORIAS.keys()))
if cat_sel != "Todas":
    opcoes_cultura = sorted([c for c in df_raw['cultura'].unique() if c in CATEGORIAS[cat_sel]])
else:
    opcoes_cultura = sorted(df_raw['cultura'].unique())
cultura_sel = st.sidebar.multiselect("Culturas:", options=opcoes_cultura, placeholder="Todas da categoria")

# C. VARIÁVEL
nomes_amigaveis = {
    'Area_Plantada_ha': 'Área Plantada (ha)', 
    'Area_Colhida_ha': 'Área Colhida (ha)',
    'Quantidade_Produzida_Ton': 'Produção (Ton)', 
    'Valor_Producao_Mil_Reais': 'Valor Produção (R$)', 
    'Rendimento_Medio_kg_ha': 'Produtividade (kg/ha)'
}
cols_num = [c for c in df_raw.columns if c in nomes_amigaveis]
metrica = st.sidebar.selectbox("Variável:", cols_num, format_func=lambda x: nomes_amigaveis.get(x, x))

# D. REGIONALIZAÇÃO
st.sidebar.markdown("---")
st.sidebar.markdown("**🗺️ Contexto Regional**")
raio_km = st.sidebar.slider("Raio de Influência (km):", 0, 200, 0, step=25, help="Soma a produção de vizinhos.")

# E. BUSCA
lista_mun = ["Todos"] + sorted(df_raw['municipio'].unique())
mun_sel = st.sidebar.selectbox("Focar em Município:", lista_mun)

# ==============================================================================
# 4. PROCESSAMENTO
# ==============================================================================

# Filtro e Agregação
df_work = df_raw.copy()
if cultura_sel: df_work = df_work[df_work['cultura'].isin(cultura_sel)]
elif cat_sel != "Todas": df_work = df_work[df_work['cultura'].isin(CATEGORIAS[cat_sel])]

cols_agg = {c: 'sum' for c in cols_num}
if 'Rendimento_Medio_kg_ha' in cols_agg: cols_agg['Rendimento_Medio_kg_ha'] = 'mean'
df_agg = df_work.groupby(['cd_mun', 'municipio', 'uf_id', 'ano']).agg(cols_agg).reset_index()

if modo_tempo == "Evolução (Diferença)" and len(anos_disponiveis) > 1:
    df_inicio = df_agg[df_agg['ano'] == ano_base].set_index('cd_mun')
    df_fim = df_agg[df_agg['ano'] == ano_selecionado].set_index('cd_mun')
    
    todos_muns = df_raw['cd_mun'].unique()
    df_inicio = df_inicio.reindex(todos_muns).fillna(0)
    df_fim = df_fim.reindex(todos_muns).fillna(0)
    ref_nomes = df_raw[['cd_mun', 'municipio', 'uf_id']].drop_duplicates('cd_mun').set_index('cd_mun')
    
    df_delta = pd.DataFrame(index=todos_muns)
    for col in cols_num: df_delta[col] = df_fim[col] - df_inicio[col]
    df_final = df_delta.join(ref_nomes).reset_index().rename(columns={'index': 'cd_mun'})
    titulo_kpi = f"Variação ({ano_base}➝{ano_selecionado})"
    cor_mapa = 'RdYlGn'
else:
    df_final = df_agg[df_agg['ano'] == ano_selecionado].copy()
    titulo_kpi = f"Total em {ano_selecionado}"
    cor_mapa = 'YlGn'

# ==============================================================================
# 5. DASHBOARD
# ==============================================================================

st.title(f"🌱 Produção Agrícola Municipal (PAM): {titulo_kpi}")
if cat_sel != "Todas": st.caption(f"Categoria: {cat_sel}")

c1, c2, c3 = st.columns(3)
val_area = df_final['Area_Plantada_ha'].sum()
val_prod = df_final['Quantidade_Produzida_Ton'].sum()
val_val = df_final['Valor_Producao_Mil_Reais'].sum()

def fmt_delta(val, is_monetary=False):
    simb = "R$" if is_monetary else ""
    color = "green" if val >= 0 else "red"
    return f":{color}[{simb}{abs(val):,.0f}]"

if modo_tempo == "Evolução (Diferença)":
    c1.markdown(f"**Δ Área Plantada**<br>{fmt_delta(val_area)} ha", unsafe_allow_html=True)
    c2.markdown(f"**Δ Produção**<br>{fmt_delta(val_prod)} t", unsafe_allow_html=True)
    c3.markdown(f"**Δ Valor**<br>{fmt_delta(val_val/1000, True)} Mi", unsafe_allow_html=True)
else:
    c1.metric("Área Plantada", f"{val_area:,.0f} ha")
    c2.metric("Produção", f"{val_prod:,.0f} t")
    c3.metric("Valor", f"R$ {val_val/1000:,.1f} Mi")

st.markdown("---")

col_map, col_tab = st.columns([2, 1])

with col_map:
    lbl_legenda = f"{nomes_amigaveis.get(metrica, metrica)} (Regional {raio_km}km)" if raio_km > 0 else nomes_amigaveis.get(metrica, metrica)
    st.subheader(f"🗺️ Mapa: {lbl_legenda}")
    
    if gdf_geo is not None:
        gdf_map = gdf_geo.merge(df_final, left_on='CD_MUN', right_on='cd_mun', how='left')
        gdf_map[cols_num] = gdf_map[cols_num].fillna(0)
        if 'municipio' in gdf_map.columns: gdf_map['municipio'] = gdf_map['municipio'].fillna(gdf_map['NM_MUN'])
        
        # Cálculo Regional
        coluna_plot = metrica
        if raio_km > 0:
            with st.spinner(f"Calculando regional ({raio_km}km)..."):
                gdf_map, col_regional = compute_regional_metric(gdf_map, metrica, raio_km)
                coluna_plot = col_regional

        # Zoom
        lat, lon, zoom = -22, -49, 6
        if mun_sel != "Todos":
            foco = gdf_map[gdf_map['municipio'] == mun_sel]
            if not foco.empty:
                lat, lon = foco.geometry.centroid.y.iloc[0], foco.geometry.centroid.x.iloc[0]
                zoom = 9

        m = folium.Map([lat, lon], zoom_start=zoom, tiles="cartodbpositron")
        
        # --- TRUQUE: Injeção de CSS DENTRO do Mapa para diminuir o pino ---
        css_mapa = """
        <style>
        .leaflet-marker-icon {
            transform: scale(0.65) !important; 
            transform-origin: bottom center !important;
        }
        </style>
        """
        m.get_root().html.add_child(Element(css_mapa))

        # --- MARCADORES DA PROTEC ---
        if df_protec is not None and not df_protec.empty:
            for _, row in df_protec.iterrows():
                if not pd.isna(row['lat']):
                    folium.Marker(
                        location=[row['lat'], row['lon']],
                        popup=f"🏠 Protec: {row['NM_MUN']}",
                        icon=folium.Icon(color='blue', icon='home', prefix='fa'),
                        tooltip=f"Protec: {row['NM_MUN']}"
                    ).add_to(m)

        # Raio Visual
        if raio_km > 0 and mun_sel != "Todos":
             folium.Circle(location=[lat, lon], radius=raio_km*1000, color="#3186cc", fill=False).add_to(m)

        folium.Choropleth(
            geo_data=gdf_map, data=gdf_map, columns=['CD_MUN', coluna_plot],
            key_on='feature.properties.CD_MUN', fill_color=cor_mapa, fill_opacity=0.7, line_opacity=0.1,
            legend_name=lbl_legenda, highlight=True,
            nan_fill_color="white", nan_fill_opacity=0.4,
            bins=5
        ).add_to(m)
        
        folium.GeoJson(
            gdf_map, style_function=lambda x: {'fillColor': '#00000000', 'color': 'transparent'},
            tooltip=folium.GeoJsonTooltip(fields=['municipio', metrica], aliases=['Município:', 'Valor:'], localize=True)
        ).add_to(m)
        
        if mun_sel != "Todos":
             destaque = gdf_map[gdf_map['municipio'] == mun_sel]
             if not destaque.empty:
                folium.GeoJson(destaque, style_function=lambda x: {'fillColor': 'transparent', 'color': 'red', 'weight': 3}).add_to(m)

        st_folium(m, width="100%", height=600)
    else: st.warning("Mapa vazio.")

with col_tab:
    st.subheader("📊 Ranking")
    if gdf_geo is not None:
        df_rank = gdf_map[gdf_map[coluna_plot] != 0].copy()
        df_rank = df_rank.sort_values(coluna_plot, ascending=False).head(30)
        
        cols_view = ['municipio', metrica]
        if raio_km > 0: cols_view.append(coluna_plot)
        st.dataframe(df_rank[cols_view], use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("📈 Histórico (Local)")
    if mun_sel != "Todos":
        historia = df_agg[df_agg['municipio'] == mun_sel].copy().sort_values('ano')
        st.line_chart(historia, x='ano', y=metrica, color="#228B22")
    else:
        historia = df_agg.groupby('ano')[cols_num].sum().reset_index().sort_values('ano')
        st.line_chart(historia, x='ano', y=metrica, color="#228B22")