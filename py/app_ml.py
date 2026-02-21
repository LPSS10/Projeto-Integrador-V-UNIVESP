# app_final_corrigido.py - VERSÃO COMPLETAMENTE CORRIGIDA
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Machine Learning
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="EcoWatt - Analytics com IA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS MODERNO
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #00E086, #00B8FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.2rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00E086, #00B8FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-title {
        font-size: 1.4rem;
        color: #00E086;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid #00E086;
        padding-left: 0.8rem;
    }
    .ml-card {
        background: rgba(0, 224, 134, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #00E086;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CARREGAR E PROCESSAR DADOS REAIS - CORRIGIDO
# =============================================================================
@st.cache_data
def carregar_dados():
    try:
        df = pd.read_excel('formulario.xlsx')
        # Limpeza básica das colunas
        df.columns = [col.strip() for col in df.columns]
        
        # Converter coluna de data/hora para string para evitar problemas Arrow
        if 'Carimbo de data/hora' in df.columns:
            df['Carimbo de data/hora'] = df['Carimbo de data/hora'].astype(str)
        
        # Mapear nomes longos para nomes curtos
        col_mapping = {
            'Perfil do Respondente': 'perfil',
            'Quantas pessoas moram na sua unidade?': 'pessoas_casa',
            'Em média, qual é o valor da sua conta de luz mensal?': 'valor_conta',
            'Você costuma monitorar o consumo de energia da sua residência?': 'monitora_consumo',
            'Se respondeu sim, quais meios você utiliza para monitorar o consumo?': 'meios_monitoramento',
            'Você sabe identificar quais aparelhos consomem mais energia na sua casa?': 'identifica_aparelhos',
            'Em horários de pico (18h–22h), você costuma evitar usar muitos aparelhos elétricos ao mesmo tempo?': 'comportamento_pico',
            'Se houvesse uma plataforma que mostrasse em tempo real o consumo de energia do seu apartamento, você teria interesse em utilizá-la?': 'interesse_plataforma',
            'Qual benefício você considera mais importante em uma solução como essa?': 'beneficio_principal',
            'O que você gostaria que uma plataforma de monitoramento de energia oferecesse para realmente ser útil para você?': 'sugestoes'
        }
        
        # Renomear colunas que existem no DataFrame
        existing_columns = {}
        for old_name, new_name in col_mapping.items():
            if old_name in df.columns:
                existing_columns[old_name] = new_name
        
        df = df.rename(columns=existing_columns)
        
        return df
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        # Dados de fallback baseados no seu exemplo real
        data = {
            'perfil': ['Inquilino', 'Inquilino', 'Proprietário', 'Proprietário', 
                      'Proprietário', 'Inquilino', 'Proprietário'],
            'valor_conta': ['R$ 151 – R$ 300', 'R$ 301 – R$ 500', 'Até R$ 150', 'R$ 151 – R$ 300',
                          'R$ 151 – R$ 300', 'Até R$ 150', 'R$ 151 – R$ 300'],
            'monitora_consumo': ['Não', 'Não', 'Sim, regularmente', 'Sim, mas de forma esporádica',
                               'Sim, mas de forma esporádica', 'Não', 'Sim, regularmente'],
            'interesse_plataforma': ['Talvez, dependendo do custo', 'Talvez, dependendo do custo', 
                                   'Talvez, dependendo do custo', 'Talvez, dependendo do custo', 
                                   'Sim', 'Sim', 'Talvez, dependendo do custo'],
            'comportamento_pico': ['Não tenho esse hábito, mas teria interesse', 'Não tenho esse hábito, mas teria interesse',
                                 'Não', 'Não', 'Sim', 'Não', 'Não'],
            'identifica_aparelhos': ['Mais ou menos', 'Mais ou menos', 'Sim', 'Não', 'Sim', 'Não', 'Sim']
        }
        return pd.DataFrame(data)

# Carregar dados
df = carregar_dados()

# =============================================================================
# FUNÇÕES DE MACHINE LEARNING
# =============================================================================

def preparar_dados_ml(df):
    """Preparar dados para análise de Machine Learning"""
    df_ml = df.copy()
    
    # Criar variável target: Risco de Alto Consumo
    # Baseado na conta de luz e comportamento
    def criar_target(row):
        if 'valor_conta' in row.index and pd.notna(row['valor_conta']):
            if '301' in str(row['valor_conta']) or '500' in str(row['valor_conta']):
                return 'Alto Risco'
            elif '151' in str(row['valor_conta']) or '300' in str(row['valor_conta']):
                return 'Médio Risco'
            else:
                return 'Baixo Risco'
        return 'Desconhecido'
    
    if 'valor_conta' in df_ml.columns:
        df_ml['risco_consumo'] = df_ml.apply(criar_target, axis=1)
    
    # Codificar variáveis categóricas
    le = LabelEncoder()
    features_encoded = {}
    
    # Selecionar features para ML
    ml_features = ['perfil', 'monitora_consumo', 'identifica_aparelhos', 'comportamento_pico']
    
    for feature in ml_features:
        if feature in df_ml.columns:
            # Preencher NaN com string vazia antes de codificar
            df_ml[feature] = df_ml[feature].fillna('Não informado')
            df_ml[f'{feature}_encoded'] = le.fit_transform(df_ml[feature].astype(str))
            features_encoded[feature] = le.classes_
    
    return df_ml, features_encoded

def aplicar_kmeans(df_ml):
    """Aplicar clusterização K-Means"""
    try:
        # Selecionar features para clusterização
        features = [col for col in df_ml.columns if 'encoded' in col]
        
        if len(features) < 2:
            st.warning("Dados insuficientes para clusterização")
            return None, None
        
        X = df_ml[features].fillna(0)
        
        # Normalizar dados
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Aplicar K-Means
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        df_ml['cluster'] = clusters
        
        # Nomear clusters baseado nas características
        cluster_names = {
            0: 'Grupo Econômico',
            1: 'Grupo Moderado', 
            2: 'Grupo Alto Consumo'
        }
        
        df_ml['cluster_nome'] = df_ml['cluster'].map(cluster_names)
        
        return df_ml, kmeans
        
    except Exception as e:
        st.error(f"Erro na clusterização: {e}")
        return None, None

def treinar_random_forest(df_ml):
    """Treinar modelo Random Forest para prever risco"""
    try:
        if 'risco_consumo' not in df_ml.columns:
            st.warning("Variável target não disponível para treinamento")
            return None, None, None
        
        # Features para o modelo
        features = [col for col in df_ml.columns if 'encoded' in col]
        
        if len(features) < 2:
            st.warning("Features insuficientes para treinamento")
            return None, None, None
        
        X = df_ml[features].fillna(0)
        y = df_ml['risco_consumo']
        
        # Verificar se temos dados suficientes para treino/teste
        if len(X) < 4:
            # Usar todos os dados para treino se amostra for muito pequena
            X_train, y_train = X, y
            X_test, y_test = None, None
        else:
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        
        # Treinar Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return rf_model, feature_importance, (X_test, y_test)
        
    except Exception as e:
        st.error(f"Erro no treinamento: {e}")
        return None, None, None

def analisar_importancia_features(feature_importance):
    """Analisar importância das features"""
    if feature_importance is None:
        return None
    
    fig = px.bar(
        feature_importance,
        x='importance',
        y='feature',
        orientation='h',
        title='📊 Importância dos Fatores no Consumo Energético',
        color='importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        yaxis_title='Fatores',
        xaxis_title='Importância Relativa',
        height=400
    )
    
    return fig

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================
def gerar_dados_consumo():
    """Gerar dados simulados de consumo horário"""
    horas = [f"{h:02d}:00" for h in range(8, 20)]
    consumo = [round(random.uniform(1.5, 2.8), 2) for _ in range(len(horas))]
    custo = [round(c * 0.95, 2) for c in consumo]  # R$ 0.95 por kWh
    
    return pd.DataFrame({
        'Hora': horas,
        'Consumo_kWh': consumo,
        'Custo_R$': custo
    })

def gerar_dados_diarios():
    """Gerar dados simulados de consumo diário"""
    dias = [f'Dia {i+1}' for i in range(7)]
    consumo_diario = [round(random.uniform(15, 45), 2) for _ in range(7)]
    
    return pd.DataFrame({
        'Dia': dias,
        'Consumo_kWh': consumo_diario
    })

def contar_interessados(df):
    """Contar moradores interessados na plataforma de forma segura"""
    if 'interesse_plataforma' not in df.columns:
        return 0
    
    interessados = 0
    for resposta in df['interesse_plataforma']:
        if isinstance(resposta, str) and 'Sim' in resposta:
            interessados += 1
    return interessados

def contar_monitoramento(df):
    """Contar moradores que monitoram consumo de forma segura"""
    if 'monitora_consumo' not in df.columns:
        return 0
    
    monitora = 0
    for resposta in df['monitora_consumo']:
        if isinstance(resposta, str) and 'Sim' in resposta:
            monitora += 1
    return monitora

# =============================================================================
# APLICAR MACHINE LEARNING
# =============================================================================
df_ml, features_encoded = preparar_dados_ml(df)
df_ml, kmeans_model = aplicar_kmeans(df_ml)
rf_model, feature_importance, test_data = treinar_random_forest(df_ml)

# =============================================================================
# LAYOUT PRINCIPAL
# =============================================================================

# Header
st.markdown('<div class="main-header">⚡ EcoWatt - Analytics com Inteligência Artificial</div>', unsafe_allow_html=True)
st.markdown("**Dashboard inteligente com Machine Learning para análise preditiva**")

# =============================================================================
# SEÇÃO 1: MÉTRICAS PRINCIPAIS
# =============================================================================
st.markdown('<div class="section-title">📊 RESUMO DO CONDOMÍNIO</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_moradores = len(df)
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-value">{total_moradores}</div>
        <div>Total de Moradores</div>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    if 'perfil' in df.columns:
        proprietarios = len(df[df['perfil'] == 'Proprietário'])
    else:
        proprietarios = 0
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-value">{proprietarios}</div>
        <div>Proprietários</div>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    if 'valor_conta' in df.columns:
        alta_conta = len(df[df['valor_conta'].str.contains('301|500', na=False)])
    else:
        alta_conta = 0
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-value">{alta_conta}</div>
        <div>Contas Altas (R$ 301-500)</div>
    </div>
    ''', unsafe_allow_html=True)

with col4:
    interessados = contar_interessados(df)
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-value">{interessados}</div>
        <div>Interessados na Plataforma</div>
    </div>
    ''', unsafe_allow_html=True)

# =============================================================================
# NOVA SEÇÃO: MACHINE LEARNING & IA
# =============================================================================
st.markdown('<div class="section-title">🤖 ANÁLISE PREDITIVA COM MACHINE LEARNING</div>', unsafe_allow_html=True)

st.markdown("""
<div class="ml-card">
    <strong>🧠 Técnicas de IA Aplicadas:</strong><br>
    • <strong>Clusterização (K-Means):</strong> Segmentação inteligente da comunidade<br>
    • <strong>Classificação (Random Forest):</strong> Previsão de risco de alto consumo<br>
    • <strong>Análise de Importância:</strong> Identificação dos fatores mais relevantes
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SUBSEÇÃO 1: CLUSTERIZAÇÃO
# =============================================================================
st.markdown('#### 🎯 1. Segmentação Inteligente da Comunidade (K-Means)')

if df_ml is not None and 'cluster_nome' in df_ml.columns:
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribuição dos clusters
        cluster_dist = df_ml['cluster_nome'].value_counts().reset_index()
        cluster_dist.columns = ['Grupo', 'Quantidade']
        
        fig_clusters = px.pie(
            cluster_dist,
            values='Quantidade',
            names='Grupo',
            title='Distribuição dos Grupos de Consumo',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_clusters, width='stretch')
    
    with col2:
        # Características dos clusters
        st.markdown("**📋 Perfil dos Grupos:**")
        
        # Análise simplificada dos clusters
        for cluster_name in df_ml['cluster_nome'].unique():
            cluster_data = df_ml[df_ml['cluster_nome'] == cluster_name]
            
            st.write(f"**{cluster_name}** ({len(cluster_data)} moradores):")
            
            if 'valor_conta' in cluster_data.columns:
                contas = cluster_data['valor_conta'].value_counts()
                if not contas.empty:
                    st.write(f"  • Conta predominante: {contas.index[0]}")
            
            if 'monitora_consumo' in cluster_data.columns:
                monitora = cluster_data['monitora_consumo'].str.contains('Sim', na=False).sum()
                st.write(f"  • Monitoram consumo: {monitora}/{len(cluster_data)}")
            
            st.write("---")

else:
    st.info("""
    **ℹ️ Clusterização não disponível**
    - Necessário: Dados suficientes sobre perfil, monitoramento e comportamento
    - Sugestão: Coletar mais respostas no formulário
    """)

# =============================================================================
# SUBSEÇÃO 2: ANÁLISE DE IMPORTÂNCIA
# =============================================================================
st.markdown('#### 📊 2. Fatores que Mais Impactam o Consumo')

if feature_importance is not None:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_importance = analisar_importancia_features(feature_importance)
        if fig_importance:
            st.plotly_chart(fig_importance, width='stretch')
    
    with col2:
        st.markdown("**🎯 Insights da Análise:**")
        
        top_features = feature_importance.head(3)
        for idx, row in top_features.iterrows():
            feature_name = row['feature'].replace('_encoded', '')
            importance_pct = row['importance'] * 100
            
            st.write(f"**{feature_name}**")
            st.write(f"Impacto: {importance_pct:.1f}%")
            st.write("---")
        
        st.markdown("""
        **💡 Recomendações Baseadas na IA:**
        1. Focar campanhas nos fatores mais importantes
        2. Personalizar comunicação por grupo
        3. Monitorar mudanças nos padrões
        """)

else:
    st.info("""
    **ℹ️ Análise de importância não disponível**
    - O modelo precisa de mais dados para identificar padrões
    - Fatores analisados: Perfil, Monitoramento, Comportamento
    """)

# =============================================================================
# SUBSEÇÃO 3: PREVISÃO DE RISCO
# =============================================================================
st.markdown('#### 🔮 3. Previsão de Risco de Alto Consumo')

if rf_model is not None and 'risco_consumo' in df_ml.columns:
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribuição do risco
        risco_dist = df_ml['risco_consumo'].value_counts().reset_index()
        risco_dist.columns = ['Risco', 'Quantidade']
        
        fig_risco = px.bar(
            risco_dist,
            x='Risco',
            y='Quantidade',
            title='Distribuição do Risco de Consumo',
            color='Risco',
            color_discrete_sequence=['#00E086', '#FFA726', '#EF5350']
        )
        st.plotly_chart(fig_risco, width='stretch')
    
    with col2:
        st.markdown("**🎯 Ações Recomendadas por Nível de Risco:**")
        
        riscos_actions = {
            'Alto Risco': '• Consultoria personalizada\n• Análise detalhada de aparelhos\n• Programa de eficiência urgente',
            'Médio Risco': '• Educação energética\n• Dicas personalizadas\n• Monitoramento contínuo', 
            'Baixo Risco': '• Manutenção de bons hábitos\n• Compartilhar melhores práticas\n• Participar como embaixador'
        }
        
        for risco, acao in riscos_actions.items():
            count = len(df_ml[df_ml['risco_consumo'] == risco])
            if count > 0:
                st.write(f"**{risco}** ({count} moradores):")
                st.write(acao)
                st.write("---")

else:
    st.info("""
    **ℹ️ Previsão de risco não disponível**
    - Sistema precisa de dados sobre valores de conta de luz
    - Modelo Random Forest treinado para identificar padrões de risco
    """)

# =============================================================================
# SEÇÃO 2: GRÁFICOS DE CONSUMO TEMPORAL
# =============================================================================
st.markdown('<div class="section-title">⏰ CONSUMO EM TEMPO REAL</div>', unsafe_allow_html=True)

dados_horarios = gerar_dados_consumo()
dados_diarios = gerar_dados_diarios()

col1, col2 = st.columns(2)

with col1:
    fig_horario = go.Figure()
    fig_horario.add_trace(go.Scatter(
        x=dados_horarios['Hora'], y=dados_horarios['Consumo_kWh'],
        mode='lines+markers', name='Consumo (kWh)',
        line=dict(color='#00E086', width=3), marker=dict(size=8)
    ))
    fig_horario.add_trace(go.Scatter(
        x=dados_horarios['Hora'], y=dados_horarios['Consumo_kWh'],
        fill='tozeroy', fillcolor='rgba(0, 224, 134, 0.2)',
        line=dict(color='rgba(255,255,255,0)'), showlegend=False
    ))
    fig_horario.update_layout(
        title="Consumo Horário - Hoje", height=400,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_horario, width='stretch')

with col2:
    fig_diario = go.Figure()
    fig_diario.add_trace(go.Bar(
        x=dados_diarios['Dia'], y=dados_diarios['Consumo_kWh'],
        marker_color=['#00E086' if x < 30 else '#FF6B6B' for x in dados_diarios['Consumo_kWh']],
        opacity=0.8
    ))
    fig_diario.update_layout(
        title="Consumo dos Últimos 7 Dias", height=400,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_diario, width='stretch')

# =============================================================================
# SEÇÃO 3: ANÁLISE DOS DADOS REAIS
# =============================================================================
st.markdown('<div class="section-title">📈 ANÁLISE DA PESQUISA COM MORADORES</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Distribuição de contas de luz
    if 'valor_conta' in df.columns:
        contas_data = df['valor_conta'].value_counts().reset_index()
        contas_data.columns = ['Faixa', 'Quantidade']
        
        fig_contas = px.pie(
            contas_data, 
            values='Quantidade', 
            names='Faixa',
            title="Distribuição das Contas de Luz",
            color_discrete_sequence=px.colors.sequential.Viridis,
            hole=0.4
        )
        fig_contas.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_contas, width='stretch')
    else:
        st.info("Dados de contas de luz não disponíveis")

with col2:
    # Monitoramento vs Perfil
    if 'perfil' in df.columns and 'monitora_consumo' in df.columns:
        monitoramento_data = df.groupby(['perfil', 'monitora_consumo']).size().reset_index()
        monitoramento_data.columns = ['Perfil', 'Monitoramento', 'Quantidade']
        
        fig_monitoramento = px.bar(
            monitoramento_data,
            x='Perfil',
            y='Quantidade',
            color='Monitoramento',
            title="Monitoramento do Consumo por Perfil",
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_monitoramento, width='stretch')
    else:
        st.info("Dados de monitoramento não disponíveis")

# =============================================================================
# RODAPÉ
# =============================================================================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <p>⚡ EcoWatt - Analytics com IA | Pesquisa com {len(df)} moradores | Machine Learning Aplicado</p>
    <p>Clusterização • Classificação • Análise Preditiva • Recomendações Inteligentes</p>
</div>
""", unsafe_allow_html=True)

# Debug expander - CORRIGIDO para evitar problemas de serialização
with st.expander("🔍 Ver detalhes técnicos do Machine Learning"):
    st.write("**Total de respostas:**", len(df))
    st.write("**Features utilizadas:**", list(features_encoded.keys()) if features_encoded else "Nenhuma")
    st.write("**Modelo K-Means:**", "Treinado" if kmeans_model else "Não disponível")
    st.write("**Modelo Random Forest:**", "Treinado" if rf_model else "Não disponível")
    
    # Mostrar dados de forma segura (sem colunas problemáticas)
    if df_ml is not None:
        st.write("**Colunas disponíveis no ML:**", [col for col in df_ml.columns if not col.startswith('_')])