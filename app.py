import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import kruskal
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

# Configurações gerais com tema escuro
st.set_page_config(
    page_title="Análise de Vermicompostagem de Bagaço de Uva", 
    layout="wide",
    page_icon="🍇"
)

# CSS para tema escuro premium
st.markdown("""
<style>
    /* Configurações gerais */
    body {
        color: #f0f2f6;
        background-color: #0e1117;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Containers e cards */
    .stApp {
        background: linear-gradient(135deg, #0c0f1d 0%, #131625 100%);
    }
    
    .card {
        background: rgba(20, 23, 40, 0.7) !important;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(100, 110, 200, 0.2);
    }
    
    .header-card {
        background: linear-gradient(135deg, #2a2f45 0%, #1a1d2b 100%);
        border-left: 4px solid #6f42c1;
        padding: 20px 30px;
    }
    
    .info-card {
        background: rgba(26, 29, 50, 0.8) !important;
        border-left: 4px solid #00c1e0;
        padding: 20px;
        border-radius: 0 12px 12px 0;
        margin-top: 15px;
    }
    
    .result-card {
        background: rgba(26, 29, 43, 0.9);
        border-left: 4px solid #6f42c1;
        padding: 20px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 20px;
    }
    
    .signif-card {
        border-left: 4px solid #00c853 !important;
    }
    
    .not-signif-card {
        border-left: 4px solid #ff5252 !important;
    }
    
    .reference-card {
        background: rgba(20, 23, 40, 0.9) !important;
        border-left: 4px solid #00c1e0;
        padding: 20px;
        border-radius: 0 12px 12px 0;
        margin-top: 40px;
    }
    
    /* Títulos */
    h1, h2, h3, h4, h5, h6 {
        color: #e0e5ff !important;
        font-weight: 600;
    }
    
    /* Widgets */
    .stButton>button {
        background: rgba(26, 29, 43, 0.8) !important;
        color: white !important;
        border: 1px solid rgba(100, 110, 200, 0.3) !important;
        border-radius: 12px !important;
    }
    
    /* Tabelas */
    .dataframe {
        background: rgba(20, 23, 40, 0.7) !important;
        color: white !important;
        border-radius: 12px;
    }
    
    .dataframe th {
        background: rgba(70, 80, 150, 0.4) !important;
        color: #e0e5ff !important;
        font-weight: 600;
    }
    
    .dataframe tr:nth-child(even) {
        background: rgba(30, 33, 50, 0.5) !important;
    }
    
    .dataframe tr:hover {
        background: rgba(70, 80, 150, 0.3) !important;
    }
    
    /* Divider */
    .stDivider {
        border-top: 1px solid rgba(100, 110, 200, 0.2) !important;
        margin: 30px 0;
    }
    
    /* Espaçamento entre gráficos */
    .graph-spacer {
        height: 40px;
        background: transparent;
    }
    
    /* Ícones informativos */
    .info-icon {
        font-size: 1.2rem;
        margin-right: 10px;
        color: #00c1e0;
    }
    
    /* Listas formatadas */
    .custom-list li {
        margin-bottom: 10px;
        line-height: 1.6;
    }
    
    .custom-list ul {
        padding-left: 25px;
        margin-top: 8px;
    }
    
    .custom-list code {
        background: rgba(100, 110, 200, 0.2);
        padding: 2px 6px;
        border-radius: 4px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Configurar matplotlib para tema escuro premium
plt.style.use('dark_background')
mpl.rcParams.update({
    'axes.facecolor': '#131625',
    'figure.facecolor': '#0c0f1d',
    'axes.edgecolor': '#6f42c1',
    'axes.labelcolor': '#e0e5ff',
    'text.color': '#e0e5ff',
    'xtick.color': '#a0a7c0',
    'ytick.color': '#a0a7c0',
    'grid.color': '#2a2f45',
    'grid.alpha': 0.4,
    'font.family': 'Segoe UI',
    'axes.titleweight': '600',
    'axes.titlesize': 14,
})

## Mapeamento de parâmetros para nomes amigáveis
PARAM_MAPPING = {
    "Total N (g/kg)": "Nitrogênio Total (N)",
    "Total P (g/kg)": "Fósforo Total (P)",
    "Total K (g/kg)": "Potássio Total (K)",
    "pH": "pH",
    "C/N ratio": "Relação C/N"
}

## Mapeamento de dias para ordenação numérica
DAY_MAPPING = {
    'Day 15': 15,
    'Day 30': 30,
    'Day 180': 180,
    'Day 360': 360,
    'Day 540': 540,
    'Day 720': 720
}

## Função para Carregar Dados de Santana et al. (2020)
@st.cache_data
def load_sample_data_with_stdev():
    # Dados baseados na Tabela 1 do artigo
    sample_param_data = {
        'Total N (g/kg)': {
            'Day 15': {'mean': 39.30, 'stdev': 2.42},
            'Day 30': {'mean': 41.70, 'stdev': 1.99},
            'Day 180': {'mean': 40.05, 'stdev': 3.07},
            'Day 360': {'mean': 37.22, 'stdev': 2.42},
            'Day 540': {'mean': 38.06, 'stdev': 2.81},
            'Day 720': {'mean': 39.42, 'stdev': 2.84}
        },
        'Total P (g/kg)': {
            'Day 15': {'mean': 5.13, 'stdev': 0.45},
            'Day 30': {'mean': 6.27, 'stdev': 0.77},
            'Day 180': {'mean': 5.44, 'stdev': 2.20},
            'Day 360': {'mean': 4.30, 'stdev': 1.10},
            'Day 540': {'mean': 3.85, 'stdev': 0.96},
            'Day 720': {'mean': 3.71, 'stdev': 0.93}
        },
        'Total K (g/kg)': {
            'Day 15': {'mean': 24.97, 'stdev': 3.47},
            'Day 30': {'mean': 24.77, 'stdev': 3.49},
            'Day 180': {'mean': 20.29, 'stdev': 7.36},
            'Day 360': {'mean': 17.34, 'stdev': 4.57},
            'Day 540': {'mean': 23.05, 'stdev': 4.64},
            'Day 720': {'mean': 15.62, 'stdev': 4.40}
        },
        'pH': {
            'Day 15': {'mean': 7.72, 'stdev': 0.18},
            'Day 30': {'mean': 7.73, 'stdev': 0.19},
            'Day 180': {'mean': 7.65, 'stdev': 0.21},
            'Day 360': {'mean': 7.31, 'stdev': 0.20},
            'Day 540': {'mean': 7.46, 'stdev': 0.14},
            'Day 720': {'mean': 7.49, 'stdev': 0.10}
        },
        'C/N ratio': {
            'Day 15': {'mean': 1.26, 'stdev': 0.08},  # Calculado: C_total/N_total
            'Day 30': {'mean': 1.16, 'stdev': 0.06},
            'Day 180': {'mean': 1.11, 'stdev': 0.09},
            'Day 360': {'mean': 1.19, 'stdev': 0.07},
            'Day 540': {'mean': 1.16, 'stdev': 0.09},
            'Day 720': {'mean': 1.26, 'stdev': 0.07}
        }
    }

    num_replications = 3
    days = list(DAY_MAPPING.keys())
    all_replicated_data = []

    for param_name, daily_stats in sample_param_data.items():
        for _ in range(num_replications):
            row_data = {'Parameter': param_name, 'Substrate': 'Grape Marc'}
            for day in days:
                stats = daily_stats.get(day)
                if stats:
                    simulated_value = np.random.normal(
                        loc=stats['mean'], 
                        scale=stats['stdev']
                    )
                    
                    if param_name == 'pH':
                        simulated_value = np.clip(simulated_value, 0.0, 14.0)
                    elif 'g/kg' in param_name or 'ratio' in param_name:
                        simulated_value = max(0.0, simulated_value)
                    
                    row_data[day] = simulated_value
                else:
                    row_data[day] = np.nan
            all_replicated_data.append(row_data)

    return pd.DataFrame(all_replicated_data)

## Função para plotar evolução temporal
def plot_parameter_evolution(ax, data, days, param_name):
    # Converter dias para numérico para ordenação
    numeric_days = [DAY_MAPPING[d] for d in days]
    
    # Paleta de cores moderna
    colors = ['#6f42c1', '#00c1e0', '#00d4b1', '#ffd166', '#ff6b6b', '#a78bfa']
    
    for i, (day, num_day) in enumerate(zip(days, numeric_days)):
        group_data = data[i]
        
        # Plotar pontos individuais
        ax.scatter(
            [num_day] * len(group_data), 
            group_data, 
            alpha=0.85, 
            s=100,
            color=colors[i % len(colors)],
            edgecolors='white',
            linewidth=1.2,
            zorder=3,
            label=f"{day.replace('Day ', 'Dia ')}",
            marker='o'
        )
    
    # Calcular e plotar medianas
    medians = [np.median(group) for group in data]
    ax.plot(
        numeric_days, 
        medians, 
        'D-', 
        markersize=10,
        linewidth=3,
        color='#ffffff',
        markerfacecolor='#6f42c1',
        markeredgecolor='white',
        markeredgewidth=1.5,
        zorder=5,
        alpha=0.95
    )
    
    # Configurar eixo X com dias numéricos
    ax.set_xticks(numeric_days)
    ax.set_xticklabels([str(d) for d in numeric_days], fontsize=11)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Melhorar formatação
    ax.set_xlabel("Dias de Vermicompostagem", fontsize=12, fontweight='bold', labelpad=15)
    ax.set_ylabel(PARAM_MAPPING.get(param_name, param_name), fontsize=12, fontweight='bold', labelpad=15)
    ax.set_title(f"Evolução do {PARAM_MAPPING.get(param_name, param_name)}", 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Grid e estilo
    ax.grid(True, alpha=0.2, linestyle='--', color='#a0a7c0', zorder=1)
    ax.legend(loc='best', fontsize=10, framealpha=0.25)
    
    # Remover bordas
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Fundo gradiente
    ax.set_facecolor('#0c0f1d')
    
    return ax

## Função para exibir resultados com design premium
def display_results_interpretation(results):
    st.markdown("""
    <div class="card">
        <h2 style="display:flex;align-items:center;gap:10px;">
            <span style="background:linear-gradient(135deg, #a78bfa 0%, #6f42c1 100%);padding:5px 15px;border-radius:30px;font-size:1.2rem;">
                📝 Interpretação dos Resultados
            </span>
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    if not results:
        st.info("Nenhuma interpretação disponível, pois não há resultados estatísticos.")
        return
    
    for res in results:
        param_name = res["Parâmetro"]
        p_val = res["p-value"]
        is_significant = p_val < 0.05
        
        card_class = "signif-card" if is_significant else "not-signif-card"
        icon = "✅" if is_significant else "❌"
        title_color = "#00c853" if is_significant else "#ff5252"
        status = "Significativo" if is_significant else "Não Significativo"
        
        st.markdown(f"""
        <div class="result-card {card_class}">
            <div style="display:flex; align-items:center; justify-content:space-between;">
                <div style="display:flex; align-items:center; gap:12px;">
                    <div style="font-size:28px; color:{title_color};">{icon}</div>
                    <h3 style="margin:0; color:{title_color}; font-weight:600;">{param_name}</h3>
                </div>
                <div style="background:rgba(42, 47, 69, 0.7); padding:8px 18px; border-radius:30px; border:1px solid {title_color}30;">
                    <span style="font-weight:bold; font-size:1.1rem; color:{title_color};">{status}</span>
                    <span style="color:#a0a7c0; margin-left:8px;">p = {p_val:.4f}</span>
                </div>
            </div>
            <div style="margin-top:20px; padding-top:15px; border-top:1px solid rgba(100, 110, 200, 0.2);">
        """, unsafe_allow_html=True)
        
        if is_significant:
            st.markdown("""
                <div style="color:#e0e5ff; line-height:1.8;">
                    <p style="margin:12px 0; display:flex; align-items:center; gap:8px;">
                        <span style="color:#00c853; font-size:1.5rem;">•</span>
                        <b>Rejeitamos a hipótese nula (H₀)</b>
                    </p>
                    <p style="margin:12px 0; display:flex; align-items:center; gap:8px;">
                        <span style="color:#00c853; font-size:1.5rem;">•</span>
                        Há evidências de que os valores do parâmetro mudam significativamente ao longo do tempo
                    </p>
                    <p style="margin:12px 0; display:flex; align-items:center; gap:8px;">
                        <span style="color:#00c853; font-size:1.5rem;">•</span>
                        A vermicompostagem afeta este parâmetro
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="color:#e0e5ff; line-height:1.8;">
                    <p style="margin:12px 0; display:flex; align-items:center; gap:8px;">
                        <span style="color:#ff5252; font-size:1.5rem;">•</span>
                        <b>Aceitamos a hipótese nula (H₀)</b>
                    </p>
                    <p style="margin:12px 0; display:flex; align-items:center; gap:8px;">
                        <span style="color:#ff5252; font-size:1.5rem;">•</span>
                        Não há evidências suficientes de mudanças significativas
                    </p>
                    <p style="margin:12px 0; display:flex; align-items:center; gap:8px;">
                        <span style="color:#ff5252; font-size:1.5rem;">•</span>
                        O parâmetro permanece estável durante o processo de vermicompostagem
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)

## Função Principal
def main():
    # Título com estilo moderno
    st.markdown("""
    <div class="header-card">
        <h1 style="margin:0;padding:0;background:linear-gradient(135deg, #a78bfa 0%, #6f42c1 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-size:2.5rem;">
            🍇 Análise de Vermicompostagem de Bagaço de Uva
        </h1>
        <p style="margin:0;padding-top:10px;color:#a0a7c0;font-size:1.1rem;">
        Estatísticas de parâmetros químicos durante 2 anos de vermicompostagem (Santana et al., 2020)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Inicialização de variáveis
    df = load_sample_data_with_stdev()
    
    # Sidebar premium
    with st.sidebar:
        st.markdown("""
        <div class="card">
            <h3 style="display:flex;align-items:center;gap:10px;">
                <span style="background:linear-gradient(135deg, #a78bfa 0%, #6f42c1 100%);padding:3px 12px;border-radius:30px;font-size:1rem;">
                    📂 Opções de Dados
                </span>
            </h3>
        """, unsafe_allow_html=True)
        
        use_sample = st.checkbox("Usar dados de exemplo", value=True, key="use_sample")
        
        if not use_sample:
            uploaded_file = st.file_uploader("Carregue o artigo PDF", type="pdf", key="pdf_uploader")
            if uploaded_file:
                st.success("Funcionalidade PDF em desenvolvimento. Usando dados de exemplo.")
            else:
                st.info("Nenhum PDF carregado. Usando dados de exemplo.")
        
        st.markdown("""
        <div class="card">
            <h3 style="display:flex;align-items:center;gap:10px;">
                <span style="background:linear-gradient(135deg, #a78bfa 0%, #6f42c1 100%);padding:3px 12px;border-radius:30px;font-size:1rem;">
                    ⚙️ Configuração de Análise
                </span>
            </h3>
        """, unsafe_allow_html=True)
        
        unique_params = df['Parameter'].unique()
        param_options = [PARAM_MAPPING.get(p, p) for p in unique_params]
        
        selected_params = st.multiselect(
            "Selecione os parâmetros:",
            options=param_options,
            default=param_options,
            key="param_select"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h3 style="display:flex;align-items:center;gap:10px;">
                <span style="background:linear-gradient(135deg, #a78bfa 0%, #6f42c1 100%);padding:3px 12px;border-radius:30px;font-size:1rem;">
                    📚 Metodologia Estatística
                </span>
            </h3>
            <div style="color:#d7dce8; line-height:1.7;">
                <p><b>Teste de Kruskal-Wallis</b></p>
                <ul style="padding-left:20px;">
                    <li>Alternativa não paramétrica à ANOVA</li>
                    <li>Compara medianas de múltiplos grupos</li>
                    <li>Hipóteses:
                        <ul>
                            <li>H₀: Distribuições idênticas</li>
                            <li>H₁: Pelo menos uma distribuição diferente</li>
                        </ul>
                    </li>
                    <li>Interpretação:
                        <ul>
                            <li>p &lt; 0.05: Diferenças significativas</li>
                            <li>p ≥ 0.05: Sem evidência de diferenças</li>
                        </ul>
                    </li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Pré-visualização dos Dados
    st.markdown("""
    <div class="card">
        <h2 style="display:flex;align-items:center;gap:10px;">
            <span style="background:linear-gradient(135deg, #a78bfa 0%, #6f42c1 100%);padding:5px 15px;border-radius:30px;font-size:1.2rem;">
                🔍 Dados do Estudo
            </span>
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(df)
    st.markdown(f"**Total de amostras:** {len(df)}")
    
    # Explicação detalhada sobre o estudo
    st.markdown("""
    <div class="info-card">
        <h3 style="display:flex;align-items:center;color:#00c1e0;">
            <span class="info-icon">ℹ️</span> Metodologia do Estudo
        </h3>
        <div style="margin-top:15px; color:#d7dce8; line-height:1.7;">
            <p>O estudo analisou a vermicompostagem de bagaço de uva durante 2 anos:</p>
            
            <ul class="custom-list">
                <li><strong>Sistema vertical</strong>: Vermireator de deposição vertical</li>
                <li><strong>Adições mensais</strong>: Bagaço de uva adicionado mensalmente</li>
                <li><strong>Perfis estratificados</strong>: Camadas de 15 a 720 dias de processamento</li>
                <li><strong>Espécie de minhoca</strong>: <em>Eisenia andrei</em></li>
                <li><strong>Parâmetros analisados</strong>: N, P, K, pH, relação C/N e outros</li>
                <li><strong>Análises estatísticas</strong>: ANOVA e teste de Tukey (p &lt; 0.05)</li>
            </ul>
            
            <p style="margin-top:15px; font-style:italic;">
                Dados simulados com base nos valores médios e desvios padrão reportados na Tabela 1 do artigo.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True) 
    
    st.divider()

    # Realizar Análise
    if not selected_params:
        st.warning("Selecione pelo menos um parâmetro para análise.")
        return

    # Converter de volta para nomes originais
    reverse_mapping = {v: k for k, v in PARAM_MAPPING.items()}
    selected_original_params = [reverse_mapping.get(p, p) for p in selected_params]
    
    results = []
    days_ordered = list(DAY_MAPPING.keys())
    
    # Configurar subplots
    num_plots = len(selected_params)
    
    if num_plots > 0:
        # Criar figura
        fig = plt.figure(figsize=(10, 6 * num_plots))
        gs = fig.add_gridspec(num_plots, 1, hspace=0.6)
        axes = []
        for i in range(num_plots):
            ax = fig.add_subplot(gs[i])
            axes.append(ax)
    
        for i, param in enumerate(selected_original_params):
            param_df = df[df['Parameter'] == param]
            
            # Coletar dados por dia
            data_by_day = []
            valid_days = []
            for day in days_ordered:
                if day in param_df.columns:
                    day_data = param_df[day].dropna().values
                    if len(day_data) > 0:
                        data_by_day.append(day_data)
                        valid_days.append(day)
            
            # Executar teste de Kruskal-Wallis
            if len(data_by_day) >= 2:
                try:
                    h_stat, p_val = kruskal(*data_by_day)
                    results.append({
                        "Parâmetro": PARAM_MAPPING.get(param, param),
                        "H-Statistic": h_stat,
                        "p-value": p_val,
                        "Significativo (p<0.05)": p_val < 0.05
                    })
                    
                    # Plotar gráfico
                    ax = axes[i]
                    plot_parameter_evolution(ax, data_by_day, valid_days, param)
                    
                    # Adicionar resultado do teste
                    annotation_text = f"Kruskal-Wallis: H = {h_stat:.2f}, p = {p_val:.4f}"
                    ax.text(
                        0.5, 0.95, 
                        annotation_text,
                        transform=ax.transAxes,
                        ha='center',
                        va='top',
                        fontsize=11,
                        color='white',
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor='#2a2f45',
                            alpha=0.8,
                            edgecolor='none'
                        )
                    )
                except Exception as e:
                    st.error(f"Erro ao processar {param}: {str(e)}")
                    continue
            else:
                st.warning(f"Dados insuficientes para {PARAM_MAPPING.get(param, param)}")
                continue
    else:
        st.warning("Nenhum parâmetro selecionado para análise.")
        return

    # Resultados Estatísticos
    st.markdown("""
    <div class="card">
        <h2 style="display:flex;align-items:center;gap:10px;">
            <span style="background:linear-gradient(135deg, #a78bfa 0%, #6f42c1 100%);padding:5px 15px;border-radius:30px;font-size:1.2rem;">
                📈 Resultados Estatísticos
            </span>
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    if results:
        # Formatar a tabela de resultados
        results_df = pd.DataFrame(results)
        results_df['Significância'] = results_df['p-value'].apply(
            lambda p: "✅ Sim" if p < 0.05 else "❌ Não"
        )
        
        # Reordenar colunas
        results_df = results_df[['Parâmetro', 'H-Statistic', 'p-value', 'Significância']]
        
        # Estilizar a tabela
        st.dataframe(
            results_df.style
            .format({"p-value": "{:.4f}", "H-Statistic": "{:.2f}"})
            .set_properties(**{
                'color': 'white',
                'background-color': '#131625',
            })
            .apply(lambda x: ['background: rgba(70, 80, 150, 0.3)' 
                               if x['p-value'] < 0.05 else '' for i in x], axis=1)
        )
        
        # Interpretação baseada no artigo
        st.markdown("""
        <div class="info-card">
            <h3 style="display:flex;align-items:center;color:#00c1e0;">
                <span class="info-icon">🔬</span> Padrões Observados no Estudo Original
            </h3>
            <div style="margin-top:15px; color:#d7dce8; line-height:1.7;">
                <ul class="custom-list">
                    <li><strong>Nitrogênio (N)</strong>: Aumento inicial seguido de estabilização (p &lt; 0.05)</li>
                    <li><strong>Fósforo (P)</strong>: Redução significativa durante a maturação (p &lt; 0.001)</li>
                    <li><strong>Potássio (K)</strong>: Redução contínua ao longo do processo (p &lt; 0.001)</li>
                    <li><strong>pH</strong>: Aumento inicial seguido de redução gradual (p &lt; 0.05)</li>
                    <li><strong>Relação C/N</strong>: Redução inicial seguida de estabilização</li>
                </ul>
                <p style="margin-top:15px; font-style:italic;">
                    A vermicompostagem apresentou duas fases distintas: fase ativa (0-30 dias) com mudanças
                    rápidas e fase de maturação (30-720 dias) com alterações mais graduais.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Nenhum resultado estatístico disponível.")
    
    # Gráficos
    if num_plots > 0:
        st.markdown("""
        <div class="card">
            <h2 style="display:flex;align-items:center;gap:10px;">
                <span style="background:linear-gradient(135deg, #a78bfa 0%, #6f42c1 100%);padding:5px 15px;border-radius:30px;font-size:1.2rem;">
                    📊 Evolução Temporal dos Parâmetros
                </span>
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Adicionar espaçamento visual entre os gráficos
        st.markdown('<div class="graph-spacer"></div>', unsafe_allow_html=True)
        
        # Ajustar layout
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    # Interpretação dos resultados
    display_results_interpretation(results)
    
    # Referência Bibliográfica
    st.markdown("""
    <div class="card">
        <h2 style="display:flex;align-items:center;gap:10px;">
            <span style="background:linear-gradient(135deg, #a78bfa 0%, #6f42c1 100%);padding:5px 15px;border-radius:30px;font-size:1.2rem;">
                📚 Referência Bibliográfica
            </span>
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="reference-card">
        <p style="line-height:1.8; text-align:justify;">
            SANTANA, N. A.; JACQUES, R. J. S.; ANTONIOLLI, Z. I.; MARTINEZ-CORDEIRO, H.; DOMINGUEZ, J. 
            Changes in the chemical and biological characteristics of grape marc vermicompost during a two-year production period.
            <strong>Applied Soil Ecology</strong>, 
            v. 154, p. 103587, 2020. 
            Disponível em: https://doi.org/10.1016/j.apsoil.2020.103587. 
        </p>
        <p style="margin-top:20px; font-style:italic;">
            Estudo realizado com bagaço de uva em sistema vertical durante 2 anos, 
            analisando parâmetros químicos, biológicos e enzimáticos.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()