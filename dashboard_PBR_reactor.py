import streamlit as st
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
import pandas as pd

# Constantes
W0 = 0
dW = 0.001
k0 = 185.3  # L2/mol.s.g
Ea = 68800  # J/mol
R = 8.314  # J/mol.K
Kw = 0.53
q = 15 / (60 * 1000000)  # L/s

# Definição das equações diferenciais
def EDOs(C, W, q, T):
    CBA = C[0]
    CEtOH = C[1]
    Cw = C[2]
    k = k0 * np.exp(-Ea / (R * (T)))
    r = k * CBA * CEtOH / ((1 + Kw * Cw) ** 2)
    dCdW = np.zeros(len(C))
    dCdW[0] = -r / q  # dCBAdt
    dCdW[1] = -r / q  # dCEtOHdt
    dCdW[2] = r / q   # dCwdt
    return dCdW

# Configuração da página para layout "wide"
st.set_page_config(layout="wide")

# Título do dashboard
st.title("Desvendando a reação de etilação")

# Adicionando abas
tab1, tab2 = st.tabs(["Embasamento Teórico", "Reator"])

with tab1:
    st.header("O que é um reator PBR (Packed Bed Reactor)")
    st.write("""
    <p style='font-size:20px'>
    Imagine um reator tubular. Este tipo em específico leva em consideração que os reagentes são continuamente consumidos a medida que passam através do reator em estado estacionário. 
    </p>
    """, unsafe_allow_html=True)
    #st.image("C:\\Users\\luan.chagas\\OneDrive - Shell\\Documents\\My Pictures\\Screenshots\\Esquema PFR.png", caption="Esquema simplificado de um reator PFR")
    st.write("""
    <p style='font-size:20px'>
    Ainda é possível considerar que o escoamento é empistonado, ou seja, não há variação da velocidade de fluido com o raio no tubo. Esta simplificação é característica dos reatores tubulares de escoamento empistonado (também chamados PFR). 
    </p>
    """, unsafe_allow_html=True)
    st.write("""
    <p style='font-size:20px'>
    A principal diferença entre os reatores envolvendo reações homogeneas e reações PFR (reações heterogêneas) é que no ultimo caso a reação acontece na superfície de um catalisador sólido. Os reatores de leixo fixo, ou "packed bed reactor", são reatores cheios com partículas de catalizadores ao longo de toda sua extensão.
    </p>
    """, unsafe_allow_html=True)
    st.image("https://github.com/amandalemette/ENG1818/blob/6fb679e023faf5918633c3fd921cb7b46d914e29/Imagens/im6.png?raw=true", caption="Esquema simplificado de um reator PBR")
    st.header("O Modelo")
    st.write("""
    <p style='font-size:20px'>
    Para entender a equação de projeto para reatores PBR é necessárrio primeiro entender como é formulada a equação para reatores PFR. A modelagem de um reator tubular parte da premissa de que o fluido tenha um escoamento empistonado, ou seja, sem gradientes de concentração, de temperatura ou de velocidade de reação, À medida que os reagentes entram e escoam axialmente ao longo do reator, eles são consumidos e a conversão aumenta ao longo do comprimeto do reator. Para desenvolvera equação de projeto do PFR, é necessário multiplicar ambos os lados da seguinte equação por -1:")
    </p>
    """, unsafe_allow_html=True)
    st.latex(r"\frac{dF_A}{dV} = r_A \rightarrow \frac{-dF_A}{dV} = -r_A  \ \ \ \ \ \ \ \ \ \    (Eq. 1)")
    st.write("""
    <p style='font-size:20px'>
    Para um sistema em escoament, FA tem sido previamente dada em termos da vazão molar de entrada FA0 e da conversão X:
    </p>
    """, unsafe_allow_html=True)
    st.latex(r"dF_A = F_{A0} - F_{A0} X \ \ \ \ \ \ (Eq.2)")
    st.write("""
    <p style='font-size:20px'>
    E substituindo em Eq. 1 obtém-se a forma diferencial da equação de projeto para um reator com escoamento empistonado (PFR)")
     </p>
    """, unsafe_allow_html=True)
    st.latex(r"F_{A0}\frac{dX}{dV}=-r_A \ \ \ \ \ \ (Eq.3)")
    st.write("""
    <p style='font-size:20px'>
    Agora, separamos as varáveis e integramos com os limites V=0 quando X=0 para obter o volume necessário para o reator com escoamente empistonado de modo a atingir uma conversão especificada X:
    </p>
    """, unsafe_allow_html=True    )
    st.latex(r"V = F_{A0}\int_{0}^{X}\frac{dX}{-r_A}(Eq.4)")
    st.write("""
    <p style='font-size:20px'>
    A dedução das equações de projeto para reatores de leito fixo é análoga à dedução para um PFR. Reatores de leito fixo são tubulares e cheio de partículas. A dedução das formas diferencial e integral das equações de projeto para os reatores de leito fixo é análoga à dedução para um PFR. Substituindo FA da Eq.2 na equação:
    </p>
    """, unsafe_allow_html=True    )
    st.latex(r"\frac{dF_A}{dW}=r_A")
    st.write("""
    <p style='font-size:20px'>
    De modo a obter:
    """, unsafe_allow_html=True)
    st.latex(r"W = F_{A0}\int_{0}^{X}\frac{dX}{-r_A}")
    st.write("""
    <p style='font-size:20px'>
    Onde a equação acima pode ser usada para determinar a massa W de catalisador necessária para atingir uma conversão X quando a pressão total permanece constante.
    """, unsafe_allow_html=True)
    st.header("Considerações do modelo utilizado")
    st.write("""
    <p style='font-size:20px'>
    Uma das condições do modelo é a garantia da utilização de uma razão molar maior do que 9:1 de etanol/ácido benzóico
    """, unsafe_allow_html=True)
    st.latex(r"r_{BA}=\frac{-kC_{BA}C_{EtOH}}{1+K_WC_W}")

with tab2:
    # Layout com duas colunas, onde o tamanho da col1 é ajustável
    col1, col2 = st.columns([3, 7])
    with col1:
        # Caixa de texto do Streamlit para 'Massa de Catalisador'
        st.markdown("Massa de Catalisador (g)", unsafe_allow_html=True)
        Massa_cat = st.text_input("", "0.1", key='massa_cat', help='Insira a massa do catalisador em gramas.')
        Massa_cat = float(Massa_cat)
        Massa_Cat_vector = np.arange(W0, Massa_cat + dW, dW)

        # Caixa de texto do Streamlit para 'Concentração inicial de Ácido Benzóico'
        st.markdown("Concentração inicial de Ácido Benzóico (mol/uL)", unsafe_allow_html=True)
        CBA0 = st.text_input("", "1.358", key='cba0', help='Insira a concentração inicial de Ácido Benzóico.')
        CBA0 = float(CBA0)

        # Caixa de texto do Streamlit para 'Temperatura'
        st.markdown("Temperatura (°C)", unsafe_allow_html=True)
        Temp = st.text_input("", "93.13", key='temp', help='Insira a temperatura em graus Celsius.')
        Temp = float(Temp) + 273.15  # Converter para Kelvin

    # Condições iniciais para C
    CEtOH0 = 9 * CBA0  # Concentração inicial de CEtOH
    Cw0 = 0.0  # Concentração inicial de Cw
    W = np.arange(0, Massa_cat, 0.01)
    C0 = [CBA0, CEtOH0, Cw0]

    # Inicializar uma lista para armazenar os resultados
    if 'results' not in st.session_state:
        st.session_state['results'] = []

    with col1:
        # Adicionar botão para gerar o gráfico
        if st.button('Rodar Código'):
            # Resolver as equações diferenciais
            C = odeint(EDOs, C0, W, args=(q, Temp))
            st.session_state['results'].append(C)
            # Manter apenas os dois últimos resultados
            if len(st.session_state['results']) > 2:
                st.session_state['results'] = st.session_state['results'][-2:]
    with col2:
        if 'results' in st.session_state and len(st.session_state['results']) > 0:
            # Plotar os resultados usando Plotly
            fig = go.Figure()
            for i, result in enumerate(st.session_state['results']):
                opacity = 1.0 if i == len(st.session_state['results']) - 1 else 0.4
                name_suffix = '' if i == len(st.session_state['results']) - 1 else ' (antigo)'
                mode = 'lines' if i == len(st.session_state['results']) - 1 else 'markers'
                fig.add_trace(go.Scatter(x=W, y=result[:, 0], mode=mode, name=f'Ácido Benzóico{name_suffix}', opacity=opacity))
                fig.add_trace(go.Scatter(x=W, y=result[:, 2], mode=mode, name=f'Etil Benzeno{name_suffix}', opacity=opacity))
            fig.update_layout(
                xaxis_title='Catalisador (g)',
                yaxis_title='Concentração (mol/uL)',
                legend_title='Componentes',
                autosize=False,
                width=1000,   # Aumentar a largura do gráfico
                height=600,   # Aumentar a altura do gráfico
                font=dict(size=50)  # Aumentar o tamanho da fonte
            )
            st.plotly_chart(fig)

            # Criar tabela dos vetores gerados
            results_df = pd.DataFrame(st.session_state['results'][-1], columns=["Ácido Benzóico (mol/uL)", "Etanol (mol/uL)", "Etil Benzeno (mol/uL)"])
            st.markdown("### Tabela de Concentrações")

            st.dataframe(results_df, width=1000, height=600)
