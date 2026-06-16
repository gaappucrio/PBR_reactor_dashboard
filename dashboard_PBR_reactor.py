import streamlit as st
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
import pandas as pd

# Variáveis globais de passo e condições iniciais
W0 = 0
dW = 0.001
k0 = 185.3  # L2/mol.s.g
Ea = 68800  # J/mol
R = 8.314  # J/mol.K
Kw = 0.53

# Definição das equações diferenciais (EDOs)
def EDOs(C, W, q, T):
    CBA = C[0]
    CEtOH = C[1]
    Cw = C[2]
    k = k0 * np.exp(-Ea / (R * (T)))
    
    # Proteção para evitar divisão por zero se a vazão for nula ou negativa
    if q <= 0:
        return np.zeros(len(C))
        
    r = k * CBA * CEtOH / ((1 + Kw * Cw) ** 2)
    dCdW = np.zeros(len(C))
    dCdW[0] = -r / q  # dCBAdt
    dCdW[1] = -r / q  # dCEtOHdt
    dCdW[2] = r / q   # dCwdt
    return dCdW

# Configuração da página para layout expandido (wide)
st.set_page_config(layout="wide")

# Título do dashboard
st.title("Desvendando a reação de etilação")

# Criação das abas estruturais
tab1, tab2 = st.tabs(["Embasamento Teórico", "Reator"])

# --- ABA 1: CONTEXTO TEÓRICO ---
with tab1:
    st.header("O que é um reator PBR (Packed Bed Reactor)")
    st.write("""
    <p style='font-size:20px'>
    A dedução da equação de desempenho para um reator de fluxo ideal em estado estacionário fundamenta-se nos princípios básicos de balanço molar estabelecidos
    na literatura clássica de engenharia das reações químicas.
    </p>
    """, unsafe_allow_html=True)

    st.write("""
    <p style='font-size:20px'>
    O ponto de partida para a modelagem de qualquer reator químico é o balanço molar global em um volume de controle, expresso pela relação fundamental:
    </p>
    """, unsafe_allow_html=True)

    st.latex(r"Entrada - Saída + Geracão = Acúmulo")
    
    st.write(r"""
    <p style='font-size:20px'>
    Para uma espécie química genérica *A*, que neste contexto representa o ácido benzóico (*BA*), a expressão matemática desse balanço é dada por:
    </p>
    """, unsafe_allow_html=True)

    st.latex(r"F_{A0} - F_A + G_A = \frac{dN_A}{dt}  \ \ \ \ \ \ \ \ \ \    (Eq. 1)")
  
    st.write(r"""
    <p style='font-size:20px'>
    Onde <i>F<sub>A0</sub></i> é a vazão molar inicial de entrada, <i>F<sub>A</sub></i> é a vazão molar de saída, <i>G<sub>A</sub></i> é a taxa de geração interna do componente e $dN_A/dt$ representa o acúmulo de matéria no sistema ao longo do tempo.
    Sob a condição de estado estacionário (steady-state), as propriedades do sistema tornam-se invariáveis em relação ao tempo, o que anula o termo de acúmulo ($\frac{dN}{dt}$ = 0$), reduzindo a expressão para:
    </p>
    """, unsafe_allow_html=True)
    # st.image("https://github.com/amandalemette/ENG1818/blob/6fb679e023faf5918633c3fd921cb7b46d914e29/Imagens/im6.png?raw=true", caption="Esquema simplificado de um reator PBR")
    
    st.header("O Modelo")
    st.write("""
    <p style='font-size:20px'>
    Para entender a equação de projeto para reatores PBR é necessário primeiro entender como é formulada a equação para reatores PFR. A modelagem de um reator tubular parte da premissa de que o fluido tenha um escoamento empistonado, ou seja, sem gradientes de concentração, de temperatura ou de velocidade de reação. À medida que os reagentes entram e escoam axialmente ao longo do reator, eles são consumidos e a conversão aumenta ao longo do comprimento do reator. Para desenvolver a equação de projeto do PFR, é necessário multiplicar ambos os lados da seguinte equação por -1:
    </p>
    """, unsafe_allow_html=True)
    st.latex(r"\frac{dF_A}{dV} = r_A \rightarrow \frac{-dF_A}{dV} = -r_A  \ \ \ \ \ \ \ \ \ \    (Eq. 1)")
    
    st.write("""
    <p style='font-size:20px'>
    Para um sistema em escoamento, FA tem sido previamente dada em termos da vazão molar de entrada FA0 e da conversão X:
    </p>
    """, unsafe_allow_html=True)
    st.latex(r"dF_A = F_{A0} - F_{A0} X \ \ \ \ \ \ (Eq.2)")
    
    st.write("""
    <p style='font-size:20px'>
    E substituindo em Eq. 1 obtém-se a forma diferencial da equação de projeto para um reator com escoamento empistonado (PFR):
    </p>
    """, unsafe_allow_html=True)
    st.latex(r"F_{A0}\frac{dX}{dV}=-r_A \ \ \ \ \ \ (Eq.3)")
    
    st.write("""
    <p style='font-size:20px'>
    Agora, separamos as variáveis e integramos com os limites V=0 quando X=0 para obter o volume necessário para o reator com escoamento empistonado de modo a atingir uma conversão especificada X:
    </p>
    """, unsafe_allow_html=True)
    st.latex(r"V = F_{A0}\int_{0}^{X}\frac{dX}{-r_A}(Eq.4)")
    
    st.write("""
    <p style='font-size:20px'>
    A dedução das equações de projeto para reatores de leito fixo é análoga à dedução para um PFR. Reatores de leito fixo são tubulares e cheios de partículas. Substituindo FA da Eq.2 na equação:
    </p>
    """, unsafe_allow_html=True)
    st.latex(r"\frac{dF_A}{dW}=r_A")
    
    st.write("""
    <p style='font-size:20px'>
    De modo a obter:
    </p>
    """, unsafe_allow_html=True)
    st.latex(r"W = F_{A0}\int_{0}^{X}\frac{dX}{-r_A}")
    
    st.write("""
    <p style='font-size:20px'>
    Onde a equação acima pode ser usada para determinar a massa W de catalisador necessária para atingir uma conversão X quando a pressão total permanece constante.
    </p>
    """, unsafe_allow_html=True)
    
    st.header("Considerações do modelo utilizado")
    st.write("""
    <p style='font-size:20px'>
    Uma das condições do modelo é a garantia da utilização de uma razão molar maior do que 9:1 de etanol/ácido benzóico.
    </p>
    """, unsafe_allow_html=True)
    st.latex(r"r_{BA}=\frac{-kC_{BA}C_{EtOH}}{1+K_WC_W}")


# --- ABA 2: SIMULAÇÃO DO REATOR ---
with tab2:
    col1, col2 = st.columns([3, 7])
    
    with col1:
        Massa_cat = st.number_input(
            "Massa de Catalisador (g)", 
            value=0.100, 
            min_value=0.001, 
            step=0.001, 
            format="%.3f",
            key='massa_cat', 
            help='Insira a massa do catalisador em gramas.'
        )

        CBA0 = st.number_input(
            "Concentração inicial de Ácido Benzóico (mol/uL)", 
            value=1.358, 
            min_value=0.001, 
            step=0.001, 
            format="%.3f",
            key='cba0', 
            help='Insira a concentração inicial de Ácido Benzóico.'
        )

        Temp_C = st.number_input(
            "Temperatura (°C)", 
            value=93.13, 
            min_value=-273.15, 
            step=1.0, 
            key='temp', 
            help='Insira a temperatura em graus Celsius.'
        )
        Temp = Temp_C + 273.15  # Conversão automática para Kelvin

        q_input = st.number_input(
            "Vazão (uL/min)", 
            value=15.0, 
            min_value=0.001, 
            step=1.0, 
            key='vazao', 
            help='Insira a vazão volumétrica em uL/min.'
        )
        # Conversão de uL/min para L/s
        q = q_input / (60 * 1000000) 

    # Variáveis e condições dependentes calculadas dinamicamente
    CEtOH0 = 9 * CBA0  
    Cw0 = 0.0          
    C0 = [CBA0, CEtOH0, Cw0]
    
    # Construção correta e dinâmica do vetor W usando o passo global dW
    W = np.arange(W0, Massa_cat + dW, dW)

    # Inicialização do histórico de resultados (Session State)
    if 'results' not in st.session_state:
        st.session_state['results'] = []

    with col1:
        if st.button('Rodar Código'):
            C = odeint(EDOs, C0, W, args=(q, Temp))
            st.session_state['results'].append(C)
            
            # Mantém apenas as duas últimas rodadas no histórico (Cenário atual e anterior)
            if len(st.session_state['results']) > 2:
                st.session_state['results'] = st.session_state['results'][-2:]
                
    with col2:
        if 'results' in st.session_state and len(st.session_state['results']) > 0:
            fig = go.Figure()
            for i, result in enumerate(st.session_state['results']):
                is_current = (i == len(st.session_state['results']) - 1)
                
                opacity = 1.0 if is_current else 0.4
                name_suffix = '' if is_current else ' (antigo)'
                line_style = 'solid' if is_current else 'dot'
                
                # ALTERAÇÃO: A espessura (width) para as linhas normais é 2.5 e para o pontilhado 50% maior é 3.75
                line_width = 2.5 if is_current else 3.75
                
                # Adiciona a curva do Ácido Benzóico (Travado em Azul)
                fig.add_trace(go.Scatter(
                    x=W, 
                    y=result[:, 0], 
                    mode='lines', 
                    name=f'Ácido Benzóico{name_suffix}', 
                    opacity=opacity,
                    line=dict(color='blue', dash=line_style, width=line_width)
                ))
                
                # Adiciona a curva do Etil Benzeno (Travado em Verde)
                fig.add_trace(go.Scatter(
                    x=W, 
                    y=result[:, 2], 
                    mode='lines', 
                    name=f'Etil Benzeno{name_suffix}', 
                    opacity=opacity,
                    line=dict(color='green', dash=line_style, width=line_width)
                ))
                
            fig.update_layout(
                xaxis_title='Catalisador (g)',
                yaxis_title='Concentração (mol/uL)',
                legend_title='Componentes',
                autosize=False,
                width=1000,   
                height=600,   
                font=dict(size=14)
            )
            st.plotly_chart(fig)

            # Geração da Tabela de Dados baseada no último vetor calculado
            results_df = pd.DataFrame(
                st.session_state['results'][-1], 
                columns=["Ácido Benzóico (mol/uL)", "Etanol (mol/uL)", "Etil Benzeno (mol/uL)"]
            )
            st.markdown("### Tabela de Concentrações")
            st.dataframe(results_df, width=1000, height=600)
