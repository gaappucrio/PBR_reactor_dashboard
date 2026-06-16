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
    Para uma espécie química genérica A, que neste contexto representa o ácido benzóico (BA), a expressão matemática desse balanço é dada por:
    </p>
    """, unsafe_allow_html=True)

    st.latex(r"F_{A0} - F_A + G_A = \frac{dN_A}{dt}  \ \ \ \ \ \ \ \ \ \    (Eq. 1)")
  
    st.write(r"""
    <p style='font-size:20px'>
    Onde <i>F<sub>A0</sub></i> é a vazão molar inicial de entrada, <i>F<sub>A</sub></i> é a vazão molar de saída, <i>G<sub>A</sub></i> é a taxa de geração interna do componente e <i>dN<sub>A</sub></i>/dt representa o acúmulo de matéria no sistema ao longo do tempo.
    Sob a condição de estado estacionário (steady-state), as propriedades do sistema tornam-se invariáveis em relação ao tempo, o que anula o termo de acúmulo (<i>dN<sub>A</sub></i>/dt = 0), reduzindo a expressão para:
    </p>
    """, unsafe_allow_html=True)
   
    st.latex(r"F_{A0} - F_A + G_A = 0")

    st.write(r"""
    <p style='font-size:20px'>
    Diferente de reatores de mistura perfeita, onde a concentração é homogênea em todo o espaço, o PFR apresenta um perfil contínuo de variação de concentração ao longo do seu comprimento.
    Devido a essa natureza distribuída, o balanço deve ser aplicado a um elemento de volume diferencial ($dV$). Dentro desse elemento, a taxa de geração diferencial é definida pelo produto 
    da taxa de reação volumétrica ($r_A$) pelo próprio diferencial de volume ($dV$):
    </p>
    """, unsafe_allow_html=True)

    st.latex(r"dG_A = r_A dV")

    st.write(r"""
    <p style='font-size:20px'>
    Ao aplicar o balanço molar estrito a essa seção infinitesimal, onde a vazão molar que entra é <i>F<sub>A</sub></i> e a que sai é <i>F<sub>A</sub></i> + <i>dF<sub>A</sub></i>, obtém-se:
    </p>
    """, unsafe_allow_html=True)

    st.latex(r"F_A - (F_A + dF_A) + r_A dV = 0")
    st.latex(r"-dF_A + r_A dV = 0")
    
    st.write(r"""
    <p style='font-size:20px'>
    Isolando os termos diferenciais, estabelece-se a equação diferencial clássica de um PFR ideal:
    </p>
    """, unsafe_allow_html=True)

    st.latex(r"\frac{dF_A}{dV} = r_A")

    st.write(r"""
    <p style='font-size:20px'>
    No entanto, quando o processo deixa de ser homogêneo e passa a envolver uma reação catalisada
    por um sólido, a literatura de engenharia química introduz uma variação desse modelo: o Reator de Leito Empacotado (Packed-Bed Reactor – PBR).
    </p>
    """, unsafe_allow_html=True)

    st.write(r"""
    <p style='font-size:20px'>
    Embora ambos os reatores compartilhem exatamente a mesma hipótese fluidodinâmica de escoamento — o modelo de fluxo em pistão ideal, caracterizado
    por um perfil de velocidade plano e ausência de mistura axial —, a diferença fundamental entre o PFR e o PBR reside na coordenada espacial utilizada
    para o dimensionamento e na definição da taxa de reação. Enquanto no PFR assume-se que a reação ocorre de forma homogênea ao longo do volume de fluido (V),
    no PBR a reação é heterogênea e processa-se estritamente nos sítios ativos localizados na massa do catalisador sólido (W).
    </p>
    """, unsafe_allow_html=True)

    st.image("https://raw.githubusercontent.com/gaappucrio/PBR_reactor_dashboard/refs/heads/main/PBR%20Reactor.png", caption="Esquema simplificado de um reator PBR")
    
    st.write(r"""
    <p style='font-size:20px'>
    Essa distinção física exige uma mudança de variável no balanço de massa. Em vez de monitorar o consumo do reagente a cada elemento de volume diferencial (dV),
    passa-se a contabilizar a conversão a cada elemento diferencial de massa de catalisador (dW). A ponte matemática entre os dois conceitos é estabelecida pela
    densidade aparente do leito empacotado (<i>ρ<sub>B</sub></i>), através da relação dW = <i>ρ<sub>B</sub></i>.dV. Dessa forma, a taxa de reação volumétrica (<i>r<sub>A</sub></i>, em mols/volume·tempo)
    é substituída pela taxa de reação específica (<i>r'<sub>A</sub></i>, em mols/massa de catalisador·tempo), mantendo a mesma estrutura matemática linear do fluxo em pistão, mas
    perfeitamente adaptada para sistemas com catalisadores sólidos.
    </p>
    """, unsafe_allow_html=True)

    st.write(r"""
    <p style='font-size:20px'>
    No sistema experimental estudado no artigo, caracterizado como um bead string reactor, o escoamento ocorre através de um leito empacotado com esferas do catalisador sólido Amberlyst-15. 
    Em processos de catálise heterogênea, a reação química processa-se nos sítios ativos localizados na superfície e nos poros do sólido. Por convenção analítica, a taxa de reação é expressa 
    em termos de massa de catalisador (<i>r'<sub>A</sub></i>, em mols reagidos por unidade de massa de catalisador por tempo) em vez de volume de fluido.
    </p>
    """, unsafe_allow_html=True)

    st.write(r"""
    <p style='font-size:20px'>
    A conversão matemática do elemento de volume (dV) para o elemento de massa de catalisador (dW) é feita por meio da relação com a densidade aparente do leito (<i>ρ<sub>b</sub></i>), onde 
    dW = <i>ρ<sub>b</sub></i> dV. Substituindo essa equivalência na equação de desempenho e adaptando a nomenclatura para o ácido benzóico (BA), a expressão assume a forma:
    </p>
    """, unsafe_allow_html=True)

    st.latex(r"\frac{dF_{BA}}{dW} = r'_{BA}")

    st.write(r"""
    <p style='font-size:20px'>
    Para alinhar o modelo matemático às técnicas de monitoramento analítico de bancada, como a cromatografia líquida de alta eficiência (HPLC),
    substitui-se a variável de vazão molar (<i>F<sub>BA</sub></i>) pela concentração molar correspondente (<i>C<sub>BA</sub></i>). A relação fundamental entre essas grandezas é dada por:
    </p>
    """, unsafe_allow_html=True)

    st.latex(r"F_{BA} = C_{BA} \cdot \nu")

    st.write(r"""
    <p style='font-size:20px'>
    Por ser uma constante fluidodinâmica, a vazão volumétrica pode ser extraída do operador diferencial:
    </p>
    """, unsafe_allow_html=True)

    st.latex(r"𝓥\ \cdot \frac{dC_{BA}}{dW} = r'_{BA}")

    st.write(r"""
    <p style='font-size:20px'>
    Através do rearranjo algébrico final, isola-se o gradiente de concentração em função da massa de catalisador, 
    resultando exatamente na equação de desempenho utilizada no artigo científico:
    </p>
    """, unsafe_allow_html=True)

    st.latex(r"\frac{dC_{BA}}{dW} = \frac{r'_{BA}}{\nu}")

    st.write(r"""
    <p style='font-size:20px'>
    A equação demonstra rigorosamente que o perfil de concentração do reagente ao longo do leito catalítico é governado de forma direta 
    pela cinética química local (<i>r'<sub>BA</sub></i>) e inversamente proporcional à velocidade de transporte volumétrico do fluido (𝓥) através do sistema.
    </p>
    """, unsafe_allow_html=True)
    
    st.latex(r"r' _{BA}=\frac{-kC_{BA}C_{EtOH}}{1+K_WC_W}")


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
