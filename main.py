from fastapi import FastAPI
from pydantic import BaseModel
import time
import os
from langchain_groq import ChatGroq # Mudamos a importação para o Groq
import os
from dotenv import load_dotenv

load_dotenv()


# =====================================================
# CONFIGURANDO A API DO GROQ
# =====================================================

# Coloque a sua chave gerada no site do Groq aqui
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

print(
    "GROQ KEY:",
    GROQ_API_KEY[:10] + "..."
)

if not GROQ_API_KEY:
  raise Exception("GROQ_API_KEY não encontrada")

local_llm = ChatGroq(
    temperature=0.3, # Um pouco de criatividade, mas mantendo a precisão financeira
    model_name="llama-3.1-8b-instant", # Modelo super rápido e inteligente disponível no Groq
    api_key=GROQ_API_KEY
)

# =====================================================
# FASTAPI
# =====================================================

app = FastAPI()

# =====================================================
# MODELO DA REQUISIÇÃO
# =====================================================

#é aqui no RequestData que transformamos a mensagem e o resumo financeiro em um objeto que o FastAPI consegue entender e validar
class RequestData(BaseModel):
    message: str
    summary: dict = {}

# =====================================================
# ENDPOINT
# =====================================================

@app.post("/analysis")
async def analysis(data: RequestData):

    print("=" * 50)
    print("ENTROU NO ENDPOINT")
    print("Mensagem:", data.message)

    print("SUMMARY:")
    print(data.summary)

    print("=" * 50)

    try:
        start = time.time()
        
        prompt = f"""
        Você é o NIQ, um conselheiro financeiro pessoal amigável, direto e inteligente do aplicativo.
        Sua missão é dar dicas, analisar a saúde financeira geral e ajudar no planejamento com base no panorama atual.

        ==================================================
        PANORAMA FINANCEIRO ATUAL DO USUÁRIO
        ==================================================
        - Entradas Totais: R$ {data.summary.get("entradasFormatadas", "0,00")}
        - Saídas Totais: R$ {data.summary.get("saidasFormatadas", "0,00")}
        - Saldo Atual: R$ {data.summary.get("saldoFormatado", "0,00")}

        ÚLTIMAS MOVIMENTAÇÕES (Apenas contexto recente):
        {data.summary.get("ultimasTransacoes", [])}
        ==================================================
        MENSAGEM DO USUÁRIO:
        "{data.message}"
        ==================================================

        REGRAS DE COMPORTAMENTO:
        1. REDIRECIONAMENTO EDUCADO: Se o usuário pedir o saldo ou transações de um MÊS ESPECÍFICO do passado (ex: "março", "ano passado", "mês passado"), explique educadamente que você foca no panorama atual e oriente-o a usar o aplicativo. 
           - Exemplo de resposta: "Como eu analiso o seu cenário geral e recente, não consigo puxar dados de meses anteriores por aqui. Mas você pode conferir isso facilmente voltando os meses lá na sua aba de 'Transações'! Sobre o seu cenário de agora, tem alguma outra dúvida?"
        
        2. FOCO EM CONSELHOS: Se o usuário pedir uma análise ("Como estão minhas finanças?"), foque em dizer se o saldo está positivo/saudável, onde ele pode melhorar e dê dicas práticas com base nas categorias que você vê nas últimas movimentações.
        
        3. NATURALIDADE: NUNCA narre suas ações (ex: "Vou analisar", "Sua pergunta foi"). Apenas converse naturalmente. Responda perguntas curtas com respostas curtas.
        
        4. LINGUAGEM SIMPLES: Use uma linguagem simples, amigável e direta. Evite termos técnicos ou jargões financeiros complexos. Seja claro, objetivo, SIMPÁTICO, utíl e com uma personalidade jovem e encantadora.

        5. CONTEXTUALIZAÇÃO: Sempre que possível, contextualize suas respostas com base nas últimas movimentações do usuário. Use exemplos práticos e reais para ilustrar seus conselhos.

        6. NÃO FAÇA PREVISÕES: Evite fazer previsões financeiras ou prometer resultados futuros. Foque em análises baseadas no panorama atual e histórico recente.
        
        7. Evitar redundância: Evite repetir a pergunta do usuário ou informações já fornecidas.

        8. Evite julgar valores como altos ou baixos sem uma referência. Compare sempre entradas e saídas.
        
        """
        print("INICIANDO IA (VIA GROQ)...")

        # Invocamos o Groq e extraímos o conteúdo da resposta com .content
        resultado = local_llm.invoke(prompt).content

        print("IA FINALIZOU")

        print("RESULTADO:")
        print(resultado)
        print(f"Tempo: {time.time() - start:.2f}s")

        return {
            "status": "sucesso",
            "resposta_agente": str(resultado)
        }

    except Exception as e:

        print("ERRO NO PYTHON:")
        print(e)

        return {
            "status": "erro",
            "mensagem": str(e)
        }