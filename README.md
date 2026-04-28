# 📌 Project Title

> Hybrid RAG reproduction: 벡터 검색과 그래프 기반 탐색을 결합하여 질의응답 성능 향상을 검증하는 프로젝트
> 

---

# 🚀 Overview

이 프로젝트는 **기존 Vector RAG가 여러 문서나 규정의 구조적 관계를 충분히 반영하지 못해 복합적인 질의에서 한계를 보일 수 있는 문제**를 해결하기 위해 시작되었습니다.

**벡터 유사도 기반 검색과 그래프 기반 탐색을 결합한 HybridRAG 접근 방식**을 활용하여 **기존 방식 대비 더 나은 검색 품질과 응답 성능을 확인하는 것**을 목표로 합니다.

### 🎯 Goals

- 문제 정의: 단순 벡터 검색만으로는 문서 간 관계나 계층 구조를 충분히 활용하기 어려움
- 해결 방식: 벡터 검색 결과를 시작점으로 삼아 그래프 기반 관계 탐색을 함께 수행하는 HybridRAG 재현
- 기대 효과: 기존 RAG 대비 검색 품질 및 질의 응답 성능 향상 여부 확인, 이후 규정 기반 QA 시스템으로 확장 가능성 검토

---

# 🧠 Key Features

![image.png](docs/image.png)

- ✅ Feature 1: 벡터 유사도를 기반으로 관련 문서를 1차 검색
- ✅ Feature 2: 그래프 구조를 활용해 문서 간 관계 및 연결 정보 탐색
- ✅ Feature 3: 최종 컨텍스트를 LLM에게 전달하여 최종 응답 생성
    - 최종 컨텍스트는 다음과 같이 구성됨.
        - **시작 노드 ($N_{i}$)**: 이 노드가 가지고 있는 원문 청크 텍스트입니다 (`decode(N_{i}.embedding)`).
        - **관계 정보 ($R_{ij}, N_{j}$)**: 시작 노드와 연결된 **'주어-서술어-목적어' 형태의 트리플 구조 자체**입니다 (`{(N_{i}, R_{ij}, N_{j})}`).

---

# 🏗️ Project Structure

```
project-root/
│
├── frontend/        # (확장 시) 웹 UI / Client
├── backend/         # 질의 처리 및 API
├── data/            # PDF 문서, chunking, relation extraction 결과
├── models/          # 임베딩, LLM, retrieval 관련 코드
├── docs/            # 논문 정리, 실험 기록, 설계 문서
└── README.md
```

---

# ⚙️ Tech Stack

### 🔹 Frontend

- 미정

### 🔹 Backend

- Python

### 🔹 AI / Data

- LLM
- Vector RAG
- Graph RAG
- PDF chunking

### 🔹 Infra

- 벡터 DB
- 그래프 저장소

---

# 🔍 How It Works

1. 입력 (User / Data)
    - 사용자가 질문을 입력하고, 시스템은 문서 PDF 및 전처리된 chunk/graph 정보를 활용함
2. 처리 (Logic / Model)
    - 벡터 유사도로 관련 chunk를 우선 검색
    - 검색 결과를 시작점으로 그래프 관계를 따라 추가 정보 탐색
    - 두 결과(시작 노드와 연결된 관계 정보)를 결합해 응답 생성
    
    > **삼성전자 예시로 본 데이터의 형태**
    
    사용자가 "삼성전자의 1분기 영업이익이 급증한 핵심 이유는 무엇인가?"라고 질문했다고 가정
    
    **① $decode(N_{i}.embedding)$ (시작 노드의 텍스트 원문)**
    벡터 유사도 검색을 통해 찾아낸 가장 관련 있는 텍스트 뭉치(Chunk)입니다.
    "삼성전자는 2024년 1분기 연결 기준 영업이익 6.6조 원을 기록했다. 이는 전년 동기 대비 931% 증가한 수치다. 특히 메모리 반도체(DS) 부문이 업황 회복에 따라 흑자 전환에 성공하며 전체 실적을 견인했다."
    
    **② $\{(N_{i}, R_{ij}, N_{j})\}$ (명시적 관계 정보 - 트리플)**
    그래프 데이터베이스에서 이 시작 노드와 연결된 **논리적 지도**입니다.
    
    - `(삼성전자) - [기록하다] - (6.6조 원)`
    - `(6.6조 원) - [이다] - (영업이익)`
    - `(영업이익) - [상태] - (931% 증가)`
    - `(메모리 반도체 부문) - [수행하다] - (흑자 전환)`
    - `(흑자 전환) - [원인이다] - (영업이익 증가)`
    > 
3. 출력 (Result)
    - 질의에 대한 최종 답변 제공
    - 기존 Vector RAG 대비 성능 비교 가능

---

# 🧪 Experiment / Evaluation

- Dataset:
    1. [ARAGOG (Advanced RAG Output Grading) ](https://github.com/predlico/ARAGOG)
    2. [서울과학기술대학교 규정](https://www.seoultech.ac.kr/intro/uvstat/rules/)
- Metrics: 검색 정확도, QA 성능, 기존 방식 대비 향상 정도
- 성능 평가 기준
    
    
    | 지표 | 무엇을 평가하는가 | 핵심 질문 |
    | --- | --- | --- |
    | Context Precision | 검색 결과의 정확성 | 가져온 문맥이 정말 관련 있는가? |
    | Context Recall | 검색 결과의 포괄성 | 필요한 문맥을 충분히 가져왔는가? |
    | Faithfulness | 답변의 근거 충실성 | 답변이 검색 문맥에 기반하고 있는가? |
    | Answer Relevancy | 질문-답변 관련성 | 답변이 질문 의도에 맞는가? |
    | Answer Correctness | 답변의 정답성 | 답변 내용이 실제로 맞는가? |
    | Answer Similarity | 정답과의 의미적 유사성 | 생성 답변이 참조 정답과 비슷한가? |
    - **문맥 정밀도 (Context Precision)** → 상위 검색 결과 중 **관련 있는 문맥이 얼마나 정확하게 포함되었는지** 평가
        
        $\frac{\sum_{k=1}^{K} Precision@k \times v_k}{\text{Total num of relevant items in the top K results}}$
        
    - **문맥 재현율 (Context Recall)** → 전체 정답 문맥 중 **얼마나 많이 검색해냈는지** 평가
        
        $\frac{|\text{Number of relevant contexts retrieved}|}{|\text{Total number of reference contexts}|}$
        
    - **신뢰도 (Faithfulness)** → 생성된 답변의 주장 중 **근거 문맥으로 뒷받침되는 비율** 평가
        
        $\frac{|\text{Number of supported claims in the generated answer}|}{|\text{Total number of claims in the generated answer}|}$
        
    - **답변 관련도 (Answer Relevancy)** → 생성된 답변이 **질문과 의미적으로 얼마나 관련 있는지** 평가
        
        $\frac{1}{N}\sum_{i=1}^{N}\cos(E_{g_i}, E_o)$
        
    - **답변 정확도 (Answer Correctness)** → 생성된 답변이 **정답과 얼마나 정확하게 일치하는지** 평가
        
        $\frac{|TP|}{|TP| + 0.5 \times (|FP| + |FN|)}$
        
    - **답변 유사도 (Answer Similarity)** → 생성된 답변과 참조 정답이 **의미적으로 얼마나 유사한지** 평가
        
        $a_i \cdot g_i$
        
- **비교군 (Baselines)**:
    - 기존 **Vector RAG**: 단순 벡터 유사도 검색 방식.
    - 기존 **Graph RAG**: Cypher 쿼리 등을 이용한 그래프 검색 방식.
    - **Hybrid RAG** : 두 검색 결과를 단순히 합치거나 요약하는 방식.

---

# 📦 Installation

```bash
git clone https://github.com/no-glass-otacku/Hybrid-RAG-Replication
cd project
```

---

# ▶️ Usage

```bash
# 실행 예시
```

또는

- Web: 추후 확장 예정
- API: `/api/...`

---

# 📈 Roadmap

- [ ]  HybridRAG 논문 reproduction
- [ ]  그래프 구축 파이프라인 구체화
- [ ]  baseline과 성능 비교 실험
- [ ]  규정 기반 QA 시스템으로 확장 검토

---

# 🤝 Team

| Name | Role | Description |
| :--- | :--- | :--- |
| [**@no-glass-otacku**](https://github.com/no-glass-otacku) | **Data & Infrastructure Architect** | **문서 전처리 모듈 개발**: PDF 로드, `RecursiveCharacterTextSplitter`를 이용한 청크 분할 및 메타데이터 관리 기능을 구현합니다.<br><br>**데이터베이스 통합 구축**: \*\*Vector DB(Pinecone 등)\*\*와 \*\*Graph DB(Neo4j 등)\*\*를 연동하고, 각 청크의 임베딩 값을 노드 속성으로 저장하는 파이프라인을 만듭니다.<br><br>**백엔드 API 서버**: 검색 로직을 호출하고 결과를 프론트엔드나 클라이언트에 전달하는 기본적인 백엔드 구조를 설계합니다. |
| [**@MelonChicken**](https://github.com/MelonChicken) | **AI Logic & Performance Analyst** | **하이버리드 검색 알고리즘 구현**: 벡터 유사도로 시작 노드를 찾고, 그래프 관계를 따라 탐색하는 논문의 핵심 로직을 개발합니다.<br><br>**프롬프트 엔지니어링**: 검색된 '원문 텍스트'와 '트리플 정보'를 결합해 LLM이 최적의 답변을 내놓도록 프롬프트를 설계합니다.<br><br>**RAGAS 성능 평가**: **ARAGOG 데이터셋**을 활용하여 **Context Precision, Faithfulness** 등 6가지 지표를 측정하고 기존 방식과 비교 분석합니다. |

---

# 📚 References

- [**[Main] Hybrid RAG: 벡터 유사도 기반 탐색을 통해 증강된 그래프 검색-증강-생성**](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11995493)
    
    *S. Cha, K. Seo, and D. Kim, “Hybrid RAG: Enhancing Graph Retrieval-Augmented Generation through Vector Similarity-based Search,” in 2024 Korean Institute of Broadcast and Media Engineers Fall Conference, 2024.*
    
- [**[Sub] HybridRAG: Integrating Knowledge Graphs and Vector Retrieval Augmented Generation for Efficient Information Extraction**](https://arxiv.org/abs/2408.04948)
- 
    *B. Sarmah, B. Hall, R. Rao, S. Patel, S. Pasquali, and D. Mehta, “HybridRAG: Integrating Knowledge Graphs and Vector Retrieval Augmented Generation for Efficient Information Extraction,” arXiv preprint arXiv:2408.04948v1, Aug. 2024.*
---

# 💡 Contribution

PR / Issue 환영합니다.

---

# 📄 License

- MIT License
