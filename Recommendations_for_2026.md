# Направления исследований ошибок LLM: приоритеты на 2026

**Дата:** 2025-11-16
**Версия:** 2.0 (Академическая редакция)
**Основание:** Gap_Map.md + Fact_Ledger.json v2.0 + New_Study_v1.md
**Цель:** Определить приоритетные направления научных исследований для заполнения критических пробелов в понимании ошибок пользователей при работе с LLM

---

## Executive Summary

На основе систематического gap-анализа 4 документов выявлено **10 критических пробелов** в текущем понимании ошибок LLM. Данный документ формулирует **5 приоритетных направлений исследований** на 2026 год:

1. **Таксономия мультимодальных ошибок** (критический gap, покрытие ~5%)
2. **Методология верификации когнитивных искажений** (4 факта со статусом "partial")
3. **RAG-специфичные паттерны ошибок** (покрытие ~40%)
4. **Теория надёжности в условиях неизбежных галлюцинаций** (покрытие ~30%)
5. **Security: систематизация атак и защит** (покрытие ~55%)

**Методологический акцент:** Смещение от описательных case studies к воспроизводимым экспериментам, количественным benchmarks и теоретическим моделям.

---

## 1. Мультимодальные ошибки: критический пробел

### 1.1 Текущее состояние (Gap Analysis)

**Проблема:** Мультимодальность имеет наименьшее покрытие (~5%) в проанализированных документах:
- D1 (noError_with_AI_for_users.md): 0% — не упоминается
- D2 (Error_with_AI_Annual.md): ~10% — упоминание image prompt placement
- D3 (Error_with_AI_1-4_summary.md): ~5% — факт F13 (неподтверждённый)
- D4 (Academic_Surveys_on_Prompt_Engineering.md): ~10% — vision models в обзоре

**Критичность:** Высокая — vision models (GPT-4V, Claude 3.5 Sonnet, Gemini Pro Vision) активно используются, но ошибки пользователей систематически не исследованы.

### 1.2 Исследовательские вопросы (RQ)

**RQ1.1:** Какие типы ошибок специфичны для мультимодальных промптов (отличны от текстовых)?

**RQ1.2:** Как размещение изображения (начало/середина/конец промпта) влияет на точность ответов?
- **Статус в Fact_Ledger:** F13 = "partial" — исследование Anthropic о "interleaved" формате
- **Необходимо:** Воспроизводимый эксперимент на 3+ моделях (GPT-4V, Claude, Gemini), 3+ типах задач (OCR, reasoning, diagram understanding)

**RQ1.3:** Существуют ли когнитивные искажения, специфичные для мультимодальности (например, приоритизация визуальной информации над текстовой при противоречии)?

**RQ1.4:** Как пользователи формулируют промпты для vision models? Какие паттерны ошибок наиболее частотны?

### 1.3 Предлагаемая методология

#### Фаза 1: Таксономия ошибок
**Метод:** Качественный анализ + grounded theory
1. Сбор корпуса мультимодальных промптов (n≥500): Reddit, StackOverflow, реальные логи (с согласия)
2. Двойная независимая аннотация (inter-rater agreement κ>0.7)
3. Индуктивное построение таксономии (bottom-up)
4. Валидация на независимой выборке

**Expected categories (гипотезы):**
- Ошибки спецификации (unclear task для визуального контента)
- Ошибки разрешения противоречий (текст vs изображение)
- Ошибки cultural context (неучёт культурной специфики образа)
- Ошибки resolution/quality (неправильный формат/размер изображения)

#### Фаза 2: Количественный benchmark
**Метод:** Controlled experiments + eval-набор
1. Создать датасет задач (n≥300): OCR, diagram understanding, scene description, visual reasoning
2. Протестировать 3 модели: GPT-4V, Claude 3.5 Sonnet, Gemini Pro Vision
3. Варьировать: положение изображения (3), формат (JPEG/PNG), разрешение (3 уровня)
4. Метрики: accuracy, error types (по таксономии), latency
5. Статистический анализ: ANOVA для факторов влияния

**Гипотезы для проверки:**
- H1: Изображение в начале промпта → выше accuracy (vs середина/конец)
- H2: Interleaved format (текст-изображение-текст) → лучше для complex reasoning
- H3: Высокое разрешение не всегда улучшает результат (due to compression/tokenization)

#### Фаза 3: User study
**Метод:** Between-subjects experiment
1. Рекрутировать пользователей (n≥60, разные уровни опыта с LLM)
2. Задачи: написать промпт для 10 визуальных задач
3. Условия: с/без best practices guide (2 группы)
4. Измерить: ошибки (по таксономии), время, успешность задачи
5. Квалитативные интервью (n=20): почему допущены ошибки

**Ожидаемый вклад:** Первая систематическая таксономия + количественные данные о частоте ошибок + best practices (evidence-based).

### 1.4 Открытые вопросы
- Как генеративные модели (DALL-E, Midjourney) влияют на паттерны ошибок? (Т.е. text→image vs image→text)
- Как видео- и аудио-модальности специфичны? (Требуют отдельного исследования)

---

## 2. Когнитивные искажения: от "unclear" к "confirmed"

### 2.1 Текущее состояние (Fact_Ledger v2.0)

**Проблема:** 4 факта о когнитивных искажениях имеют статус "partial":
- **F11 (Best-of-N):** Метод существует, но нет строгих количественных данных о снижении ошибок
- **F12 (race-adjusted eGFR):** Формула устарела (2021), но нет эмпирических данных, что LLM её рекомендуют
- **F13 (Image placement):** Interleaved лучше для сложных задач, но не универсальное правило "в начале"

**Также:** 1 факт опровергнут:
- **F7 (45% ошибок в новостях):** Не найдено исследование — требует первичного изучения

### 2.2 Исследовательские вопросы

**RQ2.1 (Best-of-N):** При каких условиях Best-of-N sampling эффективно снижает ошибки пользователей?
- **Методология:** Контролируемый эксперимент
  1. Задачи: reasoning (n=100), factual QA (n=100), code generation (n=100)
  2. Условия: N=1, 3, 5, 10
  3. Модели: GPT-4, Claude 3.5, Llama 3
  4. Метрики: accuracy gain, cost (N×API calls), latency
  5. Анализ: при каких задачах ROI положительный?

**RQ2.2 (race-adjusted eGFR):** Рекомендуют ли медицинские LLM устаревшие клинические формулы?
- **Методология:** Systematic probing
  1. Создать промпты для 20 клинических сценариев (eGFR calculation)
  2. Протестировать: GPT-4, Med-PaLM 2, специализированные медицинские модели
  3. Проверить: используют ли race-adjusted формулу (устаревшая) или CKD-EPI 2021 (актуальная)
  4. Расширить на другие домены: устаревшие рекомендации по диабету, гипертонии, антибиотикам
  5. **Broader RQ:** Как часто LLM рекомендуют устаревшие практики? (Literature vs model knowledge cutoff)

**RQ2.3 (Image placement):** Универсальные правила или task-dependent?
- См. RQ1.2 выше (мультимодальность)

**RQ2.4 (45% новостей):** Существует ли проблема высокой частоты ошибок LLM в новостном контенте?
- **Методология:** Оригинальное исследование
  1. Собрать новостные статьи (n≥200): EU + US медиа, 2023-2025
  2. Сгенерировать summaries через LLM (GPT-4, Claude, Gemini)
  3. Fact-checking: 2 независимых аннотатора + FactCheck.org / Snopes
  4. Категории ошибок: hallucination, omission, distortion, bias
  5. Метрики: % статей с ≥1 ошибкой, severity (minor/major/critical)
  6. Сравнить с baseline: человеческие summaries (если данные доступны)

**Ожидаемый вклад:** Превратить 4 "partial" факта в "supported"/"refuted" с количественными данными. Создать репликационные протоколы для других исследователей.

### 2.3 Теоретический вопрос: модель когнитивных искажений LLM

**Проблема:** Текущие исследования описательные (sycophancy ~58%, overconfidence +20-60%), но нет унифицированной теоретической модели.

**RQ2.5:** Можно ли создать формальную модель, предсказывающую когнитивные искажения LLM на основе архитектуры и обучения?

**Подход:**
- **Hypothesis:** Сервильность возникает из RLHF (reinforcement learning from human feedback), где модель обучается "нравиться" оценщикам
- **Формализация:** Моделировать preference optimization как trade-off между truthfulness и agreeableness
- **Prediction:** Модели с более агрессивным RLHF → выше sycophancy
- **Validation:** Измерить sycophancy на моделях с разной степенью RLHF tuning (base vs chat-optimized)

**Ожидаемый вклад:** Переход от "что" (описание искажений) к "почему" (механизмы) и "как предсказать" (теоретическая модель).

---

## 3. RAG-специфичные ошибки: от практики к теории

### 3.1 Текущее состояние (Gap Analysis)

**Проблема:** RAG покрыт ~40% в документах:
- D1: 0%
- D2: ~60% (chunking, embedding, retrieval)
- D3: ~30% (упоминание RAG как решения)
- D4: ~70% (LangChain, vector databases)

**Gap:** Практические руководства есть (chunking strategies, embedding models), но **систематической таксономии RAG-ошибок нет**.

### 3.2 Исследовательские вопросы

**RQ3.1:** Какие категории ошибок специфичны для RAG-систем (отличны от non-RAG)?

**Expected categories (гипотезы):**
1. **Retrieval errors:** Неправильные chunks извлечены (низкая precision/recall)
2. **Context overflow:** Слишком много chunks → превышение context window → потеря информации
3. **Citation errors:** Неправильная атрибуция источника (hallucinated citations)
4. **Conflict resolution errors:** Противоречивая информация в retrieved chunks
5. **Staleness errors:** Устаревшие документы в базе, модель не знает об обновлениях
6. **Prompt construction errors:** Неправильный шаблон для инъекции retrieved context

**RQ3.2:** Как chunking strategy влияет на типы и частоту ошибок?
- **Методология:** Controlled experiment
  1. Датасет документов (n≥50): technical docs, legal texts, scientific papers
  2. Chunking strategies (5): fixed-size (512 tokens), semantic (paragraph), recursive, sliding window, hierarchical
  3. Задачи (n≥200): factual QA, summarization, reasoning over multiple docs
  4. Метрики: accuracy, retrieval precision@k, hallucination rate, citation accuracy
  5. Анализ: какая стратегия для каких типов документов/задач оптимальна?

**RQ3.3:** Как embedding model влияет на retrieval качество и downstream errors?
- **Методология:** Benchmark
  1. Embedding models (10): OpenAI text-embedding-3, Cohere embed-v3, BGE-large, E5, Instructor, Sentence-BERT, etc.
  2. Языки: English + multilingual (Russian, Chinese, Spanish)
  3. Метрики: retrieval precision@5, recall@10, MRR (Mean Reciprocal Rank)
  4. Downstream: end-to-end QA accuracy (retrieval → LLM generation)
  5. Trade-offs: quality vs cost vs latency

**RQ3.4:** Может ли LLM самостоятельно детектировать retrieval errors?
- **Методология:** Self-consistency experiments
  1. Задать вопрос → получить retrieved chunks
  2. Попросить модель оценить: "Достаточно ли контекста для ответа? Противоречива ли информация?"
  3. Сравнить с ground truth (expert annotation)
  4. Метрики: precision/recall детекции ошибок retrieval
  5. **Application:** Self-RAG (модель решает, нужно ли дополнительное retrieval)

### 3.3 Теоретический вопрос: оптимальность RAG

**RQ3.5:** Существуют ли теоретические границы точности RAG-систем?
- **Подход:** Formal analysis
  - **Given:** Context window C, document base D (size N), chunk size c
  - **Question:** Какова максимальная recall при top-k retrieval?
  - **Trade-off:** Precision vs recall vs context usage
  - **Model:** Information-theoretic bounds (Shannon entropy документов)

**Ожидаемый вклад:** Первая систематическая таксономия RAG-ошибок + практические рекомендации (evidence-based) + теоретические границы.

---

## 4. Надёжность в условиях неизбежных галлюцинаций

### 4.1 Теоретический результат (Fact F8)

**Статус:** Confirmed — Xu et al. (2024), arXiv:2401.11817: "Hallucination is Inevitable"
- **Суть:** Доказано теоретически, что для определённых классов задач LLM не могут полностью избежать галлюцинаций (computational complexity argument)

**Импликации:** Если галлюцинации неизбежны, фокус должен сместиться с "как устранить" на "как минимизировать" и "как детектировать".

### 4.2 Исследовательские вопросы

**RQ4.1:** Для каких классов задач галлюцинации теоретически неизбежны (по Xu et al.)?
- **Методология:** Formal verification
  1. Изучить формализм Xu et al. 2024
  2. Классифицировать задачи: factual QA, reasoning, code generation, creative writing
  3. Определить: какие попадают в "неизбежные", какие нет
  4. **Практический вывод:** Для "неизбежных" задач требуются обязательные механизмы детекции

**RQ4.2:** Какие методы детекции галлюцинаций наиболее эффективны (post-hoc)?
- **Методология:** Systematic benchmark
  1. Датасет: задачи с known ground truth (n≥500)
  2. Генерация ответов: GPT-4, Claude 3.5, Gemini (с намеренными галлюцинациями)
  3. Методы детекции (10):
     - Self-consistency (multi-sampling + majority vote)
     - Perplexity analysis
     - External verification (web search, knowledge base lookup)
     - Confidence calibration (logprobs)
     - Contrastive prompting ("Какие части ответа ты не уверен?")
     - Chain-of-verification (CoVe)
     - Retrieval-augmented verification
     - Human-in-the-loop (baseline)
     - LLM-as-judge (другая модель проверяет)
     - Ensemble methods
  4. Метрики: precision, recall, F1 детекции галлюцинаций; cost (API calls); latency

**RQ4.3:** Можно ли предсказать вероятность галлюцинации до генерации ответа?
- **Методология:** Predictive modeling
  1. Features: prompt characteristics (ambiguity score, entity density), model uncertainty (entropy), task type
  2. Label: binary (ответ содержит галлюцинацию / не содержит)
  3. Датасет: n≥1000 пар (prompt, response) с экспертной разметкой
  4. Models: Logistic regression, Random Forest, Neural classifier
  5. **Application:** Если p(hallucination)>threshold → предупредить пользователя или требовать дополнительную верификацию

**RQ4.4:** Как пользователи реагируют на предупреждения о возможных галлюцинациях?
- **Методология:** User study
  1. Условия (3): no warning, generic warning ("Ответ может содержать неточности"), specific warning ("Высокая вероятность ошибки в дате публикации")
  2. Задачи: factual questions (n=20)
  3. Измерить: reliance on LLM (как часто принимают ответ), fact-checking behavior, trust
  4. **Hypothesis:** Specific warnings → выше fact-checking, ниже overreliance

**Ожидаемый вклад:** Shift от "elimination" к "management" галлюцинаций. Практические протоколы для задач, где галлюцинации неизбежны.

### 4.3 Метрики надёжности (Reliability Metrics)

**Проблема:** Нет стандартизированных метрик для оценки "надёжности" LLM-систем.

**RQ4.5:** Как определить и измерить "reliability" LLM?
- **Предлагаемые метрики:**
  1. **Factual accuracy:** % правильных фактов (на eval-наборе)
  2. **Calibration:** Соответствие confidence ↔ correctness (Expected Calibration Error)
  3. **Consistency:** Одинаковые ответы на перефразированные вопросы (% agreement)
  4. **Robustness:** Устойчивость к adversarial prompts (% атак провалились)
  5. **Source attribution:** % ответов с корректными citations
  6. **Abstention rate:** Как часто модель говорит "не знаю" (vs hallucination)
  7. **Worst-case performance:** Не только mean accuracy, но и min/5th percentile

**Методология:** Создать benchmark-suite для измерения всех 7 метрик, протестировать 5+ моделей, опубликовать leaderboard.

---

## 5. Security: от OWASP Top 10 к формальным методам

### 5.1 Текущее состояние (Gap Analysis)

**Покрытие:** ~55% в документах
- D2: ~80% (OWASP LLM Top 10, NIST GenAI Profile, CVE-2025-32711)
- D3: ~50% (prompt injection, jailbreak)
- D1, D4: ~30% (краткие упоминания)

**Gap:** Практические рекомендации есть, но **формальных методов верификации безопасности нет**.

### 5.2 Исследовательские вопросы

**RQ5.1:** Можно ли формально доказать безопасность промпт-системы?
- **Подход:** Formal verification (по аналогии с software verification)
  1. **Formalization:** Промпт как формальная спецификация (inputs, outputs, constraints)
  2. **Threat model:** Категории атак (prompt injection, jailbreak, data extraction)
  3. **Verification:** Использовать model checking / theorem proving для доказательства "no injection possible"
  4. **Challenge:** LLM non-deterministic → probabilistic guarantees вместо абсолютных
  5. **Expected result:** "With probability p>0.99, no injection succeeds" (monte-carlo verification)

**RQ5.2:** Как классифицировать prompt injection атаки (систематическая таксономия)?
- **Методология:** Grounded theory + dataset curation
  1. Собрать датасет атак (n≥1000): TrustAIRLab, LLMail-Inject, in-the-wild (Reddit, Twitter)
  2. Двойная аннотация (независимо)
  3. Индуктивная категоризация
  4. **Expected categories:**
     - Direct instruction override ("Ignore previous instructions")
     - Role redefinition ("You are now DAN")
     - Context poisoning (malicious input in RAG documents)
     - Multi-turn manipulation (gradual jailbreak)
     - Encoding tricks (base64, ROT13, visual encoding)
     - Payload hiding (translation task, code generation task)
  5. Валидация: автоматическая классификация (ML model) с accuracy>0.85

**RQ5.3:** Эффективны ли существующие защиты (input filtering, output filtering)?
- **Методология:** Adversarial testing
  1. Защиты (8): keyword filtering, perplexity detection, separate LLM-judge, prompt firewall, obfuscation, sandboxing, rate limiting, audit logging
  2. Атаки (датасет из RQ5.2, n≥1000)
  3. Метрики: Attack Success Rate (ASR), False Positive Rate (легитимные запросы заблокированы), latency overhead, cost
  4. **Goal:** Найти Pareto-optimal trade-offs (безопасность vs usability)

**RQ5.4:** Существует ли теоретическая граница детекции prompt injection?
- **Подход:** Computational complexity analysis
  - **Hypothesis:** Детекция injection аналогична spam filtering → no perfect solution (arms race)
  - **Question:** Можно ли доказать, что perfect detection impossible (undecidable / NP-hard)?
  - **Analogy:** Halting problem for LLM security

**Ожидаемый вклад:** Первая формальная модель безопасности LLM-систем. Теоретические границы детекции атак. Практические рекомендации с доказанными гарантиями.

### 5.3 Case study: EchoLeak (F6)

**Статус:** Confirmed — CVE-2025-32711, CVSS 9.3 (critical)
- **Механизм:** Zero-click vulnerability в Microsoft 365 Copilot — утечка email-переписки через права доступа

**RQ5.5:** Как систематически аудитировать LLM-интеграции на уязвимости типа EchoLeak?
- **Методология:** Security audit framework
  1. **Checklist:** 20 типов уязвимостей (на основе OWASP + EchoLeak case)
  2. **Automated scanning:** Статический анализ кода (permissions, data flows)
  3. **Manual review:** Red-team testing (2 недели)
  4. **Formal verification:** Доказательство "data cannot leak outside intended scope"
  5. **Continuous monitoring:** Runtime anomaly detection

**Ожидаемый вклад:** Открытый audit framework + checklist для enterprise LLM deployments.

---

## 6. Кросс-тематические исследовательские направления

### 6.1 Agents и инструментальная рациональность

**Gap:** Agents (ReAct, Plan-and-Execute) упомянуты в D4 (~15%), но **agent-специфичные ошибки пользователей не исследованы**.

**RQ6.1:** Какие новые категории ошибок возникают при использовании LLM agents (с tool calling)?
- **Expected categories:**
  - Unbounded tool calls (бесконечный цикл)
  - Incorrect tool selection (модель выбрала не тот инструмент)
  - Privilege escalation (agent получил больше прав, чем нужно)
  - Adversarial instructions in tool outputs (compromised API response)

**RQ6.2:** Как пользователи формулируют задачи для agents? (Отличия от single-shot промптов)

### 6.2 Multilingual errors

**Gap:** Все документы фокусируются на English, multilingual ошибки не освещены.

**RQ6.3:** Как тип и частота ошибок зависят от языка промпта?
- **Hypothesis:** Low-resource languages → выше hallucination rate
- **Methodology:** Benchmark на 10 языках (English, Russian, Chinese, Spanish, Arabic, Hindi, French, German, Japanese, Portuguese)

### 6.3 Domain-specific errors (медицина, право, финансы)

**Gap:** Общие ошибки исследованы, но **domain-specific паттерны требуют отдельного изучения**.

**RQ6.4:** Существуют ли медицинские ошибки LLM, не встречающиеся в других доменах?
- **Example (F12):** Рекомендации устаревших клинических формул (race-adjusted eGFR)
- **Broader RQ:** Систематический анализ для медицины, права, финансов

---

## 7. Методологические приоритеты на 2026

### 7.1 От описательных к воспроизводимым исследованиям

**Проблема:** Многие текущие исследования — case studies, анекдоты, non-replicable examples.

**Рекомендации:**
1. **Публиковать датасеты:** Все эксперименты должны иметь публичные eval-наборы
2. **Публиковать код:** Репликационные скрипты на GitHub
3. **Стандартизировать метрики:** Использовать общепринятые метрики (BLEU, ROUGE, accuracy, F1) + domain-specific
4. **Pre-registration:** Регистрировать гипотезы до экспериментов (против p-hacking)

### 7.2 Междисциплинарное сотрудничество

**Необходимые области:**
- **NLP/ML:** Моделирование, benchmarks
- **HCI:** User studies, UX design для промпт-интерфейсов
- **Cognitive science:** Теория когнитивных искажений
- **Security:** Formal verification, red-teaming
- **Domain experts:** Медики, юристы, финансисты (для domain-specific errors)

### 7.3 Открытые данные и инструменты

**Рекомендации:**
1. Создать **Unified Benchmark Suite:** Один датасет для тестирования ошибок LLM (все категории: hallucinations, sycophancy, multimodal, RAG, security)
2. Создать **Error Annotation Platform:** Краудсорсинговая платформа для разметки ошибок в LLM-ответах (по аналогии с ImageNet для vision)
3. Создать **Leaderboard:** Публичный рейтинг моделей по надёжности (factuality, calibration, robustness)

---

## 8. Заключение: от практики к науке

**Ключевой тезис:** Текущее состояние исследований ошибок LLM — преимущественно практическое (best practices, кейсы). Необходим переход к **строгой научной методологии**:

1. **Формулировка Research Questions** (а не "что сделать")
2. **Воспроизводимые эксперименты** (публичные датасеты, код)
3. **Количественные метрики** (а не качественные описания)
4. **Теоретические модели** (объяснение, предсказание, формализация)
5. **Peer-review публикации** (EMNLP, ACL, NeurIPS, CHI, USENIX Security)

**5 приоритетов на 2026:**
1. Мультимодальность (таксономия + benchmark)
2. Верификация когнитивных искажений (partial → confirmed)
3. RAG-специфичные ошибки (систематизация)
4. Надёжность при неизбежных галлюцинациях (detection + management)
5. Формальные методы безопасности (verification + audit frameworks)

**Ожидаемый результат к концу 2026:**
- 5+ peer-reviewed публикаций
- 3+ открытых датасета (multimodal errors, RAG errors, security attacks)
- 2+ benchmark suites (reliability metrics, security testing)
- 1 теоретическая модель (cognitive biases in LLMs)

**Следующий шаг:** Формирование исследовательских консорциумов, подача грантовых заявок (NSF, ERC, corporate research programs), организация workshop на EMNLP/NeurIPS 2026.

---

**Дата:** 2025-11-16
**Авторы:** Research Team
**Статус:** Ready for Academic Review
**Target venues:** EMNLP 2026, NeurIPS 2026 (workshops), CHI 2026 (HCI aspects), USENIX Security 2026 (security aspects)
