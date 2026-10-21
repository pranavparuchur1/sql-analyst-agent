# LEARN.md: your study guide to sql-analyst-agent

Read this top to bottom once, then use sections 5 and 6 to drill for interviews. Line links point at the exact code. Open them side by side as you read.

---

## 1. The big picture (plain English)

**Analogy: a junior analyst with a strict manager.**

A business user walks up to a junior analyst and asks, "What were our top 5 products last year?"

1. The analyst first decides **what kind of request this is**. Is it a data question, small talk, or someone asking them to delete the database? (`classify`)
2. They glance at the **list of tables** on the wall and pick the 3 they need, then pull up the **full column details** for just those. (`select_tables`)
3. They **write SQL**. (`generate_sql`)
4. Before running it, a **strict manager checks it**: reads only, known tables only, and a row limit. The manager also asks the database "would this even compile?" (`validate_sql`)
5. If the manager rejects it, the analyst gets the **exact reason** back and tries again, up to 3 times. (the retry loop)
6. If the query is big (4+ tables), the manager says **"show me before you run it."** The analyst waits, possibly overnight, without forgetting where they were. (`human_review` + `interrupt`)
7. They **run it** on a read-only copy with a stopwatch. (`execute_sql`)
8. They **explain the result** in plain English and staple the SQL to the answer, so a follow-up ("now only Texas") can build on it. (`summarize`)

LangGraph is the office procedure that makes this reliable: who does what, in what order, what happens on failure, and a filing cabinet (the **checkpointer**) that remembers every conversation.

The key idea for interviews: **the LLM is an untrusted, fallible worker inside a deterministic control loop.** The LLM writes SQL. Plain Python decides whether it's safe, whether to retry, and whether a human must approve.

---

## 2. Glossary (every concept, with where it lives)

| Concept | What it is, in plain English | Where |
|---|---|---|
| **LLM / chat model** | The text-generating model. `init_chat_model("anthropic:claude-...")` builds the right provider class from a string, so switching providers is a config change. | [config.py:49](app/config.py#L49), [:62](app/config.py#L62) |
| **Prompt template** | A reusable message list with `{slots}`. `ChatPromptTemplate.from_messages([("system", ...), ("human", "{question}")])`. | [prompts.py:16](app/prompts.py#L16), [:66](app/prompts.py#L66) |
| **MessagesPlaceholder** | A slot that takes a whole *list* of messages (the chat history) rather than one string. | [prompts.py:77](app/prompts.py#L77) |
| **LCEL pipe (`\|`)** | LangChain Expression Language. `prompt \| model` chains Runnables: the output of one feeds the next. The result is itself a Runnable with `.invoke()`. | [graph.py:116-119](app/graph.py#L116) |
| **Structured output** | `llm.with_structured_output(SQLQuery)` makes the model return a validated Pydantic object instead of free text. The field descriptions are sent to the model as instructions. | [schemas.py:38](app/schemas.py#L38), [graph.py:118](app/graph.py#L118) |
| **Few-shot** | Example question → SQL pairs inserted into the prompt. They teach house style (aliases, date filters) better than more instructions would. | [prompts.py:64](app/prompts.py#L64) |
| **State** | The shared "clipboard" every node reads and writes, defined as a `TypedDict`. | [state.py:17](app/state.py#L17) |
| **Reducer** | A rule for *merging* a node's update into state. Default: overwrite. `add_messages`: append chat messages. `operator.add`: concatenate lists. | [state.py:20](app/state.py#L20), [:22](app/state.py#L22) |
| **Node** | A plain function `state -> partial update dict`. | [graph.py:127](app/graph.py#L127) (`prepare_turn`), registered at [:273-285](app/graph.py#L273) |
| **Edge** | "After node A, always run node B." | [graph.py:288](app/graph.py#L288) |
| **Conditional edge** | "After node A, call this routing function and go wherever it says." The function is pure: state in, node name out. | [graph.py:261](app/graph.py#L261) (function), [:294](app/graph.py#L294) (wiring) |
| **Command** | A node returns `Command(update={...}, goto="next")`, deciding *and* writing state in one step. The `Command[Literal[...]]` return type tells LangGraph the possible destinations. | [graph.py:138-142](app/graph.py#L138) |
| **interrupt** | `interrupt(payload)` saves a checkpoint and pauses the run. The caller sees `payload`. Resuming with `Command(resume=x)` re-runs the node from the top, and `interrupt()` then returns `x`. | [graph.py:205](app/graph.py#L205) |
| **Checkpointer** | Saves the full state after every step. `SqliteSaver` writes to disk, so threads survive restarts. Tests use `InMemorySaver`. | [graph.py:71](app/graph.py#L71), [:301](app/graph.py#L301) |
| **thread_id** | The key for one conversation in the checkpointer. Same thread_id = same memory. | [api.py:34](app/api.py#L34) |
| **Streaming** | `graph.stream(..., stream_mode="updates")` yields `{node: update}` after each node, which is how the CLI shows steps live. | [cli.py:79](app/cli.py#L79) |
| **RetryPolicy** | Re-runs a node on matching exceptions with exponential backoff. Here it's only for transient LLM errors (rate limits, timeouts). | [graph.py:68](app/graph.py#L68), [:277](app/graph.py#L277) |
| **Time travel** | `get_state_history()` lists every checkpoint. Invoking with an old checkpoint's config replays from that point and forks history. | [api.py:126](app/api.py#L126), [test_graph.py:235](tests/test_graph.py#L235) |

---

## 3. One question's journey through the code

Two turns on thread `t1`: *"Top 5 products by revenue in 2025?"*, then *"now only for Texas"*.

**Turn 1**

1. **API** `POST /ask` checks the thread isn't paused ([api.py:86](app/api.py#L86)), then calls `graph.invoke({"question": ...}, {"configurable": {"thread_id": "t1"}})`.
2. **Checkpointer loads** thread `t1`. It's empty on the first turn.
3. **`prepare_turn`** ([graph.py:127](app/graph.py#L127)) appends `HumanMessage(question)` to `messages` (the reducer appends) and resets every per-turn field. Without the reset, turn 2 would inherit turn 1's `attempts`, `sql` and approval.
4. **`classify`** ([graph.py:138](app/graph.py#L138)) runs `ROUTER_PROMPT | llm.with_structured_output(IntentDecision)` and gets `intent="data_question"`. It returns `Command(goto="select_tables")`.
5. **`select_tables`** ([graph.py:144](app/graph.py#L144)) sends the cheap `table_overview()` and gets back `["orders", "order_items", "products"]`. Then `schema_for(...)` fetches DDL plus 3 sample rows for only those tables. An unknown table name triggers a fallback to all tables ([:154](app/graph.py#L154)).
6. **`generate_sql`** ([graph.py:160](app/graph.py#L160)) increments `attempts` to 1. The SQL prompt contains business rules, schema, few-shot examples, the history (empty) and the question. Suppose the model writes `SELECT SUM(revenue) FROM orders`.
7. **`validate_sql`** ([graph.py:176](app/graph.py#L176)): the sqlglot checks pass (it is a SELECT on a known table), but `EXPLAIN` raises `no such column: revenue`. The node writes `error` and appends to `error_history`.
8. **`route_after_validation`** ([graph.py:261](app/graph.py#L261)) sees an error with attempts 1 < 3, so it returns `"generate_sql"`.
9. **`generate_sql` again**: `retry_feedback` ([graph.py:165](app/graph.py#L165)) now contains the failed SQL and the exact error. The model fixes it. `attempts` becomes 2.
10. **`validate_sql`**: it passes, the validator adds a `LIMIT` if missing, and with 3 tables (< 4) no approval is needed. The router picks `"execute_sql"`.
11. **`execute_sql`** ([graph.py:223](app/graph.py#L223)) runs on the read-only connection with the timeout and returns columns and rows. The router picks `"summarize"`.
12. **`summarize`** ([graph.py:230](app/graph.py#L230)) produces a `FinalAnswer` and appends `AIMessage(answer + "[SQL used]" + sql)` ([:244](app/graph.py#L244)). Then END.
13. A checkpoint was saved after **every** step above. The API returns `status="answered"`.

**Turn 2: "now only for Texas"**

1. The checkpointer restores `messages = [Human, AI(with SQL)]`.
2. `prepare_turn` appends the new HumanMessage and resets per-turn fields.
3. `classify` sees the history, so it recognises a follow-up and labels it `data_question`.
4. `generate_sql`'s prompt includes the last 6 history messages, *including the previous SQL*. The model adds `JOIN customers ... WHERE c.state = 'TX'`. That's now **4 tables**.
5. `validate_sql` sets `needs_approval=True`, and the router sends it to `human_review`.
6. **`human_review`** ([graph.py:201](app/graph.py#L201)) calls `interrupt({...sql, reason...})`. The run **stops**, the state is saved, and the API returns `needs_approval` with the payload.
7. Minutes or days later, `POST /approve {"approved": true}` calls `graph.invoke(Command(resume={...}))`. `human_review` runs again from the top, `interrupt()` returns the decision, and it returns `Command(goto="execute_sql")`.
8. The rest of the turn is the same as turn 1, steps 11-13.

---

## 4. The safety layers, and why each one exists

Think "Swiss cheese model": each layer has holes, but the holes don't line up.

1. **Read-only connection** ([db.py:78](app/db.py#L78)). `file:...?mode=ro` means the *database itself* refuses writes. **Why:** this is the last line of defense if the validator has a bug. It's enforced by the engine, not by our code.
2. **sqlglot validator** ([validator.py:37](app/validator.py#L37)). It parses to an AST, allows exactly one statement, requires a `Query` root, walks the whole tree for write nodes (catching writes hidden in CTEs), and allows known tables or CTE names only. **Why:** read-only doesn't stop `ATTACH`, `PRAGMA` or access to tables you didn't intend to expose. The checks are deterministic and can't be prompt-injected, and their error messages are good retry feedback. It uses an AST and not a regex, because `deleted_flag` is not a `DELETE` ([test_validator.py:53](tests/test_validator.py#L53)).
3. **Row limit** ([validator.py:83](app/validator.py#L83), plus `fetchmany` in db.py). **Why:** "show me all orders" shouldn't ship a million rows to the LLM, which costs tokens and loses context.
4. **Timeout** ([db.py:134](app/db.py#L134)). SQLite calls the progress handler every 10k VM instructions, and returning 1 aborts the query. **Why:** an accidental cartesian join can run for minutes and pin a CPU.
5. **Human approval** ([graph.py:188](app/graph.py#L188)). Queries over 4+ tables pause. **Why:** wide joins are where both cost and "plausible but wrong" answers concentrate. A human edit is re-validated, because humans make mistakes too ([test_graph.py:180](tests/test_graph.py#L180)).

Plus an overarching rule: **errors become feedback, not crashes.** Every DB error is caught and turned into retry feedback, and attempts are capped.

---

## 5. Twenty interview questions with strong answers

**1. Walk me through the architecture.**
A LangGraph state machine: classify → select tables → generate SQL → validate → (approve) → execute → summarize, with a bounded retry loop from validate/execute back to generate. A SQLite checkpointer persists state per thread, which gives memory, pause/resume and time travel. FastAPI and a CLI are thin wrappers that call `invoke`/`stream` with a `thread_id`.

**2. Why LangGraph instead of a single chain or a ReAct agent?**
Chains can't loop, branch or pause. A free-form ReAct agent *could* do all of it, but the LLM would decide the control flow: it might skip validation or loop forever. Here the control flow is explicit code, and the LLM only fills in well-defined steps. It's more predictable, testable and auditable.

**3. Command vs conditional edges: when do you use which?**
Use `Command` when the node that does the work also makes the routing decision and should write state atomically. Examples: the router's LLM verdict, a human's resume value. Use conditional edges when routing is a *policy over state* (retry vs give up vs approve) that you want as a separate pure function. I use both on purpose.

**4. How does self-correction work, and how do you stop infinite loops?**
Validation and DB errors are written to state. The routing function sends control back to `generate_sql`, which puts the previous SQL and the *exact* error in the prompt. `attempts` increments in `generate_sql`, and the router gives up at `MAX_ATTEMPTS`. The cap is enforced by code, not by asking the LLM.

**5. Why EXPLAIN before executing?**
It catches bad columns, tables and functions at compile time without running anything. That makes it cheap feedback for the retry loop, and it avoids executing a possibly heavy query just to learn it was wrong.

**6. Why not let an LLM judge whether SQL is safe?**
LLM judges are probabilistic, cost money, add latency, and can be prompt-injected by the same input. Safety must be deterministic: sqlglot AST checks plus a read-only connection. LLMs are fine for *quality* judgments, not *security* ones.

**7. Why parse with sqlglot rather than regex?**
A regex on "DELETE" blocks a harmless column `deleted_flag` and misses writes inside a CTE or odd whitespace and casing. The AST knows each node's type. I also reject `exp.Command`, sqlglot's "unknown statement" node, so unfamiliar statements fail closed.

**8. How does interrupt/resume actually work?**
`interrupt(payload)` raises internally, LangGraph saves a checkpoint with the pending interrupt, and `invoke` returns `__interrupt__`. Resuming with `Command(resume=value)` on the same thread re-executes that node *from the beginning*, and this time `interrupt()` returns `value`. So code before `interrupt` must be idempotent.

**9. Why does `prepare_turn` exist?**
The checkpointer restores *all* state for the thread, not just messages. Without a reset, turn 2 would start with turn 1's `attempts=2`, old `sql` and `approved_sql`, and could even skip approval. Only `messages` and `error_history` should carry across turns.

**10. What are reducers and why do two fields have them?**
A reducer defines how an update merges into state. `messages` uses `add_messages`, which appends (and de-dupes by id), so conversation memory accumulates. `error_history` uses `operator.add`, an append-only audit log. Everything else overwrites, which is right for per-turn values.

**11. How do follow-up questions work?**
The `AIMessage` for each answer includes `[SQL used]` plus the SQL. The next turn's SQL prompt includes the last 6 messages, so the model sees the previous query and can modify it (add `WHERE state='TX'`) instead of starting over.

**12. Why limit history to 6 messages?**
Cost and quality. Prompt tokens would otherwise grow linearly with conversation length, and old turns add noise. Six messages is about 3 exchanges, which covers realistic follow-up chains. In production I'd summarise older turns instead of dropping them.

**13. How do you handle LLM API failures?**
A `RetryPolicy` on LLM nodes with exponential backoff, for transient errors only (rate limits, timeouts, 5xx). It deliberately doesn't retry `ValueError`/parse errors, since retrying a deterministic failure just burns money. The API returns 502 if retries are exhausted.

**14. How do you test an LLM agent without calling an LLM?**
A `ScriptedLLM` fake whose `with_structured_output(Schema)` returns a Runnable that pops scripted responses per schema and records every prompt. Tests run against a real seeded SQLite DB, so they assert both behavior ("retried and answered") and what the model was shown ("the retry prompt contained the exact error").

**15. How do you evaluate it?**
Execution accuracy on a 20-question golden set (easy/medium/hard). Run each question on a fresh thread with approvals auto-granted, then compare *result sets*, not SQL text. I also report first-try success, average attempts and p50/p95 latency, so I can see whether accuracy comes from the model or from the retry loop.

**16. What are the weaknesses of your metric?**
Columns are compared independently, so row pairing isn't verified (a test documents it). Rounding to 2dp can fail correct answers that were rounded differently. And 20 questions means a ±10-20 point uncertainty, which is fine for catching regressions but not for fine-grained model comparisons.

**17. How would you scale to a 500-table warehouse?**
The two-step retrieval already helps. Next steps: embed table and column descriptions and retrieve the top-k candidate tables before LLM selection, retrieve similar past verified queries as few-shot examples, and cache schema snippets. Then use Postgres `EXPLAIN` cost for approval rather than table count.

**18. What's the semantic layer for?**
Column names don't encode business definitions. "Revenue" here means `SUM(quantity*unit_price)` excluding canceled orders, not `products.price`. Most text-to-SQL errors in practice are semantically wrong queries, not syntax errors, and business rules plus sample rows (which show `'TX'` not `'Texas'`) prevent many of them.

**19. How is this production-ready, and what's missing?**
It has env-based config, a read-only DB, timeouts, persistence across restarts, a health check, a non-root Docker image and an offline test suite. Missing: auth and per-user table permissions, rate limiting, tracing (LangSmith), cost tracking, PII handling, and a larger human-reviewed eval set.

**20. What was the hardest bug?**
Two showed up only in a live smoke test. A newer Claude model rejects a non-default `temperature`, and it doesn't support forced tool calling, which is LangChain's default structured-output method for Anthropic. The fix was to make temperature optional and switch to `method="json_schema"`. The lesson: fakes test *your* logic, and you still need one live run to test the provider contract. A close second was making a *human-edited* query count as approved: the validator normalises SQL, so I compare normalised forms ([graph.py:187](app/graph.py#L187)).

---

## 6. Your 60-second pitch (memorise this)

> "I built a Text-to-SQL analyst agent: a business user asks a question in English and gets an answer plus the SQL that produced it. It's a LangGraph state machine. It classifies the request, retrieves only the relevant schema in two steps, writes SQL, and validates it with a sqlglot AST checker before running it on a read-only connection with a timeout. If the SQL fails, the exact error goes back to the model, with retries capped at three. If a query joins four or more tables, the graph pauses with an interrupt for human approval, and that pause survives a server restart because state is checkpointed to SQLite. The same checkpointer gives conversation memory, so 'now only for Texas' modifies the previous query. The key design choice is that the LLM never makes safety or control-flow decisions: those are deterministic code. It has 60 offline tests with a scripted fake LLM, and an eval harness that measures execution accuracy by difficulty, first-try success, latency and cost. Sonnet scored 20 out of 20, all first try, at about three cents a question. My honest takeaway is that the golden set is saturated, so the next step is a harder set that actually exercises the retry loop."

---

## 7. Five exercises (do them in order)

Each one touches a different concept. Write a test first, using `ScriptedLLM`.

1. **Add a business rule for price bands (semantic layer).** Add to `BUSINESS_RULES` in [db.py:30](app/db.py#L30): *"Budget products have price < 25, premium products have price >= 200."* Add a golden question "How many premium products are there?" with gold SQL, and confirm `make check-gold` passes. *Concept: prompts and evals.*

2. **Add a `suggest_chart` node (new node + structured output).** Create `ChartSuggestion(chart_type: Literal["bar","line","table","number"], x: str|None, y: str|None)` in schemas.py, a prompt, and a node between `summarize` and END that stores `chart` in state. Wire it with `add_edge("summarize", "suggest_chart")`. Test: script the response and assert `out["chart"]`. *Concept: nodes, edges, state.*

3. **Per-thread cost counter (reducer).** Add `llm_calls: Annotated[int, operator.add]` to `AgentState`, and return `{"llm_calls": 1}` from every LLM node. Check that it accumulates across turns of one thread and *isn't* reset by `prepare_turn`. Stretch: count real tokens with `include_raw=True`. *Concept: reducers.*

4. **Make the approval rule smarter (conditional edge policy).** Also require approval when the SQL has no `WHERE` clause on `orders` *and* no `GROUP BY`, i.e. a raw scan. Put the check in the validator (return a flag) and use it in `validate_sql`. Test both a triggering query and a non-triggering one. *Concept: deterministic policy.*

5. **Time-travel "what if" (checkpoints).** Write a script that runs a question, finds the checkpoint whose `next == ("execute_sql",)`, calls `graph.update_state(that_config, {"sql": "<a different query>"})`, then `graph.invoke(None, new_config)`. Observe the forked answer, and inspect `get_state_history` to see both branches. *Concept: time travel and forking.*

---

## 8. How to study this repo

1. Run `make test` and read [tests/test_graph.py](tests/test_graph.py) first. Each test is a short story of one behavior.
2. Read [app/graph.py](app/graph.py) top to bottom alongside section 3 of this guide.
3. Run `git log --oneline` and check out each commit. Stage 3 (`bbb91f6`) is the straight-line graph, with no retries or routing yet. Diff it against the next commit to see exactly what loops, Command and interrupt add.
4. Read `evals/report.md` (Sonnet scored 20/20 on your subscription). Then run `make eval MODEL=claude-cli:haiku` and explain every failure out loud. Interviewers *love* "here's where my agent fails and why", and a perfect score doesn't give you that story, but a smaller model will.
