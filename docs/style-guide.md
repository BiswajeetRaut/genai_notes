---
name: style-guide
---

# Writing style guide for this course

This file is the contract every module in this repo follows. If you (human or agent) are
writing or editing a module, match this structure so the course reads as one coherent book,
not 30 disconnected blog posts.

## File layout per module

Each module lives at `partX/NN-topic-name/README.md` (leaf topics under Advanced Production RAG
follow the same shape one level deeper). Structure the README as:

1. **Title + 1-paragraph "why this matters"** — plain language, no jargon yet.
2. **Learning objectives** — 3-6 bullet points, "By the end of this module you can...".
3. **Prerequisites** — links to earlier modules a reader should already know (use relative
   markdown links, e.g. `[LangChain Fundamentals](../02-langchain-fundamentals/README.md)`).
4. **Core concepts** — the theory, built up from first principles. Assume the reader knows
   everything in earlier modules but nothing in this one. Define every acronym on first use.
5. **Architecture / flow diagram(s)** — at least one Mermaid diagram per module (see below).
6. **Scenario walkthrough** — a concrete, named scenario ("You're building a support bot for
   an e-commerce company with 200k SKUs...") that motivates the concept and is referenced again
   in the code example. Reuse recurring personas across the course where natural (see below).
7. **Code example** — runnable-looking Python, using LangChain/LangGraph/Pydantic as the
   default stack (see conventions below). Keep examples focused — illustrate the concept, not
   a full production app. Include comments only where the *why* isn't obvious from the code.
8. **Production pitfalls / gotchas** — a bulleted "what breaks in the real world" section.
   This is what separates beginner notes from expert notes — always include it.
9. **Key takeaways** — 3-6 bullets, the TL;DR.
10. **Exercises** — 2-4 practice prompts, no answers given (this is a course, not a solutions
    manual).
11. **Next module** link at the very bottom.

## Recurring scenario personas (reuse these across modules for continuity)

- **"Northwind Support Copilot"** — a customer-support RAG/agent bot for a fictional
  e-commerce company (Northwind). Used for RAG, agents, memory, guardrails modules.
- **"Aster Health"** — a fictional healthcare/insurance company used when we need a scenario
  with strict compliance, PII, and security requirements (guardrails, security, multi-tenant).
- **"Ledger" case study** — a fictional fintech doing high-scale document ingestion (10M+
  filings/contracts) — used across Part 3 system design modules as the running capstone system.

Reusing these means a reader who has been through the whole course recognizes the system by
Part 3 instead of re-learning a new fictional company every module.

## Diagram conventions (Mermaid)

- Use ` ```mermaid ` fenced code blocks — GitHub and most markdown viewers render these natively.
- Prefer `flowchart TD` (top-down) for pipelines/architectures, `sequenceDiagram` for
  request/response and multi-actor flows, `erDiagram` for data models, `stateDiagram-v2` for
  agent/workflow state machines.
- Label every arrow with what flows across it (e.g. `-->|top-k chunks|`), not just a bare arrow.
- Keep diagrams under ~15 nodes; split into two diagrams rather than one dense one.

## Code conventions

- Default stack: **Python 3.11+, LangChain + LangGraph, Pydantic v2** for structured
  data/config. Use `langchain-openai` or `langchain-anthropic` style imports generically;
  don't hardcode a specific vendor as "the" answer unless the module is explicitly about
  model/vendor choice.
- Type-hint everything. Use Pydantic `BaseModel` for any structured schema.
- Examples should be self-contained enough to read top-to-bottom without external files, but
  do not need to be copy-paste-executable (omit API keys, use placeholders like
  `"YOUR_API_KEY"`).
- Prefer small, runnable-looking snippets (20-60 lines) over sprawling files.

## Tone

- Beginner-to-expert progression: Part 1 assumes zero GenAI background but real software
  engineering experience. Part 2 assumes Part 1. Part 3 assumes Part 1+2 and some distributed
  systems background.
- Explain the *why* before the *how*. Every pattern should answer "what breaks without this?"
- No filler, no hype language ("revolutionary", "game-changing"). Direct and precise.
