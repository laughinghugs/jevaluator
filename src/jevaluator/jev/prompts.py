"""Prompt templates for Jev's evaluation questions.

Every judgment template is phrased as a yes/no question so the backend can
return P(Yes) as the score. Templates use ``str.format`` fields.
"""

JEV_SYSTEM = (
    "You are Jev, a strict and impartial evaluator of retrieval-augmented generation (RAG) "
    "systems. You are given material from a RAG interaction and a single yes/no evaluation "
    "question about it. Judge only what the question asks, using only the material provided. "
    "Do not reward fluency or length; do not use outside knowledge unless the question says so. "
    "When the evidence is incomplete or ambiguous, lean towards No."
)

FAITHFULNESS_CLAIM = """\
Retrieved context:
<context>
{context}
</context>

Claim from the generated answer:
<claim>
{claim}
</claim>

Evaluation question: Is the claim fully supported by the retrieved context, such that it can \
be directly inferred from the context without outside knowledge?"""

ANSWER_RELEVANCE = """\
User question:
<question>
{question}
</question>

Generated answer:
<answer>
{answer}
</answer>

Evaluation question: Does the answer directly address the user's question and stay on topic? \
(Judge relevance only, not factual accuracy. An evasive, off-topic, or non-committal answer \
is not relevant.)"""

CONTEXT_RELEVANCE = """\
User question:
<question>
{question}
</question>

Retrieved passage:
<passage>
{passage}
</passage>

Evaluation question: Does this passage contain information that is useful for answering the \
user's question?"""

CONTEXT_RECALL_STATEMENT = """\
Retrieved context:
<context>
{context}
</context>

Statement from the reference answer:
<statement>
{statement}
</statement>

Evaluation question: Can this statement be attributed to the retrieved context, i.e. does the \
context contain the information needed to support it?"""

ANSWER_CORRECTNESS = """\
User question:
<question>
{question}
</question>

Reference (ground-truth) answer:
<reference>
{reference}
</reference>

Generated answer:
<answer>
{answer}
</answer>

Evaluation question: Is the generated answer factually correct with respect to the reference \
answer, conveying its key facts without contradicting it? (Wording may differ; extra detail is \
acceptable only if it does not contradict the reference.)"""

CLAIM_EXTRACTION = """\
Break the following answer into a list of atomic, self-contained factual claims. Each claim \
must be understandable on its own (resolve pronouns), and together the claims must cover every \
factual assertion in the answer. Ignore greetings, hedges and statements with no factual content.

Question (for context only):
<question>
{question}
</question>

Answer:
<answer>
{answer}
</answer>

Output one claim per line with no numbering or bullets. If the answer makes no factual claims, \
output exactly NONE."""


def format_contexts(contexts: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
