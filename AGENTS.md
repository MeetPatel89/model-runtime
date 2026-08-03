# Repository Instructions

These instructions apply to the entire repository. This is the Codex-native
counterpart of the always-applied rules in `.cursor/rules/`. When a shared rule
changes, update both representations in the same change so Cursor and Codex do
not receive conflicting guidance.

## Codex instruction placement

- Keep project-wide Codex instructions in the root `AGENTS.md`.
- Add a nested `AGENTS.md` only when a subtree genuinely needs more specific
  instructions; nested rules must not silently contradict project-wide rules.
- Use descriptive Markdown headings and include enough context for an agent to
  apply a rule without guessing its purpose.
- Cursor-specific `.mdc` files remain under `.cursor/rules/`; do not relocate
  them when adding their Codex equivalent.

## Python object-oriented design

Apply these principles proportionally. Do not introduce interfaces, factories,
repositories, or dependency-injection machinery without a concrete need. Use
the simplest design that preserves responsibilities, encapsulation,
testability, and dependency direction.

Before changing Python code:

1. Identify each affected module's and class's responsibility.
2. Preserve existing architectural boundaries unless they are defective.
3. Consider extension, coupling, and testability rather than placing behavior
   in whichever file produces the smallest patch.

Implementation requirements:

- Give each class one clear responsibility and primary reason to change.
- Keep orchestration separate from domain logic and keep networking/framework
  integration outside domain values.
- Do not create generic Manager, Helper, Utils, or Service classes that collect
  unrelated behavior.
- Keep implementation details private and expose behavior rather than mutable
  internals. Prefer immutable value objects where practical.
- Make external dependencies explicit and inject clients, clocks, file systems,
  and similar infrastructure at boundaries. Avoid service locators and hidden
  global state.
- Prefer composition. Use shallow inheritance only for a genuine substitutable
  “is-a” relationship, and introduce protocols only at meaningful variation or
  isolation boundaries.
- Keep interfaces small; do not force consumers to implement unused methods.
- Put validation and invariants with the domain concept that owns them, making
  invalid states difficult to represent.
- Minimize shared mutation and make state transitions explicit.
- Raise domain-specific exceptions at abstraction boundaries, translate
  infrastructure errors, and preserve their exception context.
- Make behavior testable without live networks or other external systems. Test
  observable behavior rather than private implementation details.
- When a requested change exposes a design problem, make the smallest coherent
  refactor that improves the boundary, preserve public behavior unless a change
  is requested, and report meaningful tradeoffs.

Before finalizing Python changes, review single responsibility, dependency
direction, encapsulation, cohesion/coupling, composition, explicit dependencies,
invariants, unnecessary abstractions, testability, and consistency with the
existing architecture.

## README maintenance

Treat `README.md` as part of the product and keep it synchronized with the code
that exists in the current branch.

For every task:

1. Perform a README impact check.
2. Update the README when code, configuration, dependencies, commands,
   architecture, public modules, supported inputs/outputs, extension points,
   tests, data handling, safety boundaries, or limitations change.
3. Remove stale claims when behavior is renamed, removed, or superseded.
4. Avoid README churn for changes with no reader-visible effect.
5. Verify documentation against source, configuration, tests, and executable
   output rather than memory.

README content must be evidence-based. Keep its opening skimmable; use commands
that work from the repository root; keep paths, model IDs, environment variables,
and defaults exact; label limitations and future work; never invent metrics,
adoption, maintainers, links, licenses, or roadmap commitments; never include
secrets or private data. Document observable behavior without exposing or asking
for hidden chain-of-thought.

Before completion, confirm that setup and examples match the implementation,
architecture includes material components and omits removed ones, test commands
are current, data/failure behavior is accurate, referenced files exist, and the
Markdown contains no accidental placeholders.
