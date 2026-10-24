# Multi-Agent Systems

## Anthropic's research system
Anthropic's multi-agent research system uses an orchestrator-worker design: a lead agent plans and starts parallel subagents, and a separate citation agent attaches sources. On Anthropic's internal research evaluation it beat a single Claude Opus 4 agent by 90.2 percent. Multi-agent runs used about 15 times as many tokens as a normal chat, and token usage alone explained about 80 percent of the performance variance on the BrowseComp benchmark.

## The counter-argument
Cognition's essay "Don't Build Multi-Agents" (June 2025) argues that agents which do not share full context make conflicting implicit decisions. It recommends a single-threaded agent that compresses its history for long tasks.

## When it helps
A Google and MIT scaling study (December 2025) found multi-agent designs gained up to 80.9 percent on tasks that split into parallel parts but lost 39 to 70 percent on sequential tasks. Independent agents amplified errors about 17 times, versus about 4 times when a central orchestrator checked their work.

## Why they fail
The MAST taxonomy (Cemri et al., 2025) analysed more than 1,600 traces from seven frameworks and grouped 14 failure modes into system design issues, inter-agent misalignment and task verification. The most common single failure modes were step repetition, reasoning-action mismatch and not knowing when to stop. Adding a verification step was one of the most effective fixes.
