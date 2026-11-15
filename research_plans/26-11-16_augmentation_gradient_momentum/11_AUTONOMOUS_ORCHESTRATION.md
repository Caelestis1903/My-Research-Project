# Autonomous Orchestration Contract

Pro is a one-time planner and exits after plan upload. Parent Codex: `6.1-sol-exhigh`. Astra: `6-Astra-exhigh`, XHigh, fresh native Subagent context each cycle. Parent owns implementation/training/report/context/dispatch. Astra only researches and writes plans.

Before training, create a native Codex child with independent context and verify actual model/effort metadata. Test child reads AGENT.md, EXP_AI_CONTEXT.md, Experimental_Open_Questions.md and a designated test file. Any failed verification => `AGENT_DISPATCH_BLOCKED`, stop before training. Never substitute parent model.

After experiment/report/review/main-context commit, launch a fresh Astra. It reads current main and raw results, proposes >=3 distinct mechanisms across view construction, gradient processing and optimizer recurrence, critically checks assumptions/alternatives, and commits its plan. Parent checks hard completeness, not subjective familiarity. PASS launches next Codex; REVISE returns once; STOP/BLOCKED recorded. Max two Astra-driven cycles. No user confirmation.
