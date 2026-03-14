# Blurt — Core Idea

## The One-Line Pitch

Blurt is an AI companion that catches everything rattling around in your head and quietly makes sense of it for you.

---

## The Problem

People with busy minds — the ones with 47 browser tabs, three half-used notes apps, and a graveyard of abandoned planners — don't fail at productivity because they lack tools. They fail because every tool demands they _organize before they think_.

You have an idea in the shower. By the time you've opened an app, picked a category, set a priority, chosen a due date, and filed it in a project — the idea is gone. Or worse, you just don't bother.

Traditional productivity apps treat your brain like a filing cabinet. Blurt treats it like a conversation.

---

## The Insight

**Capture and organization are two fundamentally different cognitive tasks.** Forcing them to happen simultaneously is hostile to how most people's brains work.

What if getting something out of your head was as simple as saying it out loud — and an AI that _knows you_ figured out the rest? Not a chatbot that asks twenty clarifying questions. Not an assistant that forgets you exist between sessions. An AI that listens, classifies, connects, remembers, and gets smarter about your life every single day.

---

## What Blurt Is

Blurt is an AI-first companion for thought capture. One input — voice or text — and the AI handles everything else.

You say: _"Need to return those jeans at Macy's, also remind me to call Sarah about the Q2 deck before Thursday."_

Blurt silently:

- Creates a task: "Return jeans at Macy's"
- Creates a reminder: "Call Sarah about Q2 deck" (before Thursday)
- Recognizes "Sarah" as your coworker (already in your entity graph)
- Connects "Q2 deck" to the project it already knows about
- Detects mild anticipation in your tone
- Writes all of this to memory so it's smarter next time

You didn't categorize anything. You didn't set a priority. You just... blurted.

---

## Who It's For

Anyone who thinks faster than they can organize.

People whose minds produce a constant stream of tasks, ideas, reminders, observations, and half-formed plans — and who need a system that meets them where they are, not where a productivity framework says they should be.

This includes people with ADHD, creative professionals, founders, students, parents, and anyone who has ever said "I need to write that down" and then didn't.

Blurt doesn't require a methodology. No GTD, no second brain, no bullet journal. It just catches what falls out of your head and does something useful with it.

---

## The Core Loop

```
Dump → Classify → Connect → Surface → Act → Learn
```

1. **Dump** — Say it or type it. No structure required.
2. **Classify** — AI identifies intent in ~100ms: task, event, reminder, idea, journal entry, update, or question.
3. **Connect** — AI extracts people, places, projects, organizations. Links them to what it already knows. Detects emotional state.
4. **Surface** — When you need something to do, Blurt picks _one thing_ — the right thing for your energy, mood, schedule, and priorities right now.
5. **Act** — Do it, skip it, break it down, postpone it, let it go. Every option is judgment-free.
6. **Learn** — Every interaction feeds memory. Completed tasks, skipped tasks, mood patterns, timing preferences, entity relationships — it all compounds.

After 30 days, Blurt knows you crash on Thursday afternoons, that "the deck" means Q2 planning, that Sarah is your manager, and that you do your best creative work before 10 AM.

---

## The Seven Intents

Every input maps to one of seven types:

| Intent       | What it means                      | Example                                   |
| ------------ | ---------------------------------- | ----------------------------------------- |
| **TASK**     | Something to do                    | "Pick up groceries"                       |
| **EVENT**    | Something with a specific time     | "Dentist at 3pm Friday"                   |
| **REMINDER** | A nudge at a future time           | "Remind me to call Mom tonight"           |
| **IDEA**     | A thought to save, not act on      | "What if we used serverless for the API?" |
| **JOURNAL**  | Emotional processing or reflection | "Today was rough, really overwhelmed"     |
| **UPDATE**   | A change to something existing     | "Actually, the Macy's trip is done"       |
| **QUESTION** | Asking Blurt about your own data   | "When did I last talk about the Q2 deck?" |

The user never picks an intent. AI classifies silently. Confident (>85%)? Just process. Uncertain? Confirm. No idea? Default to TASK and flag it.

---

## The Memory Architecture (The Moat)

Three tiers of memory working together. No competitor has this.

### Tier 1: Working Memory

What Blurt knows about _right now_. Current mood, energy, what you just said, what task you're on. Resets every session, rebuilds from recent observations.

### Tier 2: Episodic Memory

Everything Blurt has _ever noticed_. Every blurt writes observations: mood, intent, entities mentioned, time of day, what you did with the last task surfaced. Append-only, never deleted, compressed over time. Raw material for pattern detection — "You always feel anxious when you mention the Q2 deck."

### Tier 3: Semantic Memory

The map of your world. People, projects, places, organizations — and relationships between them. Sarah _works with_ Jake. Q2 deck _is part of_ the launch project. Relationships strengthen with co-mentions, weaken with absence.

### The Compound Effect (30 days of normal use)

- ~240 thoughts captured and classified
- ~50 entities with a relationship graph
- ~240 mood observations across your circadian rhythm
- ~200 task completion/skip/dismiss behavioral signals
- Personalized energy calibration and effort estimates

**This is the moat.** Competitors can copy features. They cannot copy a personal knowledge graph built from months of behavioral data. Switching cost grows with every interaction.

---

## How Blurt Surfaces Tasks

The task engine scores every eligible task across 21 factors including:

- **Time of day + energy**: Morning? Surface stale tasks. Afternoon crash? Only tiny tasks.
- **Mood alignment**: High fear/sadness? Easy wins only. High anticipation? Bring on challenges.
- **Calendar awareness**: 30 min before a meeting? Only short tasks.
- **Entity relevance**: Talking about Q2 deck all week? Related tasks get boosted.
- **Behavioral learning**: Weights adjust over time based on what you actually complete vs. skip (Thompson Sampling, on-device, zero API cost).

Blurt doesn't show you a to-do list. It shows you _the one thing you're most likely to actually do right now_.

---

## The Emotional Layer

Blurt detects emotion (Plutchik's 8 primary emotions, intensity 0-3, valence/arousal dimensions) and uses it to directly affect behavior:

- **High fear/sadness**: Only surface tiny and small tasks.
- **High anticipation/joy**: Surface challenging tasks.
- **Low arousal**: Prefer shorter tasks.
- **Strong negative emotion**: Blurt speaks up and offers journal coaching.
- **Over time**: Maps emotional patterns — Thursday crashes, Monday surges, post-meeting stress.

The goal isn't to manage emotions. It's to respect them. A productivity tool that ignores how you feel fails when you need it most.

---

## When Blurt Speaks (and When It Doesn't)

Blurt defaults to silent. Most AI assistants are chatty. Blurt is not.

- **High confidence (>85%)**: Silent toast. "Got it — task added."
- **Medium confidence (60-85%)**: Brief confirmation. "Looks like a reminder. That right?"
- **Low confidence (<60%)**: Default to TASK, flag it.
- **Strong emotion detected**: Speaks up with empathy.
- **Duplicate/pattern detected**: "You mentioned something similar last week."

AI should speak when it has something valuable to say, not to prove it's there.

---

## Principles

1. **Zero friction** — Just dump. AI classifies.
2. **One task at a time** — Never overwhelm.
3. **Anti-streak** — Celebrate wins, never punish absence.
4. **Shame-proof** — No overdue counters, no guilt language.
5. **Burst-friendly** — 30-second sessions, not 30-minute planning.
6. **AI-first** — Single input, AI routes everything.
7. **Memory compounds** — Every interaction makes Blurt smarter.
8. **Never fabricate** — If Blurt doesn't know, it says so.

---

## Free vs. Premium ($9.99/month)

### Free (generous, never crippled)

- 15 blurts/day, 5 breakdowns, 3 journal prompts
- Unlimited voice, reminders, updates
- Entity extraction always on (memory always builds)
- All capture modes, calendar auto-scheduling

### Premium

- Unlimited AI interactions
- QUESTION intent (ask Blurt about your own data)
- Semantic search across all blurts
- Morning briefing, pattern nudges
- Notification timing optimization
- 90-day mood timeline
- Google Calendar sync, bidirectional voice

### Philosophy

Free tier is fully functional. Premium adds _depth_ — search, briefings, patterns. Core product never feels broken without paying.

---

## Business Case

### Cost Per User

- **Free (~15 blurts/day)**: ~$0.003/day ($0.09/month)
- **Premium (~30 blurts/day)**: ~$0.009/day ($0.27/month)
- **At $9.99/month premium**: 97%+ margin on AI costs

### Two-Model Strategy

- **Gemini 2.5 Flash-Lite** (70% of calls): Classification, entity extraction, mood detection. Cheap.
- **Gemini 2.5 Flash** (30% of calls): Breakdowns, journaling, insights, Q&A. Smarter.

### Why It Works Economically

Expensive AI only fires for complex reasoning. Cheap AI handles high-volume classification. Voice runs on-device at zero cost. Notification optimization runs on-device at zero cost. AI costs are negligible relative to subscription revenue.

### Monetization Path

- Premium: $9.99/mo or $79.99/year
- Lifetime: $149
- Therapist/coach tier: $19.99/mo (PDF reports, data export, client dashboard)
- No ads, no tracking, no data selling

---

## Competitive Landscape

| Competitor   | Price     | Strength                            | Gap                                        |
| ------------ | --------- | ----------------------------------- | ------------------------------------------ |
| Saner.AI     | $8-20/mo  | Chat-first AI, email triage         | No persistent memory, no entity graph      |
| rivva        | Free/paid | Energy forecasting via Apple Health | No zero-friction capture, no observations  |
| Motion       | $29/mo    | AI Employee SuperApp                | No personal memory, clinical UX, expensive |
| Goblin Tools | Free      | Viral task breakdown                | Single purpose, no memory, no learning     |
| Neurolist    | Free/paid | AI list maker, voice                | No entity memory, no behavioral learning   |
| Nudgly       | Free      | Shame-free messaging                | No AI intelligence at all                  |

### Why Blurt Wins

No existing app combines:

1. AI-first intelligence (single input, AI classifies everything)
2. Memory that compounds (entity graph + observations grow daily)
3. Zero-friction capture (voice or text, no organizing)
4. Energy-aware surfacing (mood gates what gets shown)
5. Behavioral learning (every interaction adjusts the system)

The moat is accumulated intelligence that grows with use and cannot be exported.

---

## The Vision

**Today**: Catches thoughts, classifies them, surfaces the right task at the right time.

**Six months**: Knows your world — people, projects, patterns, rhythms. Notices Thursday crashes and pre-10AM creativity. Understands "the deck" and "Sarah" without explanation.

**One year**: The closest thing to a personal assistant who has worked with you for a decade. Understands _why_ you tell it things, _when_ you'll act, and _how_ to present information for your brain.

The aspiration isn't a better to-do list. It's an AI companion that genuinely knows you — earned through thousands of small observations, not a profile form.

**Dump thoughts. Blurt figures it out.**
