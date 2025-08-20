# Claude Code Workflow for Complex Architectural Projects

## The Problem
When implementing complex statistical models or probabilistic systems, the translation from architectural concept to implementation often breaks down due to:
- Loss of key conceptual understanding between planning and implementation phases
- Agent handoff issues where context is lost
- Hallucination or divergence from fundamental architectural choices

## Recommended Workflow

### Phase 1: Architecture Documentation (Enhanced)
Instead of just using the code-architect agent, create a more robust handoff document:

```markdown
# Architecture Specification v1.0

## Core Concepts
- **Fundamental Principle**: [Describe the mathematical/statistical foundation]
- **Key Invariants**: [What must always be true]
- **Critical Constraints**: [What absolutely cannot change]

## Implementation Requirements
### MUST HAVE (Non-negotiable)
- [ ] Requirement 1 with specific details
- [ ] Requirement 2 with validation criteria

### SHOULD HAVE (Important but flexible)
- [ ] Feature 1 with rationale

## Test Cases for Validation
1. **Input**: X → **Expected Output**: Y
2. **Edge Case**: Z → **Expected Behavior**: W

## Red Flags (Signs of Divergence)
- If the implementation does X instead of Y, STOP
- If these core types/structures are missing, STOP
```

### Phase 2: Incremental Implementation Strategy

#### Step 1: Core Model Validation
Before full implementation, create a minimal proof-of-concept:
```bash
# First, implement ONLY the core statistical/probabilistic model
# No UI, no optimization, just the mathematical foundation
```

Ask Claude Code:
"Implement ONLY the core [statistical model name] with these exact specifications:
- Input shape: [specific]
- Output shape: [specific]  
- Core algorithm: [paste pseudocode or equations]
Create a simple test that validates the output matches expected values."

#### Step 2: Layered Enhancement
Add complexity in layers, validating each:
1. Core model → Test → Validate
2. Add data preprocessing → Test → Validate
3. Add optimization → Test → Validate
4. Add UI/API → Test → Validate

### Phase 3: Agent Coordination Best Practices

#### Using Multiple Agents Effectively

1. **Context Preservation Between Agents**
```markdown
# Agent Handoff Document
## From: code-architect
## To: code-implementation
## Critical Context:
- The probabilistic model uses [specific distribution]
- The key innovation is [specific technique]
- DO NOT change [specific architectural choice]
```

2. **Validation Checkpoints**
After each agent completes work:
```bash
# Run validation script
npm run validate:architecture
# or
python validate_model.py
```

### Phase 4: Prompt Engineering for Complex Models

#### Effective Initial Prompts

**Instead of:**
"Build a probabilistic model that evaluates processes"

**Use:**
```
I need to implement a [specific model type] with these exact specifications:

MATHEMATICAL FOUNDATION:
- Distribution: [e.g., Gaussian, Poisson]
- Parameters: [list with ranges]
- Update equations: [provide exact formulas]

CRITICAL REQUIREMENTS:
1. The model MUST maintain [invariant]
2. The output MUST be [constraint]
3. The algorithm MUST follow [specific steps]

VALIDATION:
Given input [X], the output should be approximately [Y ± tolerance]

Start by implementing ONLY the core model class with a test case.
```

#### Extended Thinking for Complex Problems
Trigger deeper analysis:
```
"Think step-by-step about this statistical model implementation. 
Consider:
1. Mathematical correctness
2. Numerical stability
3. Edge cases
4. Performance implications
Think harder about potential issues with the approach."
```

### Phase 5: Recovery from Divergence

When Claude Code goes off track:

1. **Immediate Stop and Reset**
```
"STOP. The current implementation has diverged from the architecture.
The core requirement was [X] but you're implementing [Y].
Let's reset to the last correct state and fix this specific issue."
```

2. **Focused Correction**
```
"Focus ONLY on fixing [specific component].
The correct behavior is [expected].
Do not modify any other parts of the system."
```

3. **Reference Implementation**
```
"Here's a reference implementation in [language/pseudocode]:
[paste code]
Translate this EXACTLY to our current framework, maintaining all logic."
```

## Practical Example Workflow

```bash
# 1. Create architecture with validation
claude code "Create an architecture document for a Hidden Markov Model with:
- 3 hidden states
- Gaussian emissions
- Viterbi decoding
Include test cases with expected outputs"

# 2. Implement core model only
claude code "Implement ONLY the HMM class from the architecture.
Include the forward algorithm.
Create a test with known input/output."

# 3. Validate before proceeding
claude code "Run the test and verify the forward algorithm produces
expected probabilities within 0.001 tolerance"

# 4. Layer additional features
claude code "Now add the Viterbi algorithm to the existing HMM class.
Do not modify the forward algorithm.
Add a test for Viterbi."

# 5. Final integration
claude code "Now create the API wrapper for the HMM model.
The core HMM class should not be modified."
```

## Key Takeaways

1. **Explicit is better than implicit** - Provide exact formulas, not descriptions
2. **Test early and often** - Validate core logic before adding complexity  
3. **Layer complexity gradually** - Don't implement everything at once
4. **Preserve context religiously** - Document critical decisions that must not change
5. **Use validation checkpoints** - Stop and verify at each major step
6. **Provide reference implementations** - Give exact code when precision is critical

## When to Use Each Agent

- **code-architect**: Initial design, not detailed implementation
- **feature-analyzer**: Understanding existing patterns before modification  
- **code-implementation**: ONLY after architecture is crystal clear
- **code-reviewer**: After each major implementation phase

Remember: The more complex the model, the more you need to break it down into verifiable chunks. Claude Code excels at incremental, well-specified tasks but can drift on large, ambiguous ones.